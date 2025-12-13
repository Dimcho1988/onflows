import time
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st
import numpy as np
import pandas as pd
from supabase import create_client, Client

# -------------------------------------------------
# Globals
# -------------------------------------------------
SUPABASE_URL = None
SUPABASE_KEY = None
STRAVA_CLIENT_ID = None
STRAVA_CLIENT_SECRET = None

supabase: Client | None = None


# -------------------------------------------------
# Init clients from Streamlit secrets
# -------------------------------------------------
def init_clients(st_app):
    global SUPABASE_URL, SUPABASE_KEY
    global STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET
    global supabase

    STRAVA_CLIENT_ID = st_app.secrets["strava"]["client_id"]
    STRAVA_CLIENT_SECRET = st_app.secrets["strava"]["client_secret"]

    SUPABASE_URL = st_app.secrets["supabase"]["url"]
    SUPABASE_KEY = st_app.secrets["supabase"]["service_role_key"]

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# -------------------------------------------------
# HR zone ranked (match counts by speed zone, assign by lowest HR)
# -------------------------------------------------
def add_hr_zone_ranked(seg_df: pd.DataFrame, zone_col: str = "zone", hr_col: str = "hr_mean") -> pd.DataFrame:
    """
    Creates hr_zone_ranked:
      - Counts segments per speed zone (zone_col) within this activity
      - Sorts segments by hr_mean ascending (NaN at end)
      - Assigns hr_zone_ranked labels with same counts as speed zones
    """
    df = seg_df.copy()
    if df is None or df.empty:
        return df

    if zone_col not in df.columns:
        df["hr_zone_ranked"] = None
        return df

    if hr_col not in df.columns:
        df["hr_zone_ranked"] = None
        return df

    # counts per speed-zone
    zone_counts = df[zone_col].fillna("UNK").value_counts().to_dict()

    # prefer common ordering
    ordered_zones = [z for z in ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"] if z in zone_counts]
    total_needed = int(sum(zone_counts.get(z, 0) for z in ordered_zones))

    # sort by HR (ascending)
    df["_hr_sort"] = pd.to_numeric(df[hr_col], errors="coerce")
    df = df.sort_values(["_hr_sort"], ascending=True, na_position="last").reset_index(drop=True)

    # build label list
    labels: list[str] = []
    for z in ordered_zones:
        labels.extend([z] * int(zone_counts.get(z, 0)))

    hr_ranked = [None] * len(df)
    for i in range(min(total_needed, len(df))):
        hr_ranked[i] = labels[i]

    df["hr_zone_ranked"] = hr_ranked
    df = df.drop(columns=["_hr_sort"])
    return df


# -------------------------------------------------
# OAuth helpers
# -------------------------------------------------
def exchange_code_for_tokens(auth_code: str) -> dict:
    token_url = "https://www.strava.com/oauth/token"

    data = {
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "grant_type": "authorization_code",
        "code": auth_code,
    }

    resp = requests.post(token_url, data=data, timeout=20)
    resp.raise_for_status()
    token_info = resp.json()

    athlete_resp = requests.get(
        "https://www.strava.com/api/v3/athlete",
        headers={"Authorization": f"Bearer {token_info['access_token']}"},
        timeout=20,
    )
    athlete_resp.raise_for_status()

    token_info["athlete"] = athlete_resp.json()
    return token_info


def get_strava_access_token(refresh_token: str) -> str:
    token_url = "https://www.strava.com/oauth/token"

    data = {
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }

    resp = requests.post(token_url, data=data, timeout=20)
    resp.raise_for_status()
    return resp.json()["access_token"]


# -------------------------------------------------
# Strava API
# -------------------------------------------------
def fetch_activities_since(access_token: str, after_ts: int) -> list[dict]:
    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {"Authorization": f"Bearer {access_token}"}

    all_activities = []
    page = 1
    per_page = 50

    while True:
        params = {"after": after_ts, "page": page, "per_page": per_page}
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        chunk = r.json()

        if not chunk:
            break

        all_activities.extend(chunk)
        page += 1

    return all_activities


def fetch_activity_streams(access_token: str, strava_activity_id: int) -> dict:
    url = f"https://www.strava.com/api/v3/activities/{strava_activity_id}/streams"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "keys": "time,distance,altitude,heartrate,latlng",
        "key_by_type": "true",
    }

    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


# -------------------------------------------------
# Sport mapping
# -------------------------------------------------
def map_sport_group(strava_sport_type: str | None) -> str | None:
    ski_types = {"NordicSki", "BackcountrySki", "RollerSki"}
    run_types = {"Run", "TrailRun"}
    walk_types = {"Walk", "Hike"}

    if strava_sport_type in ski_types:
        return "ski"
    if strava_sport_type in run_types:
        return "run"
    if strava_sport_type in walk_types:
        return "walk"
    return None


# -------------------------------------------------
# Incremental sync helper (SAFE)
# -------------------------------------------------
def get_last_activity_start_date(user_id: int):
    try:
        res = (
            supabase.table("activities")
            .select("start_date")
            .eq("user_id", user_id)
            .order("start_date", desc=True)
            .limit(1)
            .execute()
        )

        data = res.data or []
        if not data:
            return None

        return datetime.fromisoformat(data[0]["start_date"].replace("Z", "+00:00"))
    except Exception:
        return None


# -------------------------------------------------
# Supabase: upsert activity summary
# -------------------------------------------------
def upsert_activity_summary(act: dict, user_id: int) -> int | None:
    sport_group = map_sport_group(act.get("sport_type"))
    if sport_group is None:
        return None

    row = {
        "user_id": user_id,
        "strava_activity_id": act["id"],
        "name": act.get("name"),
        "sport_type": act.get("sport_type"),
        "sport_group": sport_group,
        "start_date": act.get("start_date"),
        "timezone": act.get("timezone"),
        "distance_m": act.get("distance"),
        "moving_time_s": act.get("moving_time"),
        "elapsed_time_s": act.get("elapsed_time"),
        "total_elevation_gain_m": act.get("total_elevation_gain"),
        "avg_speed_mps": act.get("average_speed"),
        "max_speed_mps": act.get("max_speed"),
        "has_heartrate": act.get("has_heartrate", False),
        "avg_heartrate": act.get("average_heartrate"),
        "max_heartrate": act.get("max_heartrate"),
    }

    row = {k: v for k, v in row.items() if v is not None}

    res = (
        supabase.table("activities")
        .upsert(row, on_conflict="strava_activity_id")
        .execute()
    )

    data = res.data or []
    if not data:
        return None

    return data[0]["id"]


# -------------------------------------------------
# Rolling training set: last N activities per sport (excluding current)
# -------------------------------------------------
def get_last_n_activity_ids_excluding(
    user_id: int,
    sport_group: str,
    exclude_activity_id: int,
    n: int = 30
) -> list[int]:
    # take a bit more then filter
    res = (
        supabase.table("activities")
        .select("id")
        .eq("user_id", user_id)
        .eq("sport_group", sport_group)
        .order("start_date", desc=True)
        .limit(n + 10)
        .execute()
    )
    rows = res.data or []
    ids = [r["id"] for r in rows if r.get("id") is not None and r["id"] != exclude_activity_id]
    return ids[:n]


# -------------------------------------------------
# Fetch training segments (30s) from DB
# -------------------------------------------------
def fetch_training_segments_30s(
    activity_ids: list[int],
    sport_group: str,
    segment_seconds: int,
    model_key: str,
) -> pd.DataFrame:
    if not activity_ids:
        return pd.DataFrame()

    res = (
        supabase.table("activity_segments")
        .select("*")
        .in_("activity_id", activity_ids)
        .eq("sport_group", sport_group)
        .eq("segment_seconds", segment_seconds)
        .eq("model_key", model_key)
        .execute()
    )

    df = pd.DataFrame(res.data or [])
    if df.empty:
        return df

    for c in ["slope_pct", "v_kmh_raw", "v_glide", "k_glide", "dt_s", "d_m", "hr_mean"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["valid_basic", "speed_spike"]:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    return df


# -------------------------------------------------
# Fetch training segments (7s) from DB (ski only)
# -------------------------------------------------
def fetch_training_segments_7s(
    activity_ids: list[int],
    model_key: str,
) -> pd.DataFrame:
    if not activity_ids:
        return pd.DataFrame()

    res = (
        supabase.table("activity_segments_7s")
        .select("*")
        .in_("activity_id", activity_ids)
        .eq("sport_group", "ski")
        .eq("model_key", model_key)
        .execute()
    )

    df = pd.DataFrame(res.data or [])
    if df.empty:
        return df

    for c in ["slope_pct", "v_kmh_raw", "dt_s", "d_m", "hr_mean"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["valid_basic", "speed_spike"]:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    return df


# -------------------------------------------------
# Fit rolling models from DB
# -------------------------------------------------
def fit_run_walk_slope_poly_from_segments(train_seg: pd.DataFrame):
    """
    Fit slope polynomial F(s) from rolling segments (run/walk).
    Returns np.poly1d or None.
    """
    from run_walk_pipeline import fit_slope_poly

    if train_seg is None or train_seg.empty:
        return None

    df = train_seg.copy()
    m = (
        df["valid_basic"]
        & (~df["speed_spike"])
        & df["slope_pct"].between(-15.0, 15.0)
        & (df["v_kmh_raw"] > 3.0)
        & (df["v_kmh_raw"] < 30.0)
    )
    df = df.loc[m, ["slope_pct", "v_kmh_raw"]].dropna()
    if df.empty:
        return None

    V0 = df.loc[df["slope_pct"].between(-1.0, 1.0), "v_kmh_raw"].mean()
    if not np.isfinite(V0) or V0 <= 0:
        return None

    df["F_raw"] = V0 / df["v_kmh_raw"]
    train_df = df[["slope_pct", "F_raw"]].copy()
    return fit_slope_poly(train_df)


def fit_glide_poly_from_7s(train7: pd.DataFrame):
    """
    Fit glide polynomial v_model(slope) from rolling 7s segments (ski).
    Uses consecutive downhills rule (get_glide_training_segments).
    Returns np.poly1d or None.
    """
    from ski_pipeline import get_glide_training_segments, fit_glide_poly

    if train7 is None or train7.empty:
        return None

    train7 = train7.sort_values(["activity_id", "seg_idx"]).reset_index(drop=True)
    train_glide = get_glide_training_segments(train7)
    return fit_glide_poly(train_glide)


def fit_ski_slope_poly_from_30s(train30: pd.DataFrame, glide_poly_for_training, damp_glide: float):
    """
    Fit ski slope polynomial F(s) using rolling 30s segments.
    Builds temporary v_glide_tmp using per-activity k_glide derived from glide_poly_for_training (optional),
    then fits slope using ski_pipeline.get_slope_training_data_out + fit_slope_poly.
    Returns np.poly1d or None.
    """
    from ski_pipeline import get_slope_training_data_out, fit_slope_poly

    if train30 is None or train30.empty:
        return None

    df = train30.copy()
    df = df[df["valid_basic"] & (~df["speed_spike"])].copy()
    df = df.dropna(subset=["slope_pct", "v_kmh_raw"])
    if df.empty:
        return None

    # Build per-activity k_glide map from downhills (approx) to get v_glide_tmp base
    if glide_poly_for_training is not None:
        k_map = {}
        m_down = (df["slope_pct"] <= -5.0)
        down = df.loc[m_down, ["activity_id", "slope_pct", "v_kmh_raw"]].dropna()
        for aid, g in down.groupby("activity_id"):
            s_mean = float(g["slope_pct"].mean())
            v_real = float(g["v_kmh_raw"].mean())
            if v_real <= 0:
                continue
            v_model = float(glide_poly_for_training(s_mean))
            if v_model <= 0:
                continue
            k_raw = v_model / v_real
            k_clip = max(0.90, min(1.25, k_raw))
            k_final = 1.0 + float(damp_glide) * (k_clip - 1.0)
            k_map[int(aid)] = float(k_final)

        df["k_glide_tmp"] = df["activity_id"].map(k_map).fillna(1.0)
    else:
        df["k_glide_tmp"] = 1.0

    df["v_glide_tmp"] = df["v_kmh_raw"] * df["k_glide_tmp"]

    V_flat_ref = df.loc[df["slope_pct"].between(-1.0, 1.0), "v_glide_tmp"].mean()
    if not np.isfinite(V_flat_ref) or V_flat_ref <= 0:
        return None

    tmp = df.copy()
    tmp["v_glide"] = tmp["v_glide_tmp"]
    slope_train = get_slope_training_data_out(tmp, float(V_flat_ref))
    return fit_slope_poly(slope_train)


# -------------------------------------------------
# Supabase: UPSERT 30s segments (IDEMPOTENT)
# -------------------------------------------------
def insert_segments_30s(
    user_id: int,
    sport_group: str,
    activity_id: int,
    seg_df,
    segment_seconds: int,
    model_key: str,
) -> int:
    if seg_df is None or getattr(seg_df, "empty", False):
        return 0

    wanted = [
        "seg_idx", "t_start", "t_end", "dt_s", "d_m", "slope_pct",
        "v_kmh_raw", "valid_basic", "speed_spike",
        "k_glide", "v_glide",
        "v_flat_eq", "v_flat_eq_cs",
        "delta_v_plus_kmh", "r_kmh", "tau_s", "hr_mean",
        "zone",
        "hr_zone_ranked",  # NEW
    ]

    df = seg_df.copy()
    for c in wanted:
        if c not in df.columns:
            df[c] = None

    df["user_id"] = user_id
    df["sport_group"] = sport_group
    df["activity_id"] = activity_id
    df["segment_seconds"] = segment_seconds
    df["model_key"] = model_key

    for col in ["t_start", "t_end"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)

    df = df.replace({np.nan: None})

    rows = df[
        ["user_id", "sport_group", "activity_id", "segment_seconds", "model_key"] + wanted
    ].to_dict(orient="records")

    total = 0
    for i in range(0, len(rows), 1000):
        chunk = rows[i:i + 1000]
        supabase.table("activity_segments").upsert(
            chunk,
            on_conflict="activity_id,sport_group,segment_seconds,seg_idx,model_key",
        ).execute()
        total += len(chunk)

    return total


# -------------------------------------------------
# Supabase: UPSERT 7s segments (ski glide training history)
# -------------------------------------------------
def insert_segments_7s(
    user_id: int,
    activity_id: int,
    seg7_df,
    model_key: str,
) -> int:
    if seg7_df is None or getattr(seg7_df, "empty", False):
        return 0

    wanted = [
        "seg_idx", "t_start", "t_end", "dt_s", "d_m",
        "slope_pct", "v_kmh_raw", "hr_mean",
        "valid_basic", "speed_spike",
    ]

    df = seg7_df.copy()
    for c in wanted:
        if c not in df.columns:
            df[c] = None

    df["user_id"] = user_id
    df["activity_id"] = activity_id
    df["sport_group"] = "ski"
    df["model_key"] = model_key

    for col in ["t_start", "t_end"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)

    df = df.replace({np.nan: None})

    rows = df[
        ["user_id", "activity_id", "sport_group", "model_key"] + wanted
    ].to_dict(orient="records")

    total = 0
    for i in range(0, len(rows), 1000):
        chunk = rows[i:i + 1000]
        supabase.table("activity_segments_7s").upsert(
            chunk,
            on_conflict="activity_id,seg_idx,model_key",
        ).execute()
        total += len(chunk)

    return total


# -------------------------------------------------
# Full sync + process pipeline (rolling models)
# -------------------------------------------------
def sync_and_process_from_strava(
    token_info: dict,
    process_ski_activity_30s,      # must return (seg30, seg7)
    process_run_walk_activity,
    params_by_sport: dict,
    days_back_if_empty: int = 200,
):
    athlete = token_info["athlete"]
    athlete_id = athlete["id"]

    refresh_token = token_info["refresh_token"]
    access_token = get_strava_access_token(refresh_token)

    last_dt = get_last_activity_start_date(athlete_id)

    if last_dt:
        after_ts = int(last_dt.timestamp()) - 60
    else:
        after_ts = int(
            (datetime.now(timezone.utc) - timedelta(days=days_back_if_empty)).timestamp()
        )

    activities = fetch_activities_since(access_token, after_ts)
    activities = sorted(activities, key=lambda a: a.get("start_date") or "")

    new_acts = 0
    total_segments = 0

    for act in activities:
        local_activity_id = upsert_activity_summary(act, user_id=athlete_id)
        if local_activity_id is None:
            continue

        sport_group = map_sport_group(act.get("sport_type"))
        if sport_group is None:
            continue

        streams = fetch_activity_streams(access_token, act["id"])
        streams_norm = {k: {"data": v.get("data", [])} for k, v in streams.items()}

        params = params_by_sport[sport_group]
        params_proc = dict(params)
        model_key = params_proc.pop("model_key", "unknown")

        # rolling train ids (last 30 of that sport, excluding current)
        train_ids = get_last_n_activity_ids_excluding(
            user_id=athlete_id,
            sport_group=sport_group,
            exclude_activity_id=local_activity_id,
            n=30,
        )

        # --- RUN/WALK rolling slope
        if sport_group in ("run", "walk"):
            train30 = fetch_training_segments_30s(
                activity_ids=train_ids,
                sport_group=sport_group,
                segment_seconds=30,
                model_key=model_key,
            )
            slope_poly_global = fit_run_walk_slope_poly_from_segments(train30)

            seg30 = process_run_walk_activity(
                act_row={"id": local_activity_id, "start_date": act.get("start_date")},
                streams=streams_norm,
                slope_poly_override=slope_poly_global,
                **params_proc,
            )

            # NEW: add HR ranked zones (uses existing speed zones + hr_mean)
            seg30 = add_hr_zone_ranked(seg30)

            total_segments += insert_segments_30s(
                user_id=athlete_id,
                sport_group=sport_group,
                activity_id=local_activity_id,
                seg_df=seg30,
                segment_seconds=30,
                model_key=model_key,
            )
            new_acts += 1
            time.sleep(0.25)
            continue

        # --- SKI rolling glide(7s) + slope(30s)
        if sport_group == "ski":
            # glide poly from 7s history
            train7 = fetch_training_segments_7s(train_ids, model_key=model_key)
            glide_poly_global = fit_glide_poly_from_7s(train7)

            # slope poly from 30s history (uses glide_poly only to build v_glide_tmp base)
            train30 = fetch_training_segments_30s(
                activity_ids=train_ids,
                sport_group="ski",
                segment_seconds=30,
                model_key=model_key,
            )
            slope_poly_global = fit_ski_slope_poly_from_30s(
                train30=train30,
                glide_poly_for_training=glide_poly_global,
                damp_glide=params_proc.get("DAMP_GLIDE", 1.0),
            )

            # IMPORTANT: ski pipeline must return (seg30, seg7)
            seg30, seg7 = process_ski_activity_30s(
                act_row={"id": local_activity_id, "start_date": act.get("start_date")},
                streams=streams_norm,
                glide_poly_override=glide_poly_global,
                slope_poly_override=slope_poly_global,
                **params_proc,
            )

            # NEW: add HR ranked zones (uses existing speed zones + hr_mean)
            seg30 = add_hr_zone_ranked(seg30)

            # store 7s for future rolling glide
            insert_segments_7s(
                user_id=athlete_id,
                activity_id=local_activity_id,
                seg7_df=seg7,
                model_key=model_key,
            )

            # store 30s for analysis & rolling slope
            total_segments += insert_segments_30s(
                user_id=athlete_id,
                sport_group="ski",
                activity_id=local_activity_id,
                seg_df=seg30,
                segment_seconds=30,
                model_key=model_key,
            )

            new_acts += 1
            time.sleep(0.25)
            continue

    return new_acts, total_segments
