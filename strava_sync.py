import time
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st
import numpy as np
from supabase import create_client, Client

# -------------------------------------------------
# Globals (initialized via init_clients)
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
    if resp.status_code >= 400:
        st.error(f"Strava token exchange failed ({resp.status_code})")
        st.write(resp.text)
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
# TEMP: bypass last activity date (first sync)
# -------------------------------------------------
def get_last_activity_start_date(_user_id: int):
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

    # remove None values (safer with DB constraints)
    row = {k: v for k, v in row.items() if v is not None}

    try:
        res = (
            supabase.table("activities")
            .upsert(row, on_conflict="strava_activity_id")
            .execute()
        )
    except Exception as e:
        st.error("UPSERT activities FAILED (raw error):")
        st.write(e)
        st.write("Row:", row)
        raise

    data = res.data or []
    if not data:
        return None

    return data[0]["id"]


# -------------------------------------------------
# Supabase: insert 30s segments (JSON-safe)
# -------------------------------------------------
def insert_segments_30s(user_id: int, sport_group: str, activity_id: int, seg_df) -> int:
    if seg_df is None or seg_df.empty:
        return 0

    wanted = [
        "seg_idx", "t_start", "t_end", "dt_s", "d_m", "slope_pct",
        "v_kmh_raw", "valid_basic", "speed_spike",
        "k_glide", "v_glide",
        "v_flat_eq", "v_flat_eq_cs",
        "delta_v_plus_kmh", "r_kmh", "tau_s", "hr_mean",
    ]

    df = seg_df.copy()

    for c in wanted:
        if c not in df.columns:
            df[c] = None

    df["user_id"] = user_id
    df["sport_group"] = sport_group
    df["activity_id"] = activity_id

    # 🔑 JSON-safe conversions
    for col in ["t_start", "t_end"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x.isoformat() if hasattr(x, "isoformat") else x
            )

    df = df.replace({np.nan: None})

    rows = df[["user_id", "sport_group", "activity_id"] + wanted].to_dict(orient="records")

    total = 0
    for i in range(0, len(rows), 1000):
        chunk = rows[i:i + 1000]
        try:
            supabase.table("activity_segments").insert(chunk).execute()
        except Exception as e:
            st.error("INSERT activity_segments FAILED (raw error):")
            st.write(e)
            st.write("First row:", chunk[0] if chunk else None)
            raise
        total += len(chunk)

    return total


# -------------------------------------------------
# Full sync + process pipeline
# -------------------------------------------------
def sync_and_process_from_strava(
    token_info: dict,
    process_ski_activity_30s,
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

    new_acts = 0
    total_segments = 0

    for act in activities:
        local_activity_id = upsert_activity_summary(act, user_id=athlete_id)
        if local_activity_id is None:
            continue

        sport_group = map_sport_group(act.get("sport_type"))
        if sport_group is None:
            continue

        new_acts += 1

        streams = fetch_activity_streams(access_token, act["id"])
        streams_norm = {k: {"data": v.get("data", [])} for k, v in streams.items()}

        if sport_group == "ski":
            seg30 = process_ski_activity_30s(
                act_row={"id": local_activity_id, "start_date": act.get("start_date")},
                streams=streams_norm,
                **params_by_sport["ski"],
            )
            total_segments += insert_segments_30s(
                athlete_id, "ski", local_activity_id, seg30
            )

        elif sport_group in ("run", "walk"):
            seg30 = process_run_walk_activity(
                act_row={"id": local_activity_id, "start_date": act.get("start_date")},
                streams=streams_norm,
                **params_by_sport["run"],
            )
            total_segments += insert_segments_30s(
                athlete_id, sport_group, local_activity_id, seg30
            )

        time.sleep(0.25)

    return new_acts, total_segments
