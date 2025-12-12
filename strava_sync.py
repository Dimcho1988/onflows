import time
from datetime import datetime, timedelta, timezone

import requests
from dateutil.parser import isoparse
from supabase import create_client, Client

SUPABASE_URL = None
SUPABASE_KEY = None
STRAVA_CLIENT_ID = None
STRAVA_CLIENT_SECRET = None

supabase: Client | None = None


def init_clients(st):
    global SUPABASE_URL, SUPABASE_KEY, STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, supabase
    STRAVA_CLIENT_ID = st.secrets["strava"]["client_id"]
    STRAVA_CLIENT_SECRET = st.secrets["strava"]["client_secret"]
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["service_role_key"]
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------- OAuth helpers ----------
def exchange_code_for_tokens(auth_code: str):
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
    token_info = resp.json()
    return token_info["access_token"]


# ---------- Strava API ----------
def fetch_activities_since(access_token: str, after_ts: int):
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


def fetch_activity_streams(access_token: str, strava_activity_id: int):
    url = f"https://www.strava.com/api/v3/activities/{strava_activity_id}/streams"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "keys": "time,distance,altitude,heartrate,latlng",
        "key_by_type": "true",
    }
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------- Supabase helpers ----------
def map_sport_group(strava_sport_type: str) -> str | None:
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


def get_last_activity_start_date(user_id: int):
    # TEMPORARY BYPASS:
    # table is empty or PostgREST cannot order yet
    return None

def upsert_activity_summary(act: dict, user_id: int) -> int | None:
    sport_group = map_sport_group(act.get("sport_type"))

    # 👉 КРИТИЧНО: пропускаме активности извън ski / run / walk
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

    res = (
        supabase.table("activities")
        .upsert(row, on_conflict="strava_activity_id")
        .execute()
    )

    if not res.data:
        return None

    return res.data[0]["id"]


def upsert_segments_30s(user_id: int, sport_group: str, activity_id: int, seg_df):
    if seg_df is None or seg_df.empty:
        return 0

    wanted = [
        "seg_idx","t_start","t_end","dt_s","d_m","slope_pct",
        "v_kmh_raw","valid_basic","speed_spike",
        "k_glide","v_glide",
        "v_flat_eq","v_flat_eq_cs",
        "delta_v_plus_kmh","r_kmh","tau_s","hr_mean"
    ]
    df = seg_df.copy()
    for c in wanted:
        if c not in df.columns:
            df[c] = None

    df["user_id"] = user_id
    df["sport_group"] = sport_group
    df["activity_id"] = activity_id

    rows = df[["user_id","sport_group","activity_id"] + wanted].to_dict(orient="records")

    total = 0
    for i in range(0, len(rows), 1000):
        chunk = rows[i:i+1000]
        res = supabase.table("activity_segments").upsert(
            chunk,
            on_conflict="activity_id,sport_group,seg_idx"
        ).execute()
        total += len(res.data or [])
    return total


def sync_and_process_from_strava(
    token_info: dict,
    process_ski_activity_30s,
    process_run_walk_activity,
    params_by_sport: dict,
    days_back_if_empty: int = 10,
):
    athlete = token_info["athlete"]
    athlete_id = athlete["id"]

    refresh_token = token_info["refresh_token"]
    access_token = get_strava_access_token(refresh_token)

    last_dt = get_last_activity_start_date(athlete_id)
    if last_dt:
        after_ts = int(last_dt.timestamp()) - 60
    else:
        after_ts = int((datetime.now(timezone.utc) - timedelta(days=days_back_if_empty)).timestamp())

    activities = fetch_activities_since(access_token, after_ts)

    new_acts = 0
    total_segments = 0

    for act in activities:
        sport_group = map_sport_group(act.get("sport_type"))
        local_activity_id = upsert_activity_summary(act, user_id=athlete_id)
        new_acts += 1

        if sport_group is None:
            continue

        streams = fetch_activity_streams(access_token, act["id"])
        # normalize to streams dict: {k: {"data":[...]}}
        streams_norm = {k: {"data": v.get("data", [])} for k, v in streams.items()}

        if sport_group == "ski":
            p = params_by_sport.get("ski", {})
            seg30 = process_ski_activity_30s(act_row={"id": local_activity_id, "start_date": act.get("start_date")}, streams=streams_norm, **p)
            total_segments += upsert_segments_30s(athlete_id, "ski", local_activity_id, seg30)

        elif sport_group in ("run", "walk"):
            p = params_by_sport.get("run", {})
            seg30 = process_run_walk_activity(act_row={"id": local_activity_id, "start_date": act.get("start_date")}, streams=streams_norm, **p)
            total_segments += upsert_segments_30s(athlete_id, sport_group, local_activity_id, seg30)

        time.sleep(0.25)

    return new_acts, total_segments
