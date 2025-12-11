# strava_sync.py
import time
from datetime import datetime, timedelta, timezone

import requests
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
    resp = requests.post(token_url, data=data, timeout=10)
    resp.raise_for_status()
    token_info = resp.json()

    athlete_resp = requests.get(
        "https://www.strava.com/api/v3/athlete",
        headers={"Authorization": f"Bearer {token_info['access_token']}"},
        timeout=10,
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
    resp = requests.post(token_url, data=data, timeout=10)
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
        r = requests.get(url, headers=headers, params=params, timeout=10)
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
    r = requests.get(url, headers=headers, params=params, timeout=10)
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
    res = (
        supabase.table("activities")
        .select("start_date")
        .eq("user_id", user_id)
        .order("start_date", desc=True)
        .limit(1)
        .execute()
    )
    data = res.data
    if not data:
        return None
    return datetime.fromisoformat(data[0]["start_date"])


def upsert_activity(act: dict, user_id: int) -> int:
    sport_group = map_sport_group(act.get("sport_type"))
    row = {
        "user_id": user_id,
        "strava_activity_id": act["id"],
        "name": act.get("name"),
        "sport_type": act.get("sport_type"),
        "sport_group": sport_group,
        "distance": act.get("distance"),
        "moving_time": act.get("moving_time"),
        "elapsed_time": act.get("elapsed_time"),
        "start_date": act.get("start_date"),
        "timezone": act.get("timezone"),
        "average_speed": act.get("average_speed"),
        "max_speed": act.get("max_speed"),
        "has_heartrate": act.get("has_heartrate", False),
        "average_heartrate": act.get("average_heartrate"),
        "max_heartrate": act.get("max_heartrate"),
    }

    res = (
        supabase.table("activities")
        .upsert(row, on_conflict="strava_activity_id")
        .execute()
    )
    return res.data[0]["id"]


def save_streams(activity_id: int, streams: dict):
    supabase.table("activity_streams").delete().eq("activity_id", activity_id).execute()
    rows = []
    for stream_type, payload in streams.items():
        rows.append(
            {
                "activity_id": activity_id,
                "stream_type": stream_type,
                "data": payload.get("data", []),
            }
        )
    if not rows:
        return 0
    res = supabase.table("activity_streams").insert(rows).execute()
    return len(res.data)


def sync_from_strava(token_info: dict):
    athlete = token_info["athlete"]
    athlete_id = athlete["id"]

    refresh_token = token_info["refresh_token"]
    access_token = get_strava_access_token(refresh_token)

    last_dt = get_last_activity_start_date(athlete_id)
    if last_dt:
        after_ts = int(last_dt.timestamp()) - 60
    else:
        after_ts = int((datetime.now(timezone.utc) - timedelta(days=10)).timestamp())

    activities = fetch_activities_since(access_token, after_ts)

    new_acts = 0
    total_stream_rows = 0

    for act in activities:
        local_id = upsert_activity(act, user_id=athlete_id)
        new_acts += 1
        streams = fetch_activity_streams(access_token, act["id"])
        total_stream_rows += save_streams(local_id, streams)
        time.sleep(0.3)

    return new_acts, total_stream_rows
