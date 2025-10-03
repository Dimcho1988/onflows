# sync.py
"""
GitHub Actions entry-point:
- refresh tokens for all athletes
- pull new activities since last stored
- fetch streams -> 1Hz df
- normalize -> zones
- compute metrics (TIZ/Stress/HR drift)
- write to Supabase (Postgres)

Run locally:
  DB_URL=... STRAVA_CLIENT_ID=... STRAVA_CLIENT_SECRET=... python sync.py
"""

import os
import time
import math
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd

from db import (
    get_all_tokens,
    upsert_token,
    upsert_activity,
    upsert_metrics,
    upsert_daily_stress,
    last_activity_start,
)

# our project modules
from normalize import add_percent_columns
from indices import compute_tiz, compute_daily_stress, compute_hr_drift
from strava_streams import build_timeseries_1hz  # expects Strava stream JSON -> 1Hz DataFrame


def get_secret(key: str) -> str:
    # GH Actions / Streamlit secrets both export as env vars here
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing env var: {key}")
    return val


STRAVA_CLIENT_ID = get_secret("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = get_secret("STRAVA_CLIENT_SECRET")


def refresh_access_token(refresh_token: str):
    url = "https://www.strava.com/oauth/token"
    data = {
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    r = requests.post(url, data=data, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j["access_token"], j["refresh_token"], j["expires_at"], j.get("athlete")


def list_activities(access_token: str, after_dt: datetime | None):
    headers = {"Authorization": f"Bearer {access_token}"}
    url = "https://www.strava.com/api/v3/athlete/activities"
    params = {"per_page": 100, "page": 1}
    if after_dt:
        # Strava expects seconds since epoch UTC
        params["after"] = int(after_dt.replace(tzinfo=timezone.utc).timestamp())

    while True:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        arr = r.json()
        if not arr:
            break
        for a in arr:
            yield a
        params["page"] += 1


def fetch_streams(activity_id: int, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    keys = "time,heartrate,velocity_smooth,watts,altitude,distance,cadence"
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    params = {"keys": keys, "key_by_type": "true"}
    r = requests.get(url, headers=headers, params=params, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def compute_and_store(activity: dict, athlete_internal_id: int, access_token: str, hrmax: int, cs_run_kmh: float, cp_bike_w: float):
    """Pull streams -> 1Hz -> normalize -> indices -> DB"""
    # Upsert raw activity (basic fields)
    act_rec = {
        "id": activity["id"],
        "name": activity.get("name"),
        "type": activity.get("sport_type") or activity.get("type"),
        "start_date": pd.to_datetime(activity["start_date_local"]).tz_localize(None).to_pydatetime(),
        "distance": activity.get("distance") or 0.0,
        "moving_time": activity.get("moving_time") or 0,
        "average_heartrate": activity.get("average_heartrate"),
        "average_speed": activity.get("average_speed"),
        "raw": activity,
    }
    upsert_activity(act_rec, athlete_internal_id)

    # Try to fetch streams (skip if not available)
    js = fetch_streams(activity["id"], access_token)
    if not js:
        # no streams -> set simple stress proxy from avgHR if available
        # still insert dummy metrics
        tiz = {"Z1": 0, "Z2": 0, "Z3": 0, "Z4": 0, "Z5": 0}
        stress = 0.0
        hr_drift = None
        upsert_metrics(activity["id"], tiz, stress, hr_drift)
        # daily stress as 0
        day = pd.to_datetime(act_rec["start_date"]).date()
        upsert_daily_stress(athlete_internal_id, day, 0.0)
        return

    # 1Hz dataframe
    df_1hz = build_timeseries_1hz(js)  # must include 'time' (sec), hr, speed, watts...
    if df_1hz is None or df_1hz.empty:
        return

    # normalize to %CP/%CS/HR zones
    df_norm = add_percent_columns(
        df_1hz.copy(),
        cs_run_kmh=cs_run_kmh,    # running critical speed [km/h]
        cp_bike_w=cp_bike_w,      # cycling CP [W]
        hrmax=hrmax,
    )

    # METRICS
    tiz = compute_tiz(df_norm, mode="hr")  # or "pct_cs"/"pct_cp" by sport
    stress = compute_daily_stress(df_norm)  # simple per-second quadratic proxy or HR zones
    hr_drift = compute_hr_drift(df_norm)

    # Save metrics
    upsert_metrics(activity["id"], tiz, stress, hr_drift)

    # Save DAILY stress (aggregate by activity date)
    day = pd.to_datetime(act_rec["start_date"]).date()
    upsert_daily_stress(athlete_internal_id, day, stress)


def main():
    tokens = get_all_tokens()
    if not tokens:
        print("No athletes/tokens in DB yet. Log in at your Streamlit app first.")
        return

    # Defaults – можеш да ги изкараш в таблица athletes по-късно
    HRMAX_DEFAULT = int(os.getenv("HRMAX_DEFAULT", "190"))
    CS_RUN_KMH_DEFAULT = float(os.getenv("CS_RUN_KMH_DEFAULT", "18.0"))
    CP_BIKE_W_DEFAULT = float(os.getenv("CP_BIKE_W_DEFAULT", "260.0"))

    for tk in tokens:
        athlete_id = tk["athlete_id"]
        print(f"\n=== Athlete #{athlete_id} ===")

        # refresh if needed
        now = int(time.time())
        if tk["expires_at"] <= now + 60:
            print("Refreshing access token…")
            access_token, new_refresh, expires_at, athlete_obj = refresh_access_token(tk["refresh_token"])
            upsert_token(athlete_id, access_token, new_refresh, expires_at)
        else:
            access_token = tk["access_token"]

        # last stored date
        after_dt = last_activity_start(athlete_id)
        if after_dt:
            after_dt = after_dt.replace(tzinfo=timezone.utc) - timedelta(minutes=1)
        print("Pulling activities after:", after_dt)

        pulled = 0
        for act in list_activities(access_token, after_dt):
            pulled += 1
            try:
                compute_and_store(
                    activity=act,
                    athlete_internal_id=athlete_id,
                    access_token=access_token,
                    hrmax=HRMAX_DEFAULT,
                    cs_run_kmh=CS_RUN_KMH_DEFAULT,
                    cp_bike_w=CP_BIKE_W_DEFAULT,
                )
                print("Stored activity", act["id"], act.get("name"))
            except Exception as e:
                print("Failed activity", act.get("id"), e)

        print(f"Athlete {athlete_id}: pulled {pulled} new activities.")

    print("Sync finished.")


if __name__ == "__main__":
    main()
