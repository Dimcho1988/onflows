# sync.py
import os, time, math, json, requests
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from db import (
    get_conn, upsert_activity, upsert_metrics, upsert_daily_stress,
    get_last_activity_start
)
from strava_streams import build_timeseries_1hz
from normalize import add_percent_columns
from indices import compute_tiz, compute_daily_stress, aggregate_daily_stress, compute_hr_drift

import psycopg2, psycopg2.extras

STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")

def _now_ts() -> int:
    return int(time.time())

def _refresh(refresh_token: str):
    r = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def _get_token_rows():
    sql = """
    select a.id as athlete_id, a.strava_athlete_id,
           t.access_token, t.refresh_token, t.expires_at
    from athletes a
    join strava_tokens t on t.athlete_id = a.id
    """
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(sql)
        return cur.fetchall()

def _api(token: str, path: str, params=None):
    h = {"Authorization": f"Bearer {token}"}
    r = requests.get("https://www.strava.com/api/v3" + path, headers=h, params=params or {}, timeout=60)
    if r.status_code == 429:
        # rate limited
        reset = r.headers.get("X-RateLimit-Reset")
        if reset:
            sleep_for = max(0, int(reset) - _now_ts() + 2)
            time.sleep(sleep_for)
            r = requests.get("https://www.strava.com/api/v3" + path, headers=h, params=params or {}, timeout=60)
    r.raise_for_status()
    return r.json()

def _get_streams(token: str, activity_id: int):
    keys = "time,heartrate,velocity_smooth,watts,altitude,distance,cadence"
    params = {"keys": keys, "key_by_type": "true"}
    return _api(token, f"/activities/{activity_id}/streams", params)

def _ensure_token(tok_row):
    token = tok_row["access_token"]
    if _now_ts() >= int(tok_row["expires_at"]) - 30:
        new_tok = _refresh(tok_row["refresh_token"])
        # save back to DB
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                update strava_tokens set access_token=%s, refresh_token=%s, expires_at=%s, updated_at=now()
                where athlete_id=%s
                """,
                (new_tok["access_token"], new_tok["refresh_token"], new_tok["expires_at"], tok_row["athlete_id"]),
            )
        token = new_tok["access_token"]
    return token

def _to_dt(s: str):
    # "2025-07-26T18:47:59Z" -> aware datetime
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)

def sync_one_athlete(tok_row, cfg):
    athlete_id = tok_row["athlete_id"]
    token = _ensure_token(tok_row)

    # вземи последната дата
    last_dt = get_last_activity_start(athlete_id)  # may be None
    params = {"per_page": 50}
    if last_dt is not None:
        # Strava filter "after" е UNIX секунди
        params["after"] = int(last_dt.replace(tzinfo=timezone.utc).timestamp())

    # лист активности
    acts = _api(token, "/athlete/activities", params)
    if not acts:
        return 0

    new_count = 0
    for a in acts:
        act_id = a["id"]
        meta = {
            "id": act_id,
            "name": a.get("name"),
            "type": a.get("type"),
            "start_date": _to_dt(a.get("start_date")),
            "distance_m": a.get("distance"),
            "moving_time_s": a.get("moving_time"),
            "avg_hr": a.get("average_heartrate"),
            "avg_speed_mps": a.get("average_speed"),
            "raw": json.dumps(a),
        }
        upsert_activity(meta, athlete_id)

        # streams → 1 Hz
        try:
            streams = _get_streams(token, act_id)
            df = build_timeseries_1hz(streams)
            if df.empty:
                continue

            # normalize + indices
            zones_cfg = cfg["zones"]
            df = add_percent_columns(
                df,
                cs_run_kmh=float(cfg["defaults"]["CS_run_kmh"]),
                cp_bike_w=float(cfg["defaults"]["CP_bike_w"]),
                hrmax=float(cfg["defaults"]["HRmax"]),
                zones_cfg=zones_cfg,
            )
            tiz = compute_tiz(df, mode="hr", zones_cfg=zones_cfg).to_dict()
            stress = compute_daily_stress(df)
            drift = compute_hr_drift(df)
            upsert_metrics(act_id, tiz, stress, None if (drift is None or np.isnan(drift)) else float(drift))

            # дневен стрес
            day = meta["start_date"].date()
            upsert_daily_stress(athlete_id, day, stress)
            new_count += 1

            # малко спане за да пазим rate limit
            time.sleep(0.4)

        except Exception as e:
            print(f"[WARN] activity {act_id} failed: {e}")

    return new_count

def main():
    import yaml
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rows = _get_token_rows()
    total = 0
    for r in rows:
        n = sync_one_athlete(r, cfg)
        print(f"athlete_id={r['athlete_id']} synced {n} new activities")
        total += n
        time.sleep(0.5)
    print(f"Done. total_new={total}")

if __name__ == "__main__":
    main()
