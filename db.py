# db.py
import os
from contextlib import contextmanager

try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    _HAS_ST = False

import psycopg2
import psycopg2.extras as pgx


def _get_secret(key: str, default: str = None) -> str:
    if _HAS_ST and "secrets" in dir(st) and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)


DB_URL = _get_secret("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL not found in secrets or environment!")


@contextmanager
def get_conn():
    conn = psycopg2.connect(DB_URL)
    try:
        yield conn
    finally:
        conn.close()


def upsert_athlete(strava_athlete_id: int, firstname: str = None, lastname: str = None) -> int:
    """Returns internal athlete id."""
    sql = """
    INSERT INTO athletes (strava_athlete_id, firstname, lastname)
    VALUES (%s, %s, %s)
    ON CONFLICT (strava_athlete_id) DO UPDATE
      SET firstname = EXCLUDED.firstname,
          lastname  = EXCLUDED.lastname
    RETURNING id;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (strava_athlete_id, firstname, lastname))
            rid = cur.fetchone()[0]
            conn.commit()
            return rid


def upsert_token(athlete_id: int, access_token: str, refresh_token: str, expires_at: int, scope: str = None):
    sql = """
    INSERT INTO strava_tokens (athlete_id, access_token, refresh_token, expires_at, scope)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (athlete_id) DO UPDATE
       SET access_token = EXCLUDED.access_token,
           refresh_token = EXCLUDED.refresh_token,
           expires_at = EXCLUDED.expires_at,
           scope = COALESCE(EXCLUDED.scope, strava_tokens.scope),
           updated_at = NOW();
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (athlete_id, access_token, refresh_token, expires_at, scope))
            conn.commit()


def get_all_tokens():
    """Return list of dicts {athlete_id, access_token, refresh_token, expires_at}."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
            cur.execute("SELECT * FROM strava_tokens;")
            return list(cur.fetchall())


def last_activity_start(athlete_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(start_date) FROM activities WHERE athlete_id=%s;", (athlete_id,))
            row = cur.fetchone()
            return row[0]  # may be None


def upsert_activity(activity: dict, athlete_id: int):
    """
    Expects minimal Strava activity fields + precomputed values:
    id, name, type, start_date, distance, moving_time, average_heartrate, average_speed
    """
    sql = """
    INSERT INTO activities (
        id, athlete_id, name, type, start_date,
        distance_m, moving_time_s, avg_hr, avg_speed_mps, raw
    )
    VALUES (%(id)s, %(athlete_id)s, %(name)s, %(type)s, %(start_date)s,
            %(distance_m)s, %(moving_time_s)s, %(avg_hr)s, %(avg_speed_mps)s, %(raw)s)
    ON CONFLICT (id) DO UPDATE
      SET name = EXCLUDED.name,
          type = EXCLUDED.type,
          start_date = EXCLUDED.start_date,
          distance_m = EXCLUDED.distance_m,
          moving_time_s = EXCLUDED.moving_time_s,
          avg_hr = EXCLUDED.avg_hr,
          avg_speed_mps = EXCLUDED.avg_speed_mps,
          raw = EXCLUDED.raw;
    """
    rec = {
        "id": activity["id"],
        "athlete_id": athlete_id,
        "name": activity.get("name"),
        "type": activity.get("type"),
        "start_date": activity.get("start_date"),  # aware timestamp
        "distance_m": activity.get("distance", 0.0),
        "moving_time_s": activity.get("moving_time", 0),
        "avg_hr": activity.get("average_heartrate"),
        "avg_speed_mps": activity.get("average_speed"),
        "raw": pgx.Json(activity),
    }
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, rec)
            conn.commit()
def upsert_activity_stream(activity_id: int, df: pd.DataFrame) -> int:
    """
    Записва 1 Hz stream данни за дадена активност.
    Очаква df с индекс second и колони: lat, lon, altitude, heartrate, cadence, watts, velocity_smooth.
    Връща броя записани редове.
    """
    sql = """
        INSERT INTO activity_streams
        (activity_id, second, lat, lon, altitude, heartrate, cadence, power, speed)
        VALUES %s
        ON CONFLICT DO NOTHING;
    """

    rows = []
    for sec, row in df.iterrows():
        rows.append((
            activity_id,
            int(sec),
            row.get("lat"),
            row.get("lon"),
            row.get("altitude"),
            row.get("heartrate"),
            row.get("cadence"),
            row.get("watts"),
            row.get("velocity_smooth"),
        ))

    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows, page_size=1000)
        conn.commit()

    return len(rows)


def upsert_metrics(activity_id: int, tiz_minutes: dict, stress: float, hr_drift: float | None):
    sql = """
    INSERT INTO activity_metrics (
        activity_id,
        tiz_z1_minutes, tiz_z2_minutes, tiz_z3_minutes, tiz_z4_minutes, tiz_z5_minutes,
        stress, hr_drift
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (activity_id) DO UPDATE
      SET tiz_z1_minutes=EXCLUDED.tiz_z1_minutes,
          tiz_z2_minutes=EXCLUDED.tiz_z2_minutes,
          tiz_z3_minutes=EXCLUDED.tiz_z3_minutes,
          tiz_z4_minutes=EXCLUDED.tiz_z4_minutes,
          tiz_z5_minutes=EXCLUDED.tiz_z5_minutes,
          stress=EXCLUDED.stress,
          hr_drift=EXCLUDED.hr_drift,
          computed_at = NOW();
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                activity_id,
                tiz_minutes.get("Z1", 0.0),
                tiz_minutes.get("Z2", 0.0),
                tiz_minutes.get("Z3", 0.0),
                tiz_minutes.get("Z4", 0.0),
                tiz_minutes.get("Z5", 0.0),
                stress,
                hr_drift,
            ))
            conn.commit()


def upsert_daily_stress(athlete_id: int, day, stress: float):
    sql = """
    INSERT INTO daily_stress (athlete_id, day, stress)
    VALUES (%s, %s, %s)
    ON CONFLICT (athlete_id, day) DO UPDATE
      SET stress = EXCLUDED.stress;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (athlete_id, day, stress))
            conn.commit()
