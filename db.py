# db.py
import os
import psycopg2
import psycopg2.extras

DB_URL = os.getenv("DB_URL") or os.environ.get("POSTGRES_URL")

def get_conn():
    if not DB_URL:
        raise RuntimeError("DB_URL env var is required (add it in Streamlit/Actions Secrets).")
    return psycopg2.connect(DB_URL)

def upsert_athlete(strava_athlete_id: int, firstname: str = None, lastname: str = None):
    sql = """
    insert into athletes (strava_athlete_id, firstname, lastname)
    values (%s, %s, %s)
    on conflict (strava_athlete_id) do update
      set firstname = excluded.firstname,
          lastname  = excluded.lastname
    returning id;
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (strava_athlete_id, firstname, lastname))
        return cur.fetchone()[0]

def upsert_token(athlete_id: int, access_token: str, refresh_token: str, expires_at: int, scope: str = None):
    sql = """
    insert into strava_tokens (athlete_id, access_token, refresh_token, expires_at, scope)
    values (%s, %s, %s, %s, %s)
    on conflict (athlete_id) do update
      set access_token = excluded.access_token,
          refresh_token = excluded.refresh_token,
          expires_at = excluded.expires_at,
          scope = excluded.scope,
          updated_at = now();
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (athlete_id, access_token, refresh_token, expires_at, scope))

def get_last_activity_start(athlete_id: int):
    sql = "select max(start_date) from activities where athlete_id=%s"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (athlete_id,))
        row = cur.fetchone()
        return row[0]  # may be None

def upsert_activity(act: dict, athlete_id: int):
    sql = """
    insert into activities
      (id, athlete_id, name, type, start_date, distance_m, moving_time_s, avg_hr, avg_speed_mps, raw)
    values
      (%(id)s, %(athlete_id)s, %(name)s, %(type)s, %(start_date)s, %(distance_m)s, %(moving_time_s)s,
       %(avg_hr)s, %(avg_speed_mps)s, %(raw)s)
    on conflict (id) do nothing;
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, act | {"athlete_id": athlete_id})

def upsert_metrics(activity_id: int, tiz: dict, stress: float, hr_drift):
    sql = """
    insert into activity_metrics
      (activity_id, tiz_z1_minutes, tiz_z2_minutes, tiz_z3_minutes, tiz_z4_minutes, tiz_z5_minutes,
       stress, hr_drift)
    values
      (%s,%s,%s,%s,%s,%s,%s,%s)
    on conflict (activity_id) do update
      set tiz_z1_minutes=excluded.tiz_z1_minutes,
          tiz_z2_minutes=excluded.tiz_z2_minutes,
          tiz_z3_minutes=excluded.tiz_z3_minutes,
          tiz_z4_minutes=excluded.tiz_z4_minutes,
          tiz_z5_minutes=excluded.tiz_z5_minutes,
          stress=excluded.stress,
          hr_drift=excluded.hr_drift,
          computed_at=now();
    """
    vals = (
        activity_id,
        float(tiz.get("Z1", 0)), float(tiz.get("Z2", 0)), float(tiz.get("Z3", 0)),
        float(tiz.get("Z4", 0)), float(tiz.get("Z5", 0)),
        float(stress), None if hr_drift is None else float(hr_drift),
    )
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, vals)

def upsert_daily_stress(athlete_id: int, day, stress: float):
    sql = """
    insert into daily_stress(athlete_id, day, stress)
    values (%s, %s, %s)
    on conflict (athlete_id, day) do update
      set stress = excluded.stress;
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (athlete_id, day, float(stress)))
