# -*- coding: utf-8 -*-
import os
import io
import json
import yaml
import time
import pathlib
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

from strava_streams import StravaClient, build_timeseries_1hz
from normalize import add_percent_columns
from indices import compute_tiz, compute_daily_stress, aggregate_daily_stress, compute_acwr, compute_hr_drift
from calibration import suggest_thresholds
from generator import generate_plan

# -------------------- App setup --------------------
st.set_page_config(page_title="onFlows MVP", layout="wide")

# ---------- Config load/save ----------
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(cfg):
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

CFG = load_config()

def init_state():
    d = CFG["defaults"]
    st.session_state.setdefault("HRmax", d["HRmax"])
    st.session_state.setdefault("CS_run_kmh", d["CS_run_kmh"])
    st.session_state.setdefault("CP_bike_w", d["CP_bike_w"])
    st.session_state.setdefault("strava_token_full", None)
    st.session_state.setdefault("activities_cache", None)
    st.session_state.setdefault("meta_log", [])
init_state()

# ---------- Secrets / OAuth helpers ----------
def app_redirect_uri():
    return st.secrets.get("APP_REDIRECT_URI", "http://localhost:8501")

def strava_oauth_url():
    client_id = st.secrets.get("STRAVA_CLIENT_ID", "")
    redirect = app_redirect_uri()
    scope = "activity:read_all"
    return (
        "https://www.strava.com/oauth/authorize"
        f"?client_id={client_id}"
        f"&redirect_uri={redirect}"
        "&response_type=code"
        f"&scope={scope}"
    )

def exchange_code_for_token(code: str) -> dict:
    url = "https://www.strava.com/oauth/token"
    payload = {
        "client_id": st.secrets["STRAVA_CLIENT_ID"],
        "client_secret": st.secrets["STRAVA_CLIENT_SECRET"],
        "code": code,
        "grant_type": "authorization_code",
    }
    r = requests.post(url, data=payload, timeout=30)
    r.raise_for_status()
    return r.json()

# ---------- Token cache & refresh ----------
TOKEN_PATH = pathlib.Path.home() / ".streamlit" / "onflows_strava_token.json"

def save_token_to_disk(token: dict):
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_PATH, "w", encoding="utf-8") as f:
        json.dump(token, f)

def load_token_from_disk():
    if TOKEN_PATH.exists():
        with open(TOKEN_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def refresh_access_token(refresh_token: str) -> dict:
    url = "https://www.strava.com/oauth/token"
    payload = {
        "client_id": st.secrets["STRAVA_CLIENT_ID"],
        "client_secret": st.secrets["STRAVA_CLIENT_SECRET"],
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    r = requests.post(url, data=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def ensure_strava_token():
    """Връща валиден access_token. Ползва session_state + локален кеш и авто-рефреш."""
    tok = st.session_state.get("strava_token_full") or load_token_from_disk()
    if not tok:
        return None
    if time.time() >= float(tok.get("expires_at", 0)) - 30:
        try:
            tok = refresh_access_token(tok["refresh_token"])
            st.session_state["strava_token_full"] = tok
            save_token_to_disk(tok)
        except Exception as e:
            st.warning(f"Изтекъл токен. Нужно е ново вписване в Strava. ({e})")
            return None
    st.session_state["strava_token_full"] = tok
    save_token_to_disk(tok)
    return tok["access_token"]

# ---------- Download helpers ----------
def download_csv_button(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")

def download_excel_button(df: pd.DataFrame, label: str, filename: str):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Plan")
    st.download_button(label, data=buf.getvalue(), file_name=filename,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =========================
# DB Functions (Supabase)
# =========================
import psycopg2
import psycopg2.extras

def _get_conn():
    return psycopg2.connect(st.secrets["DB_URL"])

def _ensure_activities_table():
    ddl = """
    CREATE TABLE IF NOT EXISTS activities (
        id BIGINT PRIMARY KEY,
        name TEXT,
        type TEXT,
        start_date_local TIMESTAMPTZ,
        distance_km DOUBLE PRECISION,
        moving_time_min DOUBLE PRECISION,
        avg_hr DOUBLE PRECISION,
        avg_speed_kmh DOUBLE PRECISION,
        inserted_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()

def save_activities_to_db(df: pd.DataFrame) -> int:
    required = [
        "id", "name", "type", "start_date_local",
        "distance_km", "moving_time_min", "avg_hr", "avg_speed_kmh",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Липсващи колони за запис: {missing}")
        return 0

    _ensure_activities_table()

    rows = df[required].copy()
    if rows["start_date_local"].dtype == "object":
        rows["start_date_local"] = pd.to_datetime(rows["start_date_local"], errors="coerce", utc=True)

    data = [tuple(x) for x in rows.to_numpy()]

    upsert_sql = """
        INSERT INTO activities
        (id,name,type,start_date_local,distance_km,moving_time_min,avg_hr,avg_speed_kmh)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            name=EXCLUDED.name,
            type=EXCLUDED.type,
            start_date_local=EXCLUDED.start_date_local,
            distance_km=EXCLUDED.distance_km,
            moving_time_min=EXCLUDED.moving_time_min,
            avg_hr=EXCLUDED.avg_hr,
            avg_speed_kmh=EXCLUDED.avg_speed_kmh;
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, upsert_sql, data, page_size=500)
        conn.commit()
    return len(data)

# -------------------- Sidebar / Navigation --------------------
st.sidebar.title("onFlows")
page = st.sidebar.radio("Меню", ["Strava", "Индекси", "Генератор", "Настройки"])

# ============================================================
# STRAVA PAGE (във функция, за да няма проблеми с отстъпи)
# ============================================================
def render_strava():
    st.header("Strava – OAuth, активности и 1 Hz таблица")

    # 1) Ако сме върнати с ?code=...
    params = st.query_params
    if "code" in params:
        try:
            token_data = exchange_code_for_token(params["code"])
            st.session_state["strava_token_full"] = token_data
            save_token_to_disk(token_data)
            st.success("✅ Успешен вход в Strava!")
            st.query_params.clear()
        except Exception as e:
            st.error(f"Грешка при обмен на код за токен: {e}")

    # 2) Осигури валиден токен
    access_token = ensure_strava_token()
    if not access_token:
        st.markdown("1) Натисни бутона за вход → одобри достъпа → ще те върне тук.")
        if st.button("🔐 Вход със Strava"):
            st.markdown(f"[Отвори Strava OAuth]({strava_oauth_url()})")
        st.stop()

    client = StravaClient(access_token)

    # 3) Обнови и покажи активности
    if st.button("🔄 Обнови активностите (последни 10)"):
        try:
            acts = client.get_athlete_activities(per_page=10)
            st.session_state["activities_cache"] = acts
            st.success("Заредени са последните 10 активности.")
        except Exception as e:
            st.error(f"Грешка при заявка към Strava: {e}")

    acts = st.session_state.get("activities_cache") or []
    if acts:
        df_acts = pd.DataFrame([{
            "id": a["id"],
            "name": a.get("name"),
            "type": a.get("type"),
            "start_date_local": a.get("start_date_local"),
            "distance_km": round(a.get("distance", 0) / 1000.0, 2),
            "moving_time_min": int(a.get("moving_time", 0) / 60),
            "avg_hr": a.get("average_heartrate", None),
            "avg_speed_kmh": round((a.get("average_speed", 0) or 0) * 3.6, 2),
        } for a in acts])

        st.dataframe(df_acts, use_container_width=True)

        # запис в Supabase
        try:
            n = save_activities_to_db(df_acts)
            st.success(f"Записани/обновени в Supabase: {n} активности.")
        except Exception as e:
            st.warning(f"Записът към Supabase не успя: {e}")
    else:
        st.info("Няма кеширани активности. Натисни „Обнови активностите“.")

    # 4) 1 Hz таблица за конкретна активност
    activity_id = st.text_input("Въведи activity_id за 1 Hz таблица:", placeholder="напр. 1234567890")
    if st.button("⬇️ Дърпай streams и направи 1 Hz"):
        if not activity_id.strip().isdigit():
            st.warning("Моля, въведи валиден activity_id (число).")
            st.stop()
        try:
            streams = client.get_activity_streams(int(activity_id))
            df_1hz = build_timeseries_1hz(streams)
            if df_1hz.empty:
                st.error("Неуспех при генериране на 1 Hz таблица (липсва time stream).")
                st.stop()

            st.success(f"1 Hz таблица: {len(df_1hz)} реда.")
            st.dataframe(df_1hz.head(300), use_container_width=True)

            # Download CSV
            dl = df_1hz.reset_index().rename(columns={"second": "t_sec"})
            download_csv_button(dl, "💾 Download 1Hz CSV", f"activity_{activity_id}_1hz.csv")

            # Предложения за прагове
            rec = suggest_thresholds(df_1hz)
            if rec:
                st.info(f"Предложени прагове (ориентир): {rec}")
        except Exception as e:
            st.error(f"Грешка при дърпане на streams: {e}")

# ============================================================
# INDICES PAGE (съкратено – остави си твоя логика ако искаш)
# ============================================================
def render_indices():
    st.header("Индекси – TIZ, дневен стрес, ACWR, HR drift")
    uploaded = st.file_uploader("Качи 1 Hz CSV (от 'Strava' страницата)", type=["csv"])
    mode = st.selectbox("Режим за TIZ", ["hr", "pct_cs", "pct_cp"], index=0)

    if uploaded:
        df = pd.read_csv(uploaded).rename(columns={"t_sec": "second"}).set_index("second")
        zones_cfg = CFG["zones"]
        df = add_percent_columns(
            df,
            cs_run_kmh=float(st.session_state["CS_run_kmh"]),
            cp_bike_w=float(st.session_state["CP_bike_w"]),
            hrmax=float(st.session_state["HRmax"]),
            zones_cfg=zones_cfg,
        )
        tiz = compute_tiz(df, mode=mode, zones_cfg=zones_cfg)
        st.subheader("Време в зони (мин)")
        st.dataframe(tiz.to_frame(name="мин"), use_container_width=True)

        stress = compute_daily_stress(df)
        st.metric("Стрес (активност)", f"{int(stress)}")

        drift = compute_hr_drift(df)
        if np.isnan(drift):
            st.info("HR drift: недостатъчни данни.")
        else:
            st.metric("HR drift (decoupling)", f"{drift*100:.1f}%")

        date_str = st.text_input("Дата на активността (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"))
        if st.button("➕ Добави активността в дневния стрес"):
            st.session_state["meta_log"].append({"date": date_str, "stress": stress})
            st.success("Добавено. Долу виж ACWR.")

    st.subheader("ACWR (7/28)")
    if st.session_state["meta_log"]:
        meta_df = pd.DataFrame(st.session_state["meta_log"])
        meta_df["date"] = pd.to_datetime(meta_df["date"])
        series = aggregate_daily_stress(meta_df.assign(date=pd.to_datetime(meta_df["date"])))
        acwr = compute_acwr(series)
        if not acwr.empty:
            latest_date = acwr.index.max()
            st.metric("Последен ACWR", f"{acwr.loc[latest_date]:.2f}")
        st.write("Дневен стрес (по дати):")
        st.dataframe(series.to_frame(name="stress"), use_container_width=True)
    else:
        st.info("Добави поне една активност с дата, за да видиш ACWR.")

# ============================================================
# GENERATOR PAGE (съкратено)
# ============================================================
def render_generator():
    st.header("Генератор – седмичен план с адаптация по ACWR")
    latest_acwr = 1.0
    if st.session_state.get("meta_log"):
        meta_df = pd.DataFrame(st.session_state["meta_log"])
        meta_df["date"] = pd.to_datetime(meta_df["date"])
        series = aggregate_daily_stress(meta_df)
        acwr = compute_acwr(series)
        if len(acwr):
            latest_acwr = float(acwr.iloc[-1])

    st.write(f"Текущ ACWR (ако има данни): **{latest_acwr:.2f}**")

    base_calendar = ["Пон", "Вто", "Сря", "Чет", "Пет", "Съб", "Нед"]
    athlete_state = {
        "HRmax": float(st.session_state["HRmax"]),
        "CS_run_kmh": float(st.session_state["CS_run_kmh"]),
        "CP_bike_w": float(st.session_state["CP_bike_w"]),
    }
    indices_ctx = {"acwr_latest": latest_acwr}
    plan = generate_plan(athlete_state, indices_ctx, base_calendar, CFG)
    st.dataframe(plan, use_container_width=True)
    download_excel_button(plan, "💾 Download седмичен план (Excel)", "onflows_week_plan.xlsx")

# ============================================================
# SETTINGS PAGE (editable)
# ============================================================
def render_settings():
    st.header("Настройки – HRmax / CS / CP и зони (редакция)")
    with st.form("settings_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            hrmax_val = st.number_input("HRmax (bpm)", min_value=120, max_value=230,
                                        value=int(st.session_state["HRmax"]))
        with col2:
            cs_val = st.number_input("CS (бягане) km/h", min_value=8.0, max_value=26.0,
                                     value=float(st.session_state["CS_run_kmh"]), step=0.1)
        with col3:
            cp_val = st.number_input("CP (колоездене) W", min_value=80, max_value=500,
                                     value=int(st.session_state["CP_bike_w"]))

        st.markdown("### Редакция на зони")
        zones = CFG["zones"].copy()
        for metric_key, zones_dict in zones.items():
            with st.expander(f"Зони за: {metric_key}", expanded=False):
                for z_name, bounds in zones_dict.items():
                    c1, c2 = st.columns(2)
                    with c1:
                        lo = st.number_input(f"{metric_key}.{z_name} – долна граница",
                                             key=f"{metric_key}_{z_name}_lo",
                                             value=float(bounds[0]), step=0.01, format="%.2f")
                    with c2:
                        hi = st.number_input(f"{metric_key}.{z_name} – горна граница",
                                             key=f"{metric_key}_{z_name}_hi",
                                             value=float(bounds[1]), step=0.01, format="%.2f")
                    zones[metric_key][z_name] = [float(lo), float(hi)]

        submitted = st.form_submit_button("💾 Запази конфигурацията")
        if submitted:
            st.session_state["HRmax"] = hrmax_val
            st.session_state["CS_run_kmh"] = cs_val
            st.session_state["CP_bike_w"] = cp_val

            CFG["defaults"]["HRmax"] = hrmax_val
            CFG["defaults"]["CS_run_kmh"] = cs_val
            CFG["defaults"]["CP_bike_w"] = cp_val
            CFG["zones"] = zones
            save_config(CFG)
            st.success("Запазено в config.yaml и приложено веднага.")

# -------------------- Router --------------------
if page == "Strava":
    render_strava()
elif page == "Индекси":
    render_indices()
elif page == "Генератор":
    render_generator()
elif page == "Настройки":
    render_settings()

