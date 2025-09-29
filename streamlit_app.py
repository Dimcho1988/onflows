import os
import io
import json
import yaml
import time
import base64
import pandas as pd
import numpy as np
import requests
import streamlit as st

from datetime import datetime, timedelta

from strava_streams import StravaClient, build_timeseries_1hz
from normalize import add_percent_columns
from indices import compute_tiz, compute_daily_stress, aggregate_daily_stress, compute_acwr, compute_hr_drift
from calibration import infer_modalities, suggest_thresholds
from generator import generate_plan

st.set_page_config(page_title="onFlows MVP", layout="wide")


# ---------- Helpers ----------
def load_config():
   def save_config(cfg):
    with open("config.yaml", "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = load_config()

def init_state():
    d = CFG["defaults"]
    st.session_state.setdefault("HRmax", d["HRmax"])
    st.session_state.setdefault("CS_run_kmh", d["CS_run_kmh"])
    st.session_state.setdefault("CP_bike_w", d["CP_bike_w"])
    st.session_state.setdefault("strava_token", None)
    st.session_state.setdefault("activities_cache", None)
    st.session_state.setdefault("meta_log", [])  # за дневен стрес
init_state()

def app_redirect_uri():
    # Streamlit Cloud → Secrets: APP_REDIRECT_URI
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
        "grant_type": "authorization_code"
    }
    r = requests.post(url, data=payload)
    r.raise_for_status()
    return r.json()

def download_csv_button(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")

def download_excel_button(df: pd.DataFrame, label: str, filename: str):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Plan")
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ---------- Sidebar ----------
st.sidebar.title("onFlows")
page = st.sidebar.radio("Меню", ["Strava", "Индекси", "Генератор", "Настройки"])


# ---------- STRAVA PAGE ----------
if page == "Strava":
    st.header("Strava – OAuth, активности и 1 Hz таблица")

    # 1) OAuth
    params = st.experimental_get_query_params()
    if "code" in params and not st.session_state["strava_token"]:
        try:
            token_data = exchange_code_for_token(params["code"][0])
            st.session_state["strava_token"] = token_data["access_token"]
            st.success("✅ Успешен вход в Strava!")
            # Премахни code от адреса
            st.experimental_set_query_params()
        except Exception as e:
            st.error(f"Грешка при обмен на код за токен: {e}")

    if not st.session_state["strava_token"]:
        st.markdown("1) Натисни бутона за вход → одобри достъпа → връщаш се тук автоматично.")
        if st.button("🔐 Вход със Strava"):
            st.markdown(f"[Отвори Strava OAuth]({strava_oauth_url()})")
        st.stop()

    client = StravaClient(st.session_state["strava_token"])

    # 2) Списък активности
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
            "distance_km": round((a.get("distance", 0) or 0)/1000.0, 2),
            "moving_time_min": int((a.get("moving_time", 0) or 0)/60),
            "avg_hr": a.get("average_heartrate", None),
            "avg_speed_kmh": round((a.get("average_speed", 0) or 0)*3.6, 2)
        } for a in acts])
        st.dataframe(df_acts, use_container_width=True)
    else:
        st.info("Няма кеширани активности. Натисни „Обнови активностите“.")

    # 3) Избор на активност → Streams → 1 Hz
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
            # Download
            dl = df_1hz.reset_index().rename(columns={"second":"t_sec"})
            download_csv_button(dl, "💾 Download 1Hz CSV", f"activity_{activity_id}_1hz.csv")

            # Бърза калибрация/предложения
            rec = suggest_thresholds(df_1hz)
            if rec:
                st.info(f"Предложени прагове (ориентир): {rec}")
        except Exception as e:
            st.error(f"Грешка при дърпане на streams: {e}")


# ---------- INDICES PAGE ----------
elif page == "Индекси":
    st.header("Индекси – TIZ, дневен стрес, ACWR, HR drift")

    uploaded = st.file_uploader("Качи 1 Hz CSV (от 'Strava' страницата)", type=["csv"])
    mode = st.selectbox("Режим за TIZ", ["hr", "pct_cs", "pct_cp"], index=0)

    if uploaded:
        df = pd.read_csv(uploaded)
        # очакваме t_sec + колони от 1 Hz
        df = df.rename(columns={"t_sec":"second"})
        df = df.set_index("second")

        # Нормализация
        zones_cfg = CFG["zones"]
        df = add_percent_columns(
            df,
            cs_run_kmh=float(st.session_state["CS_run_kmh"]),
            cp_bike_w=float(st.session_state["CP_bike_w"]),
            hrmax=float(st.session_state["HRmax"]),
            zones_cfg=zones_cfg
        )

        # TIZ
        tiz = compute_tiz(df, mode=mode, zones_cfg=zones_cfg)
        st.subheader("Време в зони (мин)")
        st.dataframe(tiz.to_frame(name="мин"), use_container_width=True)

        # Дневен стрес
        stress = compute_daily_stress(df)
        st.metric("Стрес (активност)", f"{int(stress)}")

        # HR drift
        drift = compute_hr_drift(df)
        if np.isnan(drift):
            st.info("HR drift: недостатъчни данни.")
        else:
            st.metric("HR drift (decoupling)", f"{drift*100:.1f}%")

        # Добави в дневния лог (за ACWR)
        date_str = st.text_input("Дата на активността (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"))
        if st.button("➕ Добави активността в дневния стрес"):
            st.session_state["meta_log"].append({"date": date_str, "stress": stress})
            st.success("Добавено. Отиди долу за ACWR.")

    # ACWR секция
    st.subheader("ACWR (7/28)")
    if st.session_state["meta_log"]:
        meta_df = pd.DataFrame(st.session_state["meta_log"])
        # нормализирай дата
        meta_df["date"] = pd.to_datetime(meta_df["date"]).dt.date
        series = aggregate_daily_stress(meta_df.assign(date=pd.to_datetime(meta_df["date"])))
        acwr = compute_acwr(series)
        if not acwr.empty:
            latest_date = acwr.index.max()
            st.metric("Последен ACWR", f"{acwr.loc[latest_date]:.2f}")
        st.write("Дневен стрес:")
        st.dataframe(series.to_frame(name="stress"), use_container_width=True)
    else:
        st.info("Добави поне една активност с дата, за да видиш ACWR.")


# ---------- GENERATOR PAGE ----------
elif page == "Генератор":
    st.header("Генератор – седмичен план с адаптация по ACWR")

    # Лек state от „Индекси“
    latest_acwr = 1.0
    if st.session_state.get("meta_log"):
        meta_df = pd.DataFrame(st.session_state["meta_log"])
        meta_df["date"] = pd.to_datetime(meta_df["date"])
        series = aggregate_daily_stress(meta_df)
        acwr = compute_acwr(series)
        if len(acwr):
            latest_acwr = float(acwr.iloc[-1])

    st.write(f"Текущ ACWR (ако има данни): **{latest_acwr:.2f}**")

    # Календар/дни
    base_calendar = ["Пон","Вто","Сря","Чет","Пет","Съб","Нед"]

    athlete_state = {
        "HRmax": float(st.session_state["HRmax"]),
        "CS_run_kmh": float(st.session_state["CS_run_kmh"]),
        "CP_bike_w": float(st.session_state["CP_bike_w"]),
    }
    indices_ctx = {"acwr_latest": latest_acwr}

    plan = generate_plan(athlete_state, indices_ctx, base_calendar, CFG)
    st.dataframe(plan, use_container_width=True)

    download_excel_button(plan, "💾 Download седмичен план (Excel)", "onflows_week_plan.xlsx")


# ---------- SETTINGS PAGE ----------
elif page == "Настройки":
    st.header("Настройки – HRmax / CS / CP и зони")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state["HRmax"] = st.number_input("HRmax (bpm)", min_value=120, max_value=230, value=int(st.session_state["HRmax"]))
    with col2:
        st.session_state["CS_run_kmh"] = st.number_input("CS (бягане) km/h", min_value=8.0, max_value=26.0, value=float(st.session_state["CS_run_kmh"]), step=0.1)
    with col3:
        st.session_state["CP_bike_w"] = st.number_input("CP (колоездене) W", min_value=80, max_value=500, value=int(st.session_state["CP_bike_w"]))

    st.markdown("Зоните са дефинирани в `config.yaml`. В следващи версии ще добавим UI за редакция.")
    st.json(CFG["zones"])

    st.info("Секрети за Strava/Redirect се добавят в Streamlit Cloud → Settings → Secrets:")
    st.code(
        'STRAVA_CLIENT_ID = "XXXX"\n'
        'STRAVA_CLIENT_SECRET = "YYYY"\n'
        'APP_REDIRECT_URI = "https://onflows.streamlit.app"\n',
        language="bash"
    )

    st.markdown("Локално стартиране:")
    st.code("pip install -r requirements.txt\nstreamlit run streamlit_app.py", language="bash")
