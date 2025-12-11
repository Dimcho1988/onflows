# main_app.py
import streamlit as st
import numpy as np
import pandas as pd

from supabase import create_client
from strava_sync import init_clients, exchange_code_for_tokens, sync_from_strava, supabase
from run_walk_pipeline import process_run_walk_activity
from ski_pipeline import process_ski_activity


def get_token_info():
    if "token_info" in st.session_state:
        return st.session_state["token_info"]

    query_params = st.experimental_get_query_params()
    if "code" in query_params:
        auth_code = query_params["code"][0]
        token_info = exchange_code_for_tokens(auth_code)
        st.session_state["token_info"] = token_info
        return token_info
    return None


def get_supabase_client():
    return supabase


def load_activity_streams_for_user(user_id: int, limit: int = 30):
    sb = get_supabase_client()
    res = (
        sb.table("activities")
        .select("*")
        .eq("user_id", user_id)
        .not_.is_("sport_group", "null")
        .order("start_date", desc=True)
        .limit(limit)
        .execute()
    )
    acts = res.data or []
    activities = []

    for act in acts:
        streams_res = (
            sb.table("activity_streams")
            .select("stream_type,data")
            .eq("activity_id", act["id"])
            .execute()
        )
        streams_dict = {r["stream_type"]: r["data"] for r in streams_res.data}
        activities.append((act, streams_dict))
    return activities


def insert_segments(user_id: int, sport_group: str, seg_df: pd.DataFrame):
    if seg_df.empty:
        return 0
    sb = get_supabase_client()

    cols_map = {
        "activity_id": "activity_id",
        "seg_idx": "seg_idx",
        "t_start": "t_start",
        "t_end": "t_end",
        "dt_s": "dt_s",
        "d_m": "d_m",
        "slope_pct": "slope_pct",
        "v_kmh_raw": "v_kmh_raw",
        "valid_basic": "valid_basic",
        "speed_spike": "speed_spike",
        "k_glide": "k_glide",
        "v_glide": "v_glide",
        "v_flat_eq": "v_flat_eq",
        "v_flat_eq_cs": "v_flat_eq_cs",
        "delta_v_plus_kmh": "delta_v_plus_kmh",
        "r_kmh": "r_kmh",
        "tau_s": "tau_s",
        "hr_mean": "hr_mean",
    }

    rows = []
    for _, row in seg_df.iterrows():
        rec = {"user_id": user_id, "sport_group": sport_group}
        for src, dst in cols_map.items():
            rec[dst] = row.get(src)
        rows.append(rec)

    chunks = [rows[i:i+1000] for i in range(0, len(rows), 1000)]
    total = 0
    for chunk in chunks:
        res = sb.table("activity_segments").insert(chunk).execute()
        total += len(res.data)
    return total


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="onFlows – Strava + Ski + Run/Walk", layout="wide")
st.title("onFlows – Strava → Supabase → Сегментни модели (ски, бягане, ходене)")

init_clients(st)

token_info = get_token_info()

if not token_info:
    auth_url = (
        "https://www.strava.com/oauth/authorize"
        f"?client_id={st.secrets['strava']['client_id']}"
        "&response_type=code"
        f"&redirect_uri={st.secrets['app']['redirect_uri']}"
        "&approval_prompt=force"
        "&scope=read,activity:read_all"
    )
    st.warning("Все още не си свързал Strava акаунта си.")
    st.markdown(f"[Свържи се със Strava]({auth_url})")
    st.stop()

athlete = token_info["athlete"]
user_id = athlete["id"]
st.success(
    f"Свързан си със Strava като: "
    f"{athlete.get('username') or athlete.get('firstname')} {athlete.get('lastname')} "
    f"(athlete_id={user_id})"
)

if st.button("Синхронизирай новите Strava активности"):
    new_acts, total_rows = sync_from_strava(token_info)
    st.success(f"Синхронирани активности: {new_acts}, stream редове: {total_rows}")

st.markdown("---")
st.header("Обработка на активности (модели по спорт)")

N = st.number_input("Колко последни активности да обработим по спортните модели?", min_value=1, max_value=60, value=30)

# Параметри за моделите (тук може да сложиш sidebar, ако искаш)
V_crit_ski = st.number_input("V_crit за ски [km/h]", 5.0, 40.0, 20.0, 0.5)
DAMP_GLIDE = st.slider("Омекотяване на плъзгаемостта (ски)", 0.0, 1.0, 1.0, 0.05)

V_crit_run = st.number_input("V_crit за бягане/ходене [km/h]", 4.0, 30.0, 12.0, 0.5)

CS_ski = V_crit_ski
CS_run = V_crit_run

tau_min = 25.0
k_par = 35.0
q_par = 1.3
gamma_cs = 1.0

if st.button("Обработи последните активности и запиши сегментите"):
    activities = load_activity_streams_for_user(user_id, limit=int(N))
    total_segments = 0

    for act, streams_data in activities:
        sport_group = act.get("sport_group")
        if sport_group is None:
            continue

        # streams_data е dict: {stream_type: data_list}
        streams = {k: {"data": v} for k, v in streams_data.items()}

        if sport_group == "ski":
            seg_df = process_ski_activity(
                act_row=act,
                streams=streams,
                V_crit=V_crit_ski,
                DAMP_GLIDE=DAMP_GLIDE,
                CS=CS_ski,
                tau_min=tau_min,
                k_par=k_par,
                q_par=q_par,
                gamma_cs=gamma_cs,
            )
        elif sport_group in ("run", "walk"):
            seg_df = process_run_walk_activity(
                act_row=act,
                streams=streams,
                alpha_slope=1.0,
                V_crit=V_crit_run,
                CS=CS_run,
                tau_min=tau_min,
                k_par=k_par,
                q_par=q_par,
                gamma_cs=gamma_cs,
            )
        else:
            continue

        if seg_df is None or seg_df.empty:
            continue

        n_ins = insert_segments(user_id, sport_group, seg_df)
        total_segments += n_ins

    st.success(f"Записани сегменти в activity_segments: {total_segments}")
