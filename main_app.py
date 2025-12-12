import streamlit as st

from strava_sync import init_clients, exchange_code_for_tokens, sync_and_process_from_strava
from run_walk_pipeline import process_run_walk_activity
from ski_pipeline import process_ski_activity_30s


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


st.set_page_config(page_title="onFlows – Strava → Supabase (segments)", layout="wide")
st.title("onFlows – Strava → processing → Supabase (30s segments)")

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

st.sidebar.header("Параметри")

# Ski params
st.sidebar.subheader("Ski")
V_crit_ski = st.sidebar.number_input("V_crit ski [km/h]", 5.0, 40.0, 20.0, 0.5)
DAMP_GLIDE = st.sidebar.slider("DAMP_GLIDE", 0.0, 1.0, 1.0, 0.05)
CS_ski = st.sidebar.number_input("CS ski [km/h]", 5.0, 40.0, 20.0, 0.5)

# Run params
st.sidebar.subheader("Run/Walk")
V_crit_run = st.sidebar.number_input("V_crit run [km/h]", 4.0, 30.0, 12.0, 0.5)
CS_run = st.sidebar.number_input("CS run [km/h]", 4.0, 30.0, 12.0, 0.5)
alpha_slope = st.sidebar.slider("alpha_slope (run/walk)", 0.0, 2.0, 1.0, 0.05)

# CS params
st.sidebar.subheader("CS model")
tau_min = st.sidebar.number_input("tau_min [s]", 5.0, 120.0, 25.0, 1.0)
k_par = st.sidebar.number_input("k_par", 0.0, 200.0, 35.0, 1.0)
q_par = st.sidebar.number_input("q_par", 0.5, 3.0, 1.3, 0.05)
gamma_cs = st.sidebar.number_input("gamma", 0.0, 2.0, 1.0, 0.05)

params_by_sport = {
    "ski": dict(V_crit=V_crit_ski, DAMP_GLIDE=DAMP_GLIDE, CS=CS_ski, tau_min=tau_min, k_par=k_par, q_par=q_par, gamma_cs=gamma_cs),
    "run": dict(alpha_slope=alpha_slope, V_crit=V_crit_run, CS=CS_run, tau_min=tau_min, k_par=k_par, q_par=q_par, gamma_cs=gamma_cs),
}

if st.button("Sync + Process + Save (no streams)"):
    new_acts, total_segments = sync_and_process_from_strava(
        token_info=token_info,
        process_ski_activity_30s=process_ski_activity_30s,
        process_run_walk_activity=process_run_walk_activity,
        params_by_sport=params_by_sport,
        days_back_if_empty=100,
    )
    st.success(f"Готово. Активности: {new_acts}, записани 30s сегменти: {total_segments}")
