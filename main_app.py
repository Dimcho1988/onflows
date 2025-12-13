import streamlit as st
import pandas as pd

from strava_sync import (
    init_clients,
    exchange_code_for_tokens,
    sync_and_process_from_strava,
    supabase,  # <- use the initialized global client from strava_sync
    # helpers for visual/debug (rolling models)
    map_sport_group,
    get_last_n_activity_ids_excluding,
    fetch_training_segments_30s,
    fetch_training_segments_7s,
    fit_run_walk_slope_poly_from_segments,
    fit_glide_poly_from_7s,
    fit_ski_slope_poly_from_30s,
)
from run_walk_pipeline import process_run_walk_activity
from ski_pipeline import process_ski_activity_30s


# ----------------------------
# Utils
# ----------------------------
def get_token_info():
    if "token_info" in st.session_state:
        return st.session_state["token_info"]

    # Streamlit old API (works on older Streamlit versions)
    query_params = st.experimental_get_query_params()
    if "code" in query_params:
        auth_code = query_params["code"][0]
        token_info = exchange_code_for_tokens(auth_code)
        st.session_state["token_info"] = token_info
        return token_info

    return None


def poly_to_formula(poly, var="s"):
    if poly is None:
        return "None"
    c = poly.coefficients
    deg = len(c) - 1
    parts = []
    for i, a in enumerate(c):
        p = deg - i
        if abs(float(a)) < 1e-12:
            continue
        if p == 0:
            parts.append(f"{float(a):.6g}")
        elif p == 1:
            parts.append(f"{float(a):.6g}*{var}")
        else:
            parts.append(f"{float(a):.6g}*{var}^{p}")
    if not parts:
        return "0"
    return " + ".join(parts)


def load_recent_activities(user_id: int, sport_group: str | None, limit: int = 20) -> pd.DataFrame:
    if supabase is None:
        return pd.DataFrame()
    q = (
        supabase.table("activities")
        .select("id,strava_activity_id,name,sport_group,start_date,distance_m,moving_time_s,avg_speed_mps,avg_heartrate")
        .eq("user_id", user_id)
        .order("start_date", desc=True)
        .limit(limit)
    )
    if sport_group:
        q = q.eq("sport_group", sport_group)
    res = q.execute()
    return pd.DataFrame(res.data or [])


def load_segments_for_activity(activity_id: int, sport_group: str, model_key: str, segment_seconds: int = 30) -> pd.DataFrame:
    if supabase is None:
        return pd.DataFrame()
    res = (
        supabase.table("activity_segments")
        .select("*")
        .eq("activity_id", activity_id)
        .eq("sport_group", sport_group)
        .eq("segment_seconds", segment_seconds)
        .eq("model_key", model_key)
        .order("seg_idx", desc=False)
        .execute()
    )
    return pd.DataFrame(res.data or [])


# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="onFlows – Strava → Supabase (segments)", layout="wide")
st.title("onFlows – Strava → processing → Supabase")

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
    f"{athlete.get('username') or athlete.get('firstname')} "
    f"{athlete.get('lastname')} "
    f"(athlete_id={user_id})"
)

# -------------------------------------------------
# Sidebar parameters
# -------------------------------------------------
st.sidebar.header("Параметри")

st.sidebar.subheader("Ski")
V_crit_ski = st.sidebar.number_input("V_crit ski [km/h]", 5.0, 40.0, 20.0, 0.5)
DAMP_GLIDE = st.sidebar.slider("DAMP_GLIDE", 0.0, 1.0, 1.0, 0.05)
CS_ski = st.sidebar.number_input("CS ski [km/h]", 5.0, 40.0, 20.0, 0.5)

st.sidebar.subheader("Run / Walk")
V_crit_run = st.sidebar.number_input("V_crit run [km/h]", 4.0, 30.0, 12.0, 0.5)
CS_run = st.sidebar.number_input("CS run [km/h]", 4.0, 30.0, 12.0, 0.5)
alpha_slope = st.sidebar.slider("alpha_slope", 0.0, 2.0, 1.0, 0.05)

st.sidebar.subheader("CS model")
tau_min = st.sidebar.number_input("tau_min [s]", 5.0, 120.0, 25.0, 1.0)
k_par = st.sidebar.number_input("k_par", 0.0, 200.0, 35.0, 1.0)
q_par = st.sidebar.number_input("q_par", 0.5, 3.0, 1.3, 0.05)
gamma_cs = st.sidebar.number_input("gamma", 0.0, 2.0, 1.0, 0.05)

days_back = st.sidebar.number_input("Days back (first sync)", min_value=1, max_value=5000, value=200, step=10)

params_by_sport = {
    "ski": dict(
        model_key="ski_glide_v3",
        V_crit=V_crit_ski,
        DAMP_GLIDE=DAMP_GLIDE,
        CS=CS_ski,
        tau_min=tau_min,
        k_par=k_par,
        q_par=q_par,
        gamma_cs=gamma_cs,
    ),
    "run": dict(
        model_key="run_slope_v2",
        alpha_slope=alpha_slope,
        V_crit=V_crit_run,
        CS=CS_run,
        tau_min=tau_min,
        k_par=k_par,
        q_par=q_par,
        gamma_cs=gamma_cs,
    ),
    "walk": dict(
        model_key="walk_slope_v1",
        alpha_slope=alpha_slope * 0.6,
        V_crit=V_crit_run,
        CS=CS_run,
        tau_min=tau_min,
        k_par=k_par,
        q_par=q_par,
        gamma_cs=gamma_cs,
    ),
}

# -------------------------------------------------
# Actions
# -------------------------------------------------
colA, colB = st.columns([1, 1], gap="large")

with colA:
    if st.button("Sync + Process + Save (no streams)", use_container_width=True):
        new_acts, total_segments = sync_and_process_from_strava(
            token_info=token_info,
            process_ski_activity_30s=process_ski_activity_30s,
            process_run_walk_activity=process_run_walk_activity,
            params_by_sport=params_by_sport,
            days_back_if_empty=int(days_back),
        )
        st.success(f"Готово. Активности: {new_acts}, записани сегменти: {total_segments}")

with colB:
    st.info(
        "Debug панелът отдолу показва последни активности, сегменти и формули "
        "на rolling регресиите (последни 30 активности по спорт)."
    )

st.divider()

# -------------------------------------------------
# Debug / Views
# -------------------------------------------------
st.subheader("Преглед: активности / сегменти / rolling регресии")

sport_pick = st.selectbox("Sport group", ["ski", "run", "walk"], index=0, horizontal=True)
model_key_pick = params_by_sport[sport_pick]["model_key"]

acts_df = load_recent_activities(user_id=user_id, sport_group=sport_pick, limit=25)
if acts_df.empty:
    st.warning("Няма активности в Supabase за избрания sport_group (или няма връзка към Supabase).")
    st.stop()

acts_df = acts_df.copy()
acts_df["start_date"] = pd.to_datetime(acts_df["start_date"], errors="coerce")
acts_df["label"] = (
    acts_df["start_date"].dt.strftime("%Y-%m-%d %H:%M").fillna("NA")
    + " | "
    + acts_df["name"].fillna("Unnamed")
    + " | id="
    + acts_df["id"].astype(str)
)

picked_label = st.selectbox("Избери активност", acts_df["label"].tolist(), index=0)
picked_row = acts_df.loc[acts_df["label"] == picked_label].iloc[0]
picked_activity_id = int(picked_row["id"])

c1, c2 = st.columns([1.2, 1], gap="large")

with c1:
    st.markdown("### Активности (последни 25)")
    st.dataframe(
        acts_df[["id", "strava_activity_id", "start_date", "name", "distance_m", "moving_time_s", "avg_speed_mps", "avg_heartrate"]],
        use_container_width=True,
        hide_index=True,
    )

with c2:
    st.markdown("### Сегменти (30s)")
    seg_df = load_segments_for_activity(
        activity_id=picked_activity_id,
        sport_group=sport_pick,
        model_key=model_key_pick,
        segment_seconds=30,
    )
    if seg_df.empty:
        st.warning("Няма записани 30s сегменти за тази активност/модел.")
    else:
        # show the key columns first
        preferred = [
            "seg_idx", "dt_s", "d_m", "slope_pct", "v_kmh_raw",
            "k_glide", "v_glide", "v_flat_eq", "v_flat_eq_cs",
            "delta_v_plus_kmh", "r_kmh", "tau_s",
            "hr_mean", "zone", "hr_zone_ranked",
            "valid_basic", "speed_spike",
        ]
        cols = [c for c in preferred if c in seg_df.columns] + [c for c in seg_df.columns if c not in preferred]
        st.dataframe(seg_df[cols], use_container_width=True, hide_index=True)
        if "v_kmh_raw" in seg_df.columns:
            st.line_chart(seg_df[["v_kmh_raw"]], height=180)
        if "v_flat_eq" in seg_df.columns:
            st.line_chart(seg_df[["v_flat_eq"]], height=180)
        if "v_flat_eq_cs" in seg_df.columns:
            st.line_chart(seg_df[["v_flat_eq_cs"]], height=180)

st.divider()

# -------------------------------------------------
# Rolling model formulas (last 30 activities excluding picked)
# -------------------------------------------------
st.markdown("### Rolling регресии (последни 30 активности, без избраната)")

train_ids = get_last_n_activity_ids_excluding(
    user_id=user_id,
    sport_group=sport_pick,
    exclude_activity_id=picked_activity_id,
    n=30,
)

if not train_ids:
    st.warning("Няма достатъчно други активности за rolling тренировка.")
    st.stop()

if sport_pick in ("run", "walk"):
    train30 = fetch_training_segments_30s(
        activity_ids=train_ids,
        sport_group=sport_pick,
        segment_seconds=30,
        model_key=model_key_pick,
    )
    slope_poly = fit_run_walk_slope_poly_from_segments(train30)

    st.write(f"**Slope poly:** `{poly_to_formula(slope_poly, var='s')}`")
    if slope_poly is not None and not train30.empty:
        tmp = train30.copy()
        tmp = tmp[tmp["valid_basic"] & (~tmp["speed_spike"])].dropna(subset=["slope_pct", "v_kmh_raw"])
        if not tmp.empty:
            x = tmp["slope_pct"].astype(float).to_numpy()
            V0 = tmp.loc[(tmp["slope_pct"].between(-1.0, 1.0)), "v_kmh_raw"].mean()
            if V0 and V0 > 0:
                y = (V0 / tmp["v_kmh_raw"].astype(float)).to_numpy()
                chart_df = pd.DataFrame({"slope_pct": x, "F_raw": y}).sort_values("slope_pct")
                st.scatter_chart(chart_df, x="slope_pct", y="F_raw", height=260)

elif sport_pick == "ski":
    train7 = fetch_training_segments_7s(train_ids, model_key=model_key_pick)
    glide_poly = fit_glide_poly_from_7s(train7)

    train30 = fetch_training_segments_30s(
        activity_ids=train_ids,
        sport_group="ski",
        segment_seconds=30,
        model_key=model_key_pick,
    )
    slope_poly = fit_ski_slope_poly_from_30s(
        train30=train30,
        glide_poly_for_training=glide_poly,
        damp_glide=float(params_by_sport["ski"]["DAMP_GLIDE"]),
    )

    st.write(f"**Glide poly (v_model vs slope):** `{poly_to_formula(glide_poly, var='s')}`")
    st.write(f"**Slope poly (F vs slope):** `{poly_to_formula(slope_poly, var='s')}`")

    if glide_poly is not None and not train7.empty:
        tmp7 = train7.copy()
        tmp7 = tmp7[tmp7["valid_basic"] & (~tmp7["speed_spike"])].dropna(subset=["slope_pct", "v_kmh_raw"])
        if not tmp7.empty:
            chart_df = tmp7[["slope_pct", "v_kmh_raw"]].copy().sort_values("slope_pct")
            st.scatter_chart(chart_df, x="slope_pct", y="v_kmh_raw", height=260)
