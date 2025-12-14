import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from supabase import Client, create_client

# -----------------------------
# Supabase init
# -----------------------------
SUPABASE_URL = None
SUPABASE_KEY = None
supabase: Client | None = None


def init_clients(st_app):
    global SUPABASE_URL, SUPABASE_KEY, supabase
    SUPABASE_URL = st_app.secrets["supabase"]["url"]
    SUPABASE_KEY = st_app.secrets["supabase"]["service_role_key"]
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def to_utc_dt(s):
    if s is None:
        return pd.NaT
    try:
        return pd.to_datetime(s, utc=True)
    except Exception:
        return pd.NaT


def safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


# -----------------------------
# Zone helpers (UI-only ranges)
# -----------------------------
ZONE_BOUNDS = [0.0, 0.75, 0.85, 0.95, 1.05, 1.15, np.inf]
ZONE_NAMES = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]


def zones_from_intensity(intensity: np.ndarray) -> np.ndarray:
    z = np.full_like(intensity, "Z?", dtype=object)
    x = intensity
    z[x < 0.75] = "Z1"
    z[(x >= 0.75) & (x < 0.85)] = "Z2"
    z[(x >= 0.85) & (x < 0.95)] = "Z3"
    z[(x >= 0.95) & (x < 1.05)] = "Z4"
    z[(x >= 1.05) & (x < 1.15)] = "Z5"
    z[x >= 1.15] = "Z6"
    z[~np.isfinite(x)] = "Z?"
    return z


def zone_ranges_table(reference_value: float, ref_name: str = "CS (UI)") -> pd.DataFrame:
    rows = []
    for i, z in enumerate(ZONE_NAMES):
        lo = float(ZONE_BOUNDS[i])
        hi = float(ZONE_BOUNDS[i + 1]) if np.isfinite(ZONE_BOUNDS[i + 1]) else np.inf
        lo_kmh = lo * reference_value if np.isfinite(reference_value) else np.nan
        hi_kmh = hi * reference_value if (np.isfinite(reference_value) and np.isfinite(hi)) else np.inf
        rows.append(
            dict(
                Zone=z,
                rel_from=lo,
                rel_to=hi,
                kmh_from=lo_kmh,
                kmh_to=hi_kmh,
                reference=ref_name,
            )
        )
    return pd.DataFrame(rows)


# -----------------------------
# Data fetch (cached)
# -----------------------------
@st.cache_data(ttl=30)
def fetch_activities(user_id: int) -> pd.DataFrame:
    res = (
        supabase.table("activities")
        .select("*")
        .eq("user_id", user_id)
        .order("start_date", desc=True)
        .limit(400)
        .execute()
    )
    df = pd.DataFrame(res.data or [])
    if df.empty:
        return df
    df["start_date"] = df["start_date"].apply(to_utc_dt)
    return df


@st.cache_data(ttl=30)
def fetch_available_models(activity_id: int) -> pd.DataFrame:
    res = (
        supabase.table("activity_segments")
        .select("sport_group,segment_seconds,model_key")
        .eq("activity_id", activity_id)
        .execute()
    )
    df = pd.DataFrame(res.data or [])
    if df.empty:
        return df
    return df.drop_duplicates().sort_values(["sport_group", "segment_seconds", "model_key"])


@st.cache_data(ttl=30)
def fetch_segments(activity_id: int, sport_group: str, segment_seconds: int, model_key: str) -> pd.DataFrame:
    res = (
        supabase.table("activity_segments")
        .select("*")
        .eq("activity_id", activity_id)
        .eq("sport_group", sport_group)
        .eq("segment_seconds", int(segment_seconds))
        .eq("model_key", model_key)
        .order("seg_idx", desc=False)
        .limit(50000)
        .execute()
    )
    df = pd.DataFrame(res.data or [])
    if df.empty:
        return df

    df["t_start"] = df["t_start"].apply(to_utc_dt)
    df["t_end"] = df["t_end"].apply(to_utc_dt)

    for c in ["dt_s", "d_m", "slope_pct", "v_kmh_raw", "v_flat_eq", "v_flat_eq_cs", "hr_mean"]:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)

    for c in ["valid_basic", "speed_spike"]:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    return df


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="onFlows – Visualization", layout="wide")
st.title("onFlows – Visualization")

init_clients(st)

with st.sidebar:
    st.header("User / Params")
    user_id = st.number_input("user_id (athlete_id)", min_value=1, value=41661616, step=1)

    st.subheader("Zoning / CS")
    CS_run = st.number_input("CS run [km/h]", 4.0, 30.0, 12.0, 0.5)
    CS_walk = st.number_input("CS walk [km/h]", 2.0, 20.0, 7.0, 0.5)
    CS_ski = st.number_input("CS ski [km/h]", 5.0, 40.0, 20.0, 0.5)
    use_cs_mod = st.checkbox("Use v_flat_eq_cs for intensity", value=True)

    if st.button("Clear cache"):
        st.cache_data.clear()
        st.rerun()

acts = fetch_activities(int(user_id))
if acts.empty:
    st.warning("Няма активности за този user_id в activities.")
    st.stop()

acts_show = acts.copy()
acts_show["label"] = (
    acts_show["start_date"].dt.strftime("%Y-%m-%d %H:%M").fillna("NA")
    + " | "
    + acts_show["sport_group"].fillna("?").astype(str)
    + " | "
    + acts_show["name"].fillna("(no name)").astype(str)
    + " | id="
    + acts_show["id"].astype(str)
)

act_label = st.selectbox("Избери активност", acts_show["label"].tolist(), index=0)
act_row = acts_show.loc[acts_show["label"] == act_label].iloc[0]
activity_id = int(act_row["id"])
sport_group = str(act_row.get("sport_group") or "run")

avail = fetch_available_models(activity_id)
if avail.empty:
    st.error("Няма сегменти за тази активност в activity_segments.")
    st.stop()

avail_sg = avail[avail["sport_group"] == sport_group]
if avail_sg.empty:
    avail_sg = avail.copy()

seg_seconds = int(st.selectbox("segment_seconds", sorted(avail_sg["segment_seconds"].unique().tolist()), index=0))

# prefer non-unknown model_key
model_keys = sorted(
    avail_sg[avail_sg["segment_seconds"] == seg_seconds]["model_key"].unique().tolist(),
    key=lambda x: (str(x) == "unknown", str(x)),
)
model_key = st.selectbox("model_key", model_keys, index=0)

seg = fetch_segments(activity_id, sport_group, seg_seconds, model_key)
if seg.empty:
    st.error("Сегментите не се зареждат.")
    st.stop()

CS = float(CS_run if sport_group == "run" else (CS_walk if sport_group == "walk" else CS_ski))

seg = seg.sort_values("seg_idx").reset_index(drop=True)
seg["t_rel_s"] = np.cumsum(seg["dt_s"].fillna(0.0).to_numpy(dtype=float)) - seg["dt_s"].fillna(0.0).to_numpy(dtype=float)

seg["v_base"] = seg["v_flat_eq_cs"] if (use_cs_mod and "v_flat_eq_cs" in seg.columns and seg["v_flat_eq_cs"].notna().any()) else seg["v_flat_eq"]
seg["intensity"] = seg["v_base"] / CS if np.isfinite(CS) and CS > 0 else np.nan
seg["zone_u"] = zones_from_intensity(seg["intensity"].to_numpy(dtype=float))

# Header
c1, c2, c3, c4 = st.columns(4)
c1.metric("sport_group", sport_group)
c2.metric("segments", int(len(seg)))
c3.metric("segment_seconds", int(seg_seconds))
c4.metric("model_key", str(model_key))

tabs = st.tabs(["Приравняване", "Зони (UI)", "Зони + HR (от DB)"])

# -----------------------------
# Tab: Equalization plot
# -----------------------------
with tabs[0]:
    plot_df = pd.DataFrame({
        "t_min": seg["t_rel_s"] / 60.0,
        "v_kmh_raw": seg.get("v_kmh_raw"),
        "v_flat_eq": seg.get("v_flat_eq"),
        "v_flat_eq_cs": seg.get("v_flat_eq_cs"),
    })
    long = plot_df.melt(id_vars=["t_min"], var_name="metric", value_name="speed_kmh")
    chart = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("t_min:Q", title="време [min]"),
            y=alt.Y("speed_kmh:Q", title="скорост [km/h]"),
            color="metric:N",
            tooltip=["t_min", "metric", "speed_kmh"],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# Tab: UI zones (preview)
# -----------------------------
with tabs[1]:
    st.subheader("Зони (UI preview) – на база intensity = v_base/CS")
    zr = zone_ranges_table(float(CS), ref_name="CS (UI)")
    st.dataframe(zr, use_container_width=True)

    agg = (
        seg.groupby("zone_u", dropna=False)
        .agg(time_s=("dt_s", "sum"), dist_m=("d_m", "sum"), hr_mean=("hr_mean", "mean"))
        .reset_index()
        .rename(columns={"zone_u": "zone"})
    )
    agg["time_min"] = agg["time_s"] / 60.0
    agg["dist_km"] = agg["dist_m"] / 1000.0
    st.dataframe(agg[["zone", "time_min", "dist_km", "hr_mean"]], use_container_width=True)

# -----------------------------
# Tab: DB zones (your method)
# -----------------------------
with tabs[2]:
    st.subheader("Зони по методиката (от DB): Speed zones + HR-ranked zones")

    required = ["zone", "hr_zone_ranked", "dt_s", "d_m", "hr_mean", "v_kmh_raw", "v_flat_eq", "v_flat_eq_cs"]
    missing = [c for c in required if c not in seg.columns]
    if missing:
        st.error(f"Липсват колони в DB сегментите: {missing}")
        st.stop()

    zr = zone_ranges_table(float(CS), ref_name="CS (UI)")
    st.markdown("### Диапазони на зоните (ориентир спрямо текущия CS от UI)")
    st.dataframe(zr, use_container_width=True)

    df = seg.copy()

    speed = (
        df.groupby("zone", dropna=False)
        .agg(
            speed_segments=("zone", "count"),
            speed_time_s=("dt_s", "sum"),
            speed_dist_m=("d_m", "sum"),
            speed_hr_mean=("hr_mean", "mean"),
            speed_v_raw_mean=("v_kmh_raw", "mean"),
            speed_v_eq_mean=("v_flat_eq", "mean"),
            speed_v_eq_cs_mean=("v_flat_eq_cs", "mean"),
        )
        .reset_index()
        .rename(columns={"zone": "Zone"})
    )
    speed["speed_time_min"] = speed["speed_time_s"] / 60.0
    speed["speed_dist_km"] = speed["speed_dist_m"] / 1000.0

    hrz = (
        df.groupby("hr_zone_ranked", dropna=False)
        .agg(
            hr_segments=("hr_zone_ranked", "count"),
            hr_time_s=("dt_s", "sum"),
            hr_dist_m=("d_m", "sum"),
            hr_hr_mean=("hr_mean", "mean"),
            hr_v_raw_mean=("v_kmh_raw", "mean"),
            hr_v_eq_mean=("v_flat_eq", "mean"),
            hr_v_eq_cs_mean=("v_flat_eq_cs", "mean"),
        )
        .reset_index()
        .rename(columns={"hr_zone_ranked": "Zone"})
    )
    hrz["hr_time_min"] = hrz["hr_time_s"] / 60.0
    hrz["hr_dist_km"] = hrz["hr_dist_m"] / 1000.0

    # Build a combined table Z1..Z6 plus any extra (None)
    combined = pd.merge(speed, hrz, on="Zone", how="outer")
    st.markdown("### Една таблица: Speed zones vs HR-ranked zones")
    st.dataframe(combined, use_container_width=True)

    st.markdown("### Raw segments (preview, първите 200)")
    st.dataframe(df[["seg_idx", "dt_s", "d_m", "hr_mean", "v_kmh_raw", "v_flat_eq", "v_flat_eq_cs", "zone", "hr_zone_ranked"]].head(200), use_container_width=True)
