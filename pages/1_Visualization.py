import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from supabase import create_client, Client


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
# Zone helpers (UI-only zones from intensity/CS)
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


def zone_ranges_table(reference_value: float, ref_name: str = "CS/V_crit") -> pd.DataFrame:
    rows = []
    for i, z in enumerate(ZONE_NAMES):
        lo = ZONE_BOUNDS[i]
        hi = ZONE_BOUNDS[i + 1]
        lo_kmh = lo * reference_value if np.isfinite(reference_value) else np.nan
        hi_kmh = hi * reference_value if (np.isfinite(reference_value) and np.isfinite(hi)) else np.nan
        rows.append(
            dict(
                Zone=z,
                rel_from=lo,
                rel_to=hi if np.isfinite(hi) else np.inf,
                kmh_from=lo_kmh,
                kmh_to=hi_kmh if np.isfinite(hi_kmh) else np.inf,
                reference=ref_name,
            )
        )
    return pd.DataFrame(rows)


# -----------------------------
# Fit slope poly from last-N (activities) - robust
# -----------------------------
def fit_slope_poly_lastN(
    seg_df: pd.DataFrame,
    deg: int = 2,
    f_clip_min: float = 0.7,
    f_clip_max: float = 1.7,
    min_rows: int = 80,
):
    """
    Robust fit:
      - V0 per-activity from flat slopes [-1,1]
      - F_raw = V0_activity / v_raw
      - CLIP F_raw BEFORE polyfit
      - shift so F(0)=1
    Returns np.poly1d or None.
    """
    if seg_df is None or seg_df.empty:
        return None

    df = seg_df.copy()
    if "activity_id" not in df.columns:
        df["activity_id"] = 0

    df = df.dropna(subset=["activity_id", "slope_pct", "v_kmh_raw"])
    if df.empty:
        return None

    m = (
        df["valid_basic"].fillna(True)
        & (~df["speed_spike"].fillna(False))
        & df["slope_pct"].between(-15.0, 15.0)
        & (df["v_kmh_raw"] > 3.0)
        & (df["v_kmh_raw"] < 30.0)
    )
    df = df.loc[m, ["activity_id", "slope_pct", "v_kmh_raw"]].copy()
    if df.empty or len(df) < max(10, int(min_rows)):
        return None

    flat = df[df["slope_pct"].between(-1.0, 1.0)].copy()
    if flat.empty:
        return None

    v0_map = flat.groupby("activity_id")["v_kmh_raw"].mean().to_dict()
    df["V0"] = df["activity_id"].map(v0_map)
    df = df.dropna(subset=["V0"])
    df = df[df["V0"] > 0]
    if df.empty or len(df) < max(10, int(min_rows)):
        return None

    df["F_raw"] = (df["V0"] / df["v_kmh_raw"]).clip(lower=f_clip_min, upper=f_clip_max)

    x = df["slope_pct"].to_numpy(dtype=float)
    y = df["F_raw"].to_numpy(dtype=float)

    if len(x) <= deg + 2:
        return None

    coeffs = np.polyfit(x, y, deg)
    raw_poly = np.poly1d(coeffs)

    # shift to enforce F(0)=1
    F0 = float(raw_poly(0.0))
    coeffs_corr = raw_poly.coefficients.copy()
    coeffs_corr[-1] -= (F0 - 1.0)
    return np.poly1d(coeffs_corr)


def apply_shape_constraints(F: np.ndarray, slopes: np.ndarray, fmin=0.7, fmax=1.7, alpha=1.0):
    slopes = np.asarray(slopes, dtype=float)
    F = np.asarray(F, dtype=float)

    F = np.clip(F, fmin, fmax)

    mask_mid = np.abs(slopes) <= 1.0
    F[mask_mid] = 1.0

    mask_down = slopes < -1.0
    F[mask_down] = np.minimum(F[mask_down], 1.0)

    mask_up = slopes > 1.0
    F[mask_up] = np.maximum(F[mask_up], 1.0)

    F = 1.0 + float(alpha) * (F - 1.0)
    return F


# -----------------------------
# Data fetch
# -----------------------------
@st.cache_data(ttl=30)
def fetch_activities(user_id: int) -> pd.DataFrame:
    res = (
        supabase.table("activities")
        .select("*")
        .eq("user_id", user_id)
        .order("start_date", desc=True)
        .limit(200)
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
        .eq("segment_seconds", segment_seconds)
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

    for c in ["dt_s", "d_m", "slope_pct", "v_kmh_raw", "v_flat_eq", "v_flat_eq_cs", "hr_mean",
              "delta_v_plus_kmh", "r_kmh", "tau_s", "k_glide", "v_glide"]:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)

    for c in ["valid_basic", "speed_spike"]:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    return df


@st.cache_data(ttl=30)
def fetch_lastN_segments_for_fit(
    user_id: int,
    sport_group: str,
    segment_seconds: int,
    model_key: str,
    exclude_activity_id: int | None,
    n_acts: int = 30,
    lookback_limit: int = 200,
    use_all_model_keys: bool = True,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Returns (segments_df, activity_ids_used)
    Uses up to last n_acts activities by start_date.
    Optionally excludes current activity_id from training.

    If use_all_model_keys=True => ignores model_key in training fetch.
    """
    res = (
        supabase.table("activities")
        .select("id")
        .eq("user_id", user_id)
        .eq("sport_group", sport_group)
        .order("start_date", desc=True)
        .limit(int(lookback_limit))
        .execute()
    )
    ids_all = [int(r["id"]) for r in (res.data or []) if r.get("id") is not None]

    if exclude_activity_id is not None:
        ids_all = [i for i in ids_all if int(i) != int(exclude_activity_id)]

    ids = ids_all[: max(1, int(n_acts))]
    if not ids:
        return pd.DataFrame(), []

    q = (
        supabase.table("activity_segments")
        .select("activity_id,seg_idx,slope_pct,v_kmh_raw,valid_basic,speed_spike,model_key")
        .in_("activity_id", ids)
        .eq("sport_group", sport_group)
        .eq("segment_seconds", int(segment_seconds))
    )

    if not use_all_model_keys:
        q = q.eq("model_key", model_key)

    res2 = q.execute()
    df = pd.DataFrame(res2.data or [])
    if df.empty:
        return df, ids

    for c in ["slope_pct", "v_kmh_raw"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["valid_basic", "speed_spike"]:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    return df, ids


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="onFlows – Visualization", layout="wide")
st.title("onFlows – Visualization (rolling last-N activities)")

init_clients(st)

with st.sidebar:
    st.header("User / Params")
    user_id = st.number_input("user_id (athlete_id)", min_value=1, value=41661616, step=1)

    st.subheader("Zoning / CS")
    CS_run = st.number_input("CS run [km/h]", 4.0, 30.0, 12.0, 0.5)
    CS_walk = st.number_input("CS walk [km/h]", 2.0, 20.0, 7.0, 0.5)
    CS_ski = st.number_input("CS ski [km/h]", 5.0, 40.0, 20.0, 0.5)
    use_cs_mod = st.checkbox("Use v_flat_eq_cs for intensity", value=True)

    st.subheader("Slope regression display")
    N_train = st.slider("N_train activities (rolling)", 1, 30, 30, 1)
    exclude_current = st.checkbox("Exclude текущата активност от training", value=True)
    lookback_limit = st.number_input("Training lookback (max activities scanned)", 50, 500, 200, 10)
    use_all_model_keys_train = st.checkbox("Training: използвай ALL model_key", value=True)

    alpha_slope_viz = st.slider("alpha_slope (display)", 0.0, 2.0, 1.0, 0.05)
    show_lastN_curve = st.checkbox("Show last-N fitted curve", value=True)

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
model_key = st.selectbox(
    "model_key",
    sorted(avail_sg[avail_sg["segment_seconds"] == seg_seconds]["model_key"].unique().tolist()),
    index=0
)

seg = fetch_segments(activity_id, sport_group, seg_seconds, model_key)
if seg.empty:
    st.error("Сегментите не се зареждат.")
    st.stop()

CS = CS_run if sport_group == "run" else (CS_walk if sport_group == "walk" else CS_ski)
CS = float(CS) if CS and CS > 0 else np.nan

seg = seg.sort_values("seg_idx").reset_index(drop=True)
seg["t_rel_s"] = np.cumsum(seg["dt_s"].fillna(0.0).to_numpy(dtype=float)) - seg["dt_s"].fillna(0.0).to_numpy(dtype=float)

seg["v_base"] = seg["v_flat_eq_cs"] if use_cs_mod and ("v_flat_eq_cs" in seg.columns) and seg["v_flat_eq_cs"].notna().any() else seg["v_flat_eq"]
seg["intensity"] = seg["v_base"] / CS if np.isfinite(CS) else np.nan
seg["zone_u"] = zones_from_intensity(seg["intensity"].to_numpy(dtype=float))
seg["load_u"] = seg["dt_s"].fillna(0.0) * np.nan_to_num(seg["intensity"].to_numpy(dtype=float), nan=0.0) ** 2

# Header
c1, c2, c3, c4 = st.columns(4)
c1.metric("sport_group", sport_group)
c2.metric("segments", int(len(seg)))
c3.metric("segment_seconds", seg_seconds)
c4.metric("model_key", model_key)

tabs = st.tabs(["Приравняване", "Модели (slope/glide)", "CS модулатор", "Зони (UI)", "Зони + HR (от DB)"])


# -----------------------------
# Tab: Models
# -----------------------------
with tabs[1]:
    base_col = "v_flat_eq_cs" if (use_cs
