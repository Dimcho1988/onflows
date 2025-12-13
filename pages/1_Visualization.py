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
# Zone helpers (speed/intensity)
# -----------------------------
ZONE_ORDER = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]


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


def compute_hr_zone_ranked(seg: pd.DataFrame, zone_col: str = "zone_u", hr_col: str = "hr_mean") -> pd.Series:
    """
    HR zones by ranked HR, using counts from speed zones:
      - count how many segments are in each speed zone (Z1..Z6)
      - sort segments by hr_mean ascending
      - assign first n(Z1) -> Z1, next n(Z2) -> Z2, ... etc
    """
    if seg.empty:
        return pd.Series([], dtype=object)

    out = pd.Series(["Z?"] * len(seg), index=seg.index, dtype=object)

    if zone_col not in seg.columns:
        return out

    counts = seg[zone_col].value_counts(dropna=False).to_dict()
    zone_counts = {z: int(counts.get(z, 0)) for z in ZONE_ORDER}
    total_need = sum(zone_counts.values())

    if total_need <= 0 or hr_col not in seg.columns:
        return out

    # only consider rows that have a valid speed zone among Z1..Z6
    valid_idx = seg.index[seg[zone_col].isin(ZONE_ORDER)]
    if len(valid_idx) == 0:
        return out

    # sort by HR (NaN HR goes last)
    tmp = seg.loc[valid_idx, [hr_col]].copy()
    tmp["__hr__"] = pd.to_numeric(tmp[hr_col], errors="coerce")
    tmp = tmp.sort_values("__hr__", ascending=True, na_position="last")

    ordered_idx = tmp.index.tolist()

    # if HR missing for many segments, still keep ordering; they will be last -> higher zones typically
    pos = 0
    for z in ZONE_ORDER:
        k = zone_counts.get(z, 0)
        if k <= 0:
            continue
        take = ordered_idx[pos:pos + k]
        out.loc[take] = z
        pos += k

    # any leftovers (shouldn't happen if counts match) -> keep Z?
    return out


# -----------------------------
# Auto CS of activity (critical speed of the activity)
# -----------------------------
def estimate_activity_cs(seg: pd.DataFrame, speed_col: str) -> float | None:
    """
    Heuristic CS(activity) from segments:
      - uses valid_basic & non-spike if present
      - drops NaNs
      - CS = 90th percentile of speed_col (stable for most sessions)
    """
    if seg.empty or speed_col not in seg.columns:
        return None

    df = seg.copy()
    if "valid_basic" in df.columns:
        df = df[df["valid_basic"].fillna(True)]
    if "speed_spike" in df.columns:
        df = df[~df["speed_spike"].fillna(False)]

    v = pd.to_numeric(df[speed_col], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 8:
        return None

    cs = float(np.nanpercentile(v, 90))
    if not np.isfinite(cs) or cs <= 0:
        return None
    return cs


# -----------------------------
# Fit slope poly from last-30 segments
# -----------------------------
def fit_slope_poly_last30(seg_df: pd.DataFrame, deg: int = 2):
    """
    Fit F_raw(s) from segments:
      V0 = mean(v_raw) on near-flat slopes [-1,1]
      F_raw = V0 / v_raw
      polyfit(F_raw vs slope)
      shift so F(0)=1
    Returns np.poly1d or None.
    """
    if seg_df is None or seg_df.empty:
        return None

    df = seg_df.copy()
    df = df.dropna(subset=["slope_pct", "v_kmh_raw"])
    if df.empty:
        return None

    m = (
        df["valid_basic"].fillna(True)
        & (~df["speed_spike"].fillna(False))
        & df["slope_pct"].between(-15.0, 15.0)
        & (df["v_kmh_raw"] > 3.0)
        & (df["v_kmh_raw"] < 30.0)
    )
    df = df.loc[m, ["slope_pct", "v_kmh_raw"]].copy()
    if len(df) < 10:
        return None

    V0 = df.loc[df["slope_pct"].between(-1.0, 1.0), "v_kmh_raw"].mean()
    if not np.isfinite(V0) or V0 <= 0:
        return None

    df["F_raw"] = V0 / df["v_kmh_raw"]
    x = df["slope_pct"].to_numpy(dtype=float)
    y = df["F_raw"].to_numpy(dtype=float)

    if len(x) <= deg + 2:
        return None

    coeffs = np.polyfit(x, y, deg)
    raw_poly = np.poly1d(coeffs)

    # shift to enforce F(0)=1
    F0 = float(raw_poly(0.0))
    offset = F0 - 1.0
    coeffs_corr = raw_poly.coefficients.copy()
    coeffs_corr[-1] -= offset
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


def poly_to_formula(poly: np.poly1d | None, var: str = "s") -> str:
    if poly is None:
        return "None"
    c = np.array(poly.coefficients, dtype=float)
    deg = len(c) - 1
    parts = []
    for i, a in enumerate(c):
        p = deg - i
        if not np.isfinite(a):
            continue
        if p == 0:
            parts.append(f"{a:+.6f}")
        elif p == 1:
            parts.append(f"{a:+.6f}·{var}")
        else:
            parts.append(f"{a:+.6f}·{var}^{p}")
    s = " ".join(parts).lstrip("+").strip()
    return s if s else "0"


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
    # include hr_zone_ranked if exists
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

    for c in [
        "dt_s", "d_m", "slope_pct", "v_kmh_raw", "v_flat_eq", "v_flat_eq_cs", "hr_mean",
        "delta_v_plus_kmh", "r_kmh", "tau_s", "k_glide", "v_glide"
    ]:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)

    for c in ["valid_basic", "speed_spike"]:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    # hr_zone_ranked might be missing in some DB versions
    if "hr_zone_ranked" not in df.columns:
        df["hr_zone_ranked"] = None

    return df


@st.cache_data(ttl=30)
def fetch_last30_segments_for_fit(user_id: int, sport_group: str, segment_seconds: int, model_key: str, n_acts: int = 30) -> pd.DataFrame:
    res = (
        supabase.table("activities")
        .select("id")
        .eq("user_id", user_id)
        .eq("sport_group", sport_group)
        .order("start_date", desc=True)
        .limit(n_acts)
        .execute()
    )
    ids = [r["id"] for r in (res.data or []) if r.get("id") is not None]
    if not ids:
        return pd.DataFrame()

    res2 = (
        supabase.table("activity_segments")
        .select("activity_id,seg_idx,slope_pct,v_kmh_raw,valid_basic,speed_spike")
        .in_("activity_id", ids)
        .eq("sport_group", sport_group)
        .eq("segment_seconds", segment_seconds)
        .eq("model_key", model_key)
        .execute()
    )
    df = pd.DataFrame(res2.data or [])
    if df.empty:
        return df

    for c in ["slope_pct", "v_kmh_raw"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["valid_basic", "speed_spike"]:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)
    return df


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="onFlows – Visualization", layout="wide")
st.title("onFlows – Visualization (speed-zones + hr_zone_ranked + last-30 slope curve)")

init_clients(st)

with st.sidebar:
    st.header("User / Params")
    user_id = st.number_input("user_id (athlete_id)", min_value=1, value=41661616, step=1)

    st.subheader("Intensity base")
    use_cs_mod = st.checkbox("Use v_flat_eq_cs as base speed (if available)", value=True)

    st.subheader("CS of activity")
    cs_mode = st.selectbox("CS mode", ["AUTO (from activity)", "MANUAL (override)"], index=0)
    cs_manual = st.number_input("CS manual [km/h]", 2.0, 60.0, 12.0, 0.5)

    st.subheader("Slope regression display")
    alpha_slope_viz = st.slider("alpha_slope (display)", 0.0, 2.0, 1.0, 0.05)
    show_last30_curve = st.checkbox("Show last-30 fitted curve", value=True)

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

seg = seg.sort_values("seg_idx").reset_index(drop=True)
seg["t_rel_s"] = np.cumsum(seg["dt_s"].fillna(0.0).to_numpy(dtype=float)) - seg["dt_s"].fillna(0.0).to_numpy(dtype=float)

# base speed for intensity
has_cs_mod = ("v_flat_eq_cs" in seg.columns) and seg["v_flat_eq_cs"].notna().any()
base_col = "v_flat_eq_cs" if (use_cs_mod and has_cs_mod) else "v_flat_eq"
seg["v_base"] = seg[base_col]

# CS(activity)
cs_auto = estimate_activity_cs(seg, speed_col="v_base")
CS = cs_auto if (cs_mode.startswith("AUTO") and cs_auto is not None) else float(cs_manual)
if not np.isfinite(CS) or CS <= 0:
    CS = np.nan

seg["intensity"] = seg["v_base"] / CS if np.isfinite(CS) else np.nan
seg["zone_u"] = zones_from_intensity(seg["intensity"].to_numpy(dtype=float))

# HR zone ranked (compute every time, so it matches the current speed zone definition)
seg["hr_zone_ranked_calc"] = compute_hr_zone_ranked(seg, zone_col="zone_u", hr_col="hr_mean")

# load
seg["load_u"] = seg["dt_s"].fillna(0.0) * np.nan_to_num(seg["intensity"].to_numpy(dtype=float), nan=0.0) ** 2

# Header
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("sport_group", sport_group)
c2.metric("segments", int(len(seg)))
c3.metric("segment_seconds", seg_seconds)
c4.metric("model_key", model_key)
c5.metric("CS(activity) [km/h]", f"{CS:.2f}" if np.isfinite(CS) else "NA")

tabs = st.tabs(["Приравняване", "Модели (slope)", "CS модулатор", "Зони + HR (ranked) + натоварване", "Таблица сегменти"])


# -----------------------------
# Tab: Models (slope)
# -----------------------------
with tabs[1]:
    m = (
        seg["valid_basic"].fillna(True)
        & seg["slope_pct"].notna()
        & seg["v_kmh_raw"].notna()
        & (seg["v_kmh_raw"] > 0)
        & seg[base_col].notna()
    )
    dfm = seg.loc[m, ["slope_pct", "v_kmh_raw", base_col]].copy()
    dfm["F_implied"] = dfm[base_col] / dfm["v_kmh_raw"]

    st.subheader("Наклонен модел (точки + last-30 регресионна крива)")
    scat = (
        alt.Chart(dfm)
        .mark_circle(size=45, opacity=0.55)
        .encode(
            x=alt.X("slope_pct:Q", title="наклон [%]"),
            y=alt.Y("F_implied:Q", title="F (имплицитно = v_eq / v_raw)"),
            tooltip=["slope_pct:Q", "v_kmh_raw:Q", f"{base_col}:Q", "F_implied:Q"],
        )
        .properties(height=340)
        .interactive()
    )

    line = None
    poly = None
    if show_last30_curve:
        train_last30 = fetch_last30_segments_for_fit(
            user_id=int(user_id),
            sport_group=sport_group,
            segment_seconds=seg_seconds,
            model_key=model_key,
            n_acts=30,
        )

        poly = fit_slope_poly_last30(train_last30, deg=2)
        if poly is None:
            st.warning("Не успях да fit-на регресия от last-30 (вероятно липсват flat сегменти или има малко валидни данни).")
        else:
            s_grid = np.linspace(-15, 15, 241)
            F_grid = poly(s_grid)
            F_grid = apply_shape_constraints(F_grid, s_grid, fmin=0.7, fmax=1.7, alpha=alpha_slope_viz)

            df_line = pd.DataFrame({"slope_pct": s_grid, "F_fit": F_grid})
            line = (
                alt.Chart(df_line)
                .mark_line()
                .encode(
                    x="slope_pct:Q",
                    y=alt.Y("F_fit:Q", title="F"),
                    tooltip=["slope_pct:Q", "F_fit:Q"],
                )
            )

    if line is not None:
        st.altair_chart(scat + line, use_container_width=True)
    else:
        st.altair_chart(scat, use_container_width=True)

    st.markdown("**Формула (last-30 poly, преди shape constraints):**")
    st.code(f"F(s) = {poly_to_formula(poly, var='s')}", language="text")


# -----------------------------
# Tab: Normalization
# -----------------------------
with tabs[0]:
    dfp = seg[["seg_idx", "t_rel_s", "slope_pct", "v_kmh_raw", "v_flat_eq", "v_flat_eq_cs", "v_base", "hr_mean"]].copy()
    dfp["t_min"] = dfp["t_rel_s"] / 60.0
    long = pd.melt(
        dfp,
        id_vars=["seg_idx", "t_min"],
        value_vars=["v_kmh_raw", "v_flat_eq", "v_flat_eq_cs", "v_base"],
        var_name="metric",
        value_name="v_kmh",
    )
    st.altair_chart(
        alt.Chart(long.dropna(subset=["v_kmh"]))
        .mark_line()
        .encode(
            x=alt.X("t_min:Q", title="време [min]"),
            y=alt.Y("v_kmh:Q", title="скорост [km/h]"),
            color="metric:N",
        )
        .properties(height=320)
        .interactive(),
        use_container_width=True,
    )


# -----------------------------
# Tab: CS Modulator
# -----------------------------
with tabs[2]:
    cols = ["seg_idx", "t_rel_s", "v_base", "delta_v_plus_kmh", "r_kmh", "tau_s"]
    cols = [c for c in cols if c in seg.columns]
    dfc = seg[cols].copy()
    dfc["t_min"] = dfc["t_rel_s"] / 60.0

    st.altair_chart(
        alt.Chart(dfc.dropna(subset=["v_base"]))
        .mark_line()
        .encode(
            x=alt.X("t_min:Q", title="време [min]"),
            y=alt.Y("v_base:Q", title="v_base [km/h]"),
        )
        .properties(height=220)
        .interactive(),
        use_container_width=True,
    )


# -----------------------------
# Tab: Zones + HR ranked + Load
# -----------------------------
with tabs[3]:
    st.subheader("Speed zones (по % от CS(activity))")
    ztab = (
        seg.groupby("zone_u", dropna=False)
        .agg(
            segments=("seg_idx", "count"),
            time_s=("dt_s", "sum"),
            dist_m=("d_m", "sum"),
            load=("load_u", "sum"),
            v_mean=("v_base", "mean"),
            hr_mean=("hr_mean", "mean"),
        )
        .reset_index()
    )
    ztab["time_min"] = ztab["time_s"] / 60.0
    ztab["dist_km"] = ztab["dist_m"] / 1000.0
    st.dataframe(ztab[["zone_u", "segments", "time_min", "dist_km", "v_mean", "hr_mean", "load"]], use_container_width=True)

    st.subheader("HR zones ranked (същия брой сегменти като speed зоните)")
    htab = (
        seg.groupby("hr_zone_ranked_calc", dropna=False)
        .agg(
            segments=("seg_idx", "count"),
            time_s=("dt_s", "sum"),
            dist_m=("d_m", "sum"),
            v_mean=("v_base", "mean"),
            hr_mean=("hr_mean", "mean"),
        )
        .reset_index()
        .rename(columns={"hr_zone_ranked_calc": "hr_zone_ranked"})
    )
    htab["time_min"] = htab["time_s"] / 60.0
    htab["dist_km"] = htab["dist_m"] / 1000.0
    st.dataframe(htab[["hr_zone_ranked", "segments", "time_min", "dist_km", "v_mean", "hr_mean"]], use_container_width=True)

    st.subheader("Cross-tab: speed zone vs HR-ranked zone (segments)")
    ctab = pd.crosstab(seg["zone_u"], seg["hr_zone_ranked_calc"], dropna=False)
    st.dataframe(ctab, use_container_width=True)


# -----------------------------
# Tab: Segments table
# -----------------------------
with tabs[4]:
    show_cols = [
        "seg_idx", "t_start", "dt_s", "d_m",
        "slope_pct", "v_kmh_raw", "v_glide", "k_glide",
        "v_flat_eq", "v_flat_eq_cs", "v_base",
        "hr_mean", "zone_u", "hr_zone_ranked_calc",
        "valid_basic", "speed_spike",
        "delta_v_plus_kmh", "r_kmh", "tau_s",
    ]
    show_cols = [c for c in show_cols if c in seg.columns]
    st.dataframe(seg[show_cols], use_container_width=True)
