import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from supabase import create_client, Client


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


def fit_slope_poly_last30(seg_df: pd.DataFrame, deg: int = 2):
    """
    LAST-30 ACTIVITIES robust fit:

    - Compute V0 PER ACTIVITY from FLAT segments only: |slope| <= 1
    - For NON-FLAT segments: F_raw = V0_activity / v_raw
    - Pool across activities and polyfit(F_raw vs slope)
    - Shift so F(0)=1
    """
    if seg_df is None or seg_df.empty:
        return None

    df = seg_df.copy()
    need_cols = {"activity_id", "slope_pct", "v_kmh_raw"}
    if not need_cols.issubset(set(df.columns)):
        return None

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
    if df.empty:
        return None

    flat = df.loc[df["slope_pct"].abs() <= 1.0].copy()
    if flat.empty:
        return None

    v0 = flat.groupby("activity_id")["v_kmh_raw"].mean().rename("V0").reset_index()
    nflat = flat.groupby("activity_id")["v_kmh_raw"].size().rename("n_flat").reset_index()
    v0 = v0.merge(nflat, on="activity_id", how="left")
    v0 = v0.loc[v0["n_flat"] >= 2].copy()
    if v0.empty:
        return None

    df2 = df.merge(v0[["activity_id", "V0"]], on="activity_id", how="inner")
    if df2.empty:
        return None

    df_fit = df2.loc[df2["slope_pct"].abs() > 1.0].copy()
    if len(df_fit) <= deg + 6:
        return None

    df_fit["F_raw"] = df_fit["V0"] / df_fit["v_kmh_raw"]

    x = df_fit["slope_pct"].to_numpy(dtype=float)
    y = df_fit["F_raw"].to_numpy(dtype=float)
    if len(x) <= deg + 6:
        return None

    coeffs = np.polyfit(x, y, deg)
    raw_poly = np.poly1d(coeffs)

    F0 = float(raw_poly(0.0))
    if not np.isfinite(F0):
        return None

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


st.set_page_config(page_title="onFlows – Visualization", layout="wide")
st.title("onFlows – Visualization (last-30 activities regression)")

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
    alpha_slope_viz = st.slider("alpha_slope (display)", 0.0, 2.0, 1.0, 0.05)
    show_last30_curve = st.checkbox("Show last-30 ACTIVITIES fitted curve", value=True)

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
seg["v_base"] = seg["v_flat_eq_cs"] if use_cs_mod and seg["v_flat_eq_cs"].notna().any() else seg["v_flat_eq"]
seg["intensity"] = seg["v_base"] / CS if np.isfinite(CS) else np.nan
seg["zone_u"] = zones_from_intensity(seg["intensity"].to_numpy(dtype=float))
seg["load_u"] = seg["dt_s"].fillna(0.0) * np.nan_to_num(seg["intensity"].to_numpy(dtype=float), nan=0.0) ** 2

c1, c2, c3, c4 = st.columns(4)
c1.metric("sport_group", sport_group)
c2.metric("segments", int(len(seg)))
c3.metric("segment_seconds", seg_seconds)
c4.metric("model_key", model_key)

tabs = st.tabs(["Приравняване", "Модели (slope/glide)", "CS модулатор", "Зони + натоварване", "Таблица сегменти"])

with tabs[1]:
    base_col = "v_flat_eq_cs" if (use_cs_mod and seg["v_flat_eq_cs"].notna().any()) else "v_flat_eq"

    m = (
        seg["valid_basic"].fillna(True)
        & seg["slope_pct"].notna()
        & seg["v_kmh_raw"].notna()
        & (seg["v_kmh_raw"] > 0)
        & seg[base_col].notna()
    )
    dfm = seg.loc[m, ["slope_pct", "v_kmh_raw", base_col]].copy()
    dfm["F_implied"] = dfm[base_col] / dfm["v_kmh_raw"]

    st.subheader("Наклонен модел (точки + last-30 ACTIVITIES регресионна крива)")
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
    if show_last30_curve:
        train_last30 = fetch_last30_segments_for_fit(
            user_id=int(user_id),
            sport_group=sport_group,
            segment_seconds=seg_seconds,
            model_key=model_key,
            n_acts=30,
        )

        if train_last30.empty:
            st.warning("Няма last-30 сегменти за fit.")
        else:
            st.caption(
                f"last-30: rows={len(train_last30)} | flat(|s|<=1%)={int((train_last30['slope_pct'].abs()<=1).sum())} "
                f"| activities={train_last30['activity_id'].nunique()}"
            )

        poly = fit_slope_poly_last30(train_last30, deg=2)
        if poly is None:
            st.warning("Не успях да fit-на регресия от last-30 activities (липсват flat сегменти за V0 или има малко валидни данни).")
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

            st.caption(f"Last-30 fit: poly deg=2 (V0 per-activity), then clipped + alpha={alpha_slope_viz:.2f}")

    if line is not None:
        st.altair_chart(scat + line, use_container_width=True)
    else:
        st.altair_chart(scat, use_container_width=True)

with tabs[0]:
    dfp = seg[["seg_idx", "t_rel_s", "slope_pct", "v_kmh_raw", "v_flat_eq", "v_flat_eq_cs", "hr_mean"]].copy()
    dfp["t_min"] = dfp["t_rel_s"] / 60.0
    long = pd.melt(dfp, id_vars=["seg_idx", "t_min"], value_vars=["v_kmh_raw", "v_flat_eq", "v_flat_eq_cs"], var_name="metric", value_name="v_kmh")
    st.altair_chart(
        alt.Chart(long.dropna(subset=["v_kmh"])).mark_line().encode(
            x=alt.X("t_min:Q", title="време [min]"),
            y=alt.Y("v_kmh:Q", title="скорост [km/h]"),
            color="metric:N",
        ).properties(height=320).interactive(),
        use_container_width=True
    )

with tabs[2]:
    cols = ["seg_idx", "t_rel_s", "v_base", "delta_v_plus_kmh", "r_kmh", "tau_s"]
    dfc = seg[cols].copy()
    dfc["t_min"] = dfc["t_rel_s"] / 60.0
    st.altair_chart(
        alt.Chart(dfc.dropna(subset=["v_base"])).mark_line().encode(
            x=alt.X("t_min:Q", title="време [min]"),
            y=alt.Y("v_base:Q", title="v (eq or eq_cs) [km/h]"),
        ).properties(height=220).interactive(),
        use_container_width=True
    )

with tabs[3]:
    ztab = (
        seg.groupby("zone_u", dropna=False)
        .agg(time_s=("dt_s", "sum"), dist_m=("d_m", "sum"), load=("load_u", "sum"), v_mean=("v_base", "mean"))
        .reset_index()
    )
    ztab["time_min"] = ztab["time_s"] / 60.0
    ztab["dist_km"] = ztab["dist_m"] / 1000.0
    st.dataframe(ztab[["zone_u", "time_min", "dist_km", "v_mean", "load"]], use_container_width=True)

    if "hr_zone_ranked" in seg.columns:
        st.subheader("HR zone ranked (ако го има в DB)")
        htab = (
            seg.groupby("hr_zone_ranked", dropna=False)
            .agg(time_s=("dt_s", "sum"), mean_hr=("hr_mean", "mean"))
            .reset_index()
        )
        htab["time_min"] = htab["time_s"] / 60.0
        st.dataframe(htab[["hr_zone_ranked", "time_min", "mean_hr"]], use_container_width=True)

with tabs[4]:
    show_cols = [c for c in [
        "seg_idx","t_start","dt_s","d_m","slope_pct","v_kmh_raw","v_glide","k_glide",
        "v_flat_eq","v_flat_eq_cs","zone","hr_zone_ranked","hr_mean","valid_basic","speed_spike"
    ] if c in seg.columns]
    st.dataframe(seg[show_cols], use_container_width=True)
