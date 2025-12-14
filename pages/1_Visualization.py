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
seg["v_base"] = seg["v_flat_eq_cs"] if use_cs_mod and seg.get("v_flat_eq_cs") is not None and seg["v_flat_eq_cs"].notna().any() else seg["v_flat_eq"]
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
    base_col = "v_flat_eq_cs" if (use_cs_mod and "v_flat_eq_cs" in seg.columns and seg["v_flat_eq_cs"].notna().any()) else "v_flat_eq"

    m = (
        seg["valid_basic"].fillna(True)
        & seg["slope_pct"].notna()
        & seg["v_kmh_raw"].notna()
        & (seg["v_kmh_raw"] > 0)
        & seg[base_col].notna()
    )
    dfm = seg.loc[m, ["slope_pct", "v_kmh_raw", base_col]].copy()
    dfm["F_implied"] = dfm[base_col] / dfm["v_kmh_raw"]
    dfm["source"] = "selected_activity"

    st.subheader("Наклонен модел (точки + rolling last-N регресионна крива)")

    scat = (
        alt.Chart(dfm)
        .mark_circle(size=45, opacity=0.55)
        .encode(
            x=alt.X("slope_pct:Q", title="наклон [%]"),
            y=alt.Y("F_implied:Q", title="F (имплицитно = v_eq / v_raw)"),
            tooltip=["slope_pct:Q", "v_kmh_raw:Q", f"{base_col}:Q", "F_implied:Q", "source:N"],
            color=alt.Color("source:N", legend=alt.Legend(title="source"))
        )
        .properties(height=340)
        .interactive()
    )

    line = None

    if show_lastN_curve:
        exclude_id = activity_id if exclude_current else None

        train_seg, train_ids = fetch_lastN_segments_for_fit(
            user_id=int(user_id),
            sport_group=sport_group,
            segment_seconds=seg_seconds,
            model_key=model_key,
            exclude_activity_id=exclude_id,
            n_acts=int(N_train),
            lookback_limit=int(lookback_limit),
            use_all_model_keys=bool(use_all_model_keys_train),
        )

        mk_txt = "ALL" if use_all_model_keys_train else str(model_key)
        train_info = (
            f"Training set: activities_used={len(train_ids)} (target N={int(N_train)}), "
            f"segments={len(train_seg)} | model_key={mk_txt} | exclude_current={exclude_current} | lookback={int(lookback_limit)}"
        )
        st.caption(train_info)

        if train_seg is not None and not train_seg.empty:
            train_plot = train_seg.copy()
            train_plot = train_plot.dropna(subset=["slope_pct", "v_kmh_raw"])
            train_plot["F_raw_tmp"] = np.nan
            train_plot["source"] = "training_lastN"

            # for scatter of training: use global flat estimate just for visualization
            v0_vis = train_plot.loc[train_plot["slope_pct"].between(-1.0, 1.0), "v_kmh_raw"].mean()
            if np.isfinite(v0_vis) and v0_vis > 0:
                train_plot["F_raw_tmp"] = (v0_vis / train_plot["v_kmh_raw"]).clip(0.5, 3.5)

            scat_train = (
                alt.Chart(train_plot.dropna(subset=["F_raw_tmp"]))
                .mark_circle(size=35, opacity=0.35)
                .encode(
                    x="slope_pct:Q",
                    y=alt.Y("F_raw_tmp:Q", title="F (имплицитно)"),
                    tooltip=["activity_id:Q", "slope_pct:Q", "v_kmh_raw:Q", "F_raw_tmp:Q"],
                    color=alt.Color("source:N", legend=alt.Legend(title="source"))
                )
            )

            scat = scat + scat_train

        poly = fit_slope_poly_lastN(train_seg, deg=2, f_clip_min=0.7, f_clip_max=1.7, min_rows=80)

        if poly is None:
            st.warning("Не успях да fit-на регресия от rolling last-N (липса на flat сегменти или твърде малко валидни данни).")
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

            st.caption(f"Fit: poly deg=2 (shifted F(0)=1) + constraints + alpha={alpha_slope_viz:.2f}")

    if line is not None:
        st.altair_chart(scat + line, use_container_width=True)
    else:
        st.altair_chart(scat, use_container_width=True)


# -----------------------------
# Tab: Приравняване
# -----------------------------
with tabs[0]:
    dfp = seg[["seg_idx", "t_rel_s", "slope_pct", "v_kmh_raw", "v_flat_eq", "v_flat_eq_cs", "hr_mean"]].copy()
    dfp["t_min"] = dfp["t_rel_s"] / 60.0
    long = pd.melt(
        dfp,
        id_vars=["seg_idx", "t_min"],
        value_vars=[c for c in ["v_kmh_raw", "v_flat_eq", "v_flat_eq_cs"] if c in dfp.columns],
        var_name="metric",
        value_name="v_kmh"
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
        use_container_width=True
    )


# -----------------------------
# Tab: CS modulator
# -----------------------------
with tabs[2]:
    cols = [c for c in ["seg_idx", "t_rel_s", "v_base", "delta_v_plus_kmh", "r_kmh", "tau_s"] if c in seg.columns]
    dfc = seg[cols].copy()
    dfc["t_min"] = dfc["t_rel_s"] / 60.0
    ycol = "v_base" if "v_base" in dfc.columns else (cols[0] if cols else None)
    if ycol is not None and "t_min" in dfc.columns:
        st.altair_chart(
            alt.Chart(dfc.dropna(subset=[ycol]))
            .mark_line()
            .encode(
                x=alt.X("t_min:Q", title="време [min]"),
                y=alt.Y(f"{ycol}:Q", title="v (eq or eq_cs) [km/h]"),
            )
            .properties(height=220)
            .interactive(),
            use_container_width=True
        )
    else:
        st.info("Няма достатъчно данни за CS графика.")


# -----------------------------
# Tab: UI zones (from intensity)
# -----------------------------
with tabs[3]:
    ztab = (
        seg.groupby("zone_u", dropna=False)
        .agg(time_s=("dt_s", "sum"), dist_m=("d_m", "sum"), load=("load_u", "sum"), v_mean=("v_base", "mean"))
        .reset_index()
    )
    ztab["time_min"] = ztab["time_s"] / 60.0
    ztab["dist_km"] = ztab["dist_m"] / 1000.0
    st.dataframe(ztab[["zone_u", "time_min", "dist_km", "v_mean", "load"]], use_container_width=True)


# -----------------------------
# Tab: Zones + HR (from DB columns: zone, hr_zone_ranked)
# -----------------------------

with tabs[4]:
    st.subheader("Зони по методиката (от DB): Speed zones + HR-ranked zones")

    required = [
        "seg_idx", "dt_s", "d_m", "hr_mean",
        "v_kmh_raw", "v_flat_eq", "v_flat_eq_cs",
        "zone", "hr_zone_ranked",
    ]
    missing = [c for c in required if c not in seg.columns]
    if missing:
        st.error(f"Липсват колони в сегментите (activity_segments): {missing}")
        st.stop()

    dfz = seg[required].copy()

    st.markdown("### Диапазони на зоните (ориентир спрямо текущия CS от UI)")
    if np.isfinite(CS) and CS > 0:
        zr = zone_ranges_table(float(CS), ref_name="CS (UI)")
        st.dataframe(zr, use_container_width=True, hide_index=True)
    else:
        st.info("CS не е валиден — не мога да покажа km/h граници за зоните.")

    speed = (
        dfz.groupby("zone", dropna=False)
        .agg(
            segments=("zone", "count"),
            time_s=("dt_s", "sum"),
            dist_m=("d_m", "sum"),
            hr_mean=("hr_mean", "mean"),
            v_raw_mean=("v_kmh_raw", "mean"),
            v_eq_mean=("v_flat_eq", "mean"),
            v_eq_cs_mean=("v_flat_eq_cs", "mean"),
        )
        .reset_index()
        .rename(columns={"zone": "Zone"})
    )
    speed["time_min"] = speed["time_s"] / 60.0
    speed["dist_km"] = speed["dist_m"] / 1000.0

    hrz = (
        dfz.groupby("hr_zone_ranked", dropna=False)
        .agg(
            segments=("hr_zone_ranked", "count"),
            time_s=("dt_s", "sum"),
            dist_m=("d_m", "sum"),
            hr_mean=("hr_mean", "mean"),
            v_raw_mean=("v_kmh_raw", "mean"),
            v_eq_mean=("v_flat_eq", "mean"),
            v_eq_cs_mean=("v_flat_eq_cs", "mean"),
        )
        .reset_index()
        .rename(columns={"hr_zone_ranked": "Zone"})
    )
    hrz["time_min"] = hrz["time_s"] / 60.0
    hrz["dist_km"] = hrz["dist_m"] / 1000.0

    st.markdown("### Една таблица: Speed zones vs HR-ranked zones")
    comb = (
        speed.set_index("Zone").add_prefix("speed_")
        .join(hrz.set_index("Zone").add_prefix("hr_"), how="outer")
        .reset_index()
    )

    order = {z: i for i, z in enumerate(["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"], start=1)}
    comb["_ord"] = comb["Zone"].map(order).fillna(999)
    comb = comb.sort_values(["_ord", "Zone"]).drop(columns=["_ord"])

    st.dataframe(
        comb[
            [
                "Zone",
                "speed_segments", "speed_time_min", "speed_dist_km", "speed_v_raw_mean", "speed_v_eq_mean", "speed_v_eq_cs_mean", "speed_hr_mean",
                "hr_segments", "hr_time_min", "hr_dist_km", "hr_v_raw_mean", "hr_v_eq_mean", "hr_v_eq_cs_mean", "hr_hr_mean",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Raw segments (preview, първите 200)")
    st.dataframe(dfz.head(200), use_container_width=True)

