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


# -----------------------------
# Helpers
# -----------------------------
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
    """
    Единна зона система по относителна интензивност = v_base / CS.
    v_base = v_flat_eq_cs (ако е включено) иначе v_flat_eq.

    Граници (можеш да ги коригираш по-късно):
      Z1 < 0.75
      Z2 0.75–0.85
      Z3 0.85–0.95
      Z4 0.95–1.05
      Z5 1.05–1.15
      Z6 >= 1.15
    """
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


def compute_zone_hr_sorted_allocation(
    seg_df: pd.DataFrame,
    zone_col: str = "zone_u",
    hr_col: str = "hr_mean",
    time_col: str = "dt_s",
    spike_col: str = "speed_spike",
    valid_col: str = "valid_basic",
    zone_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    ВАРИАНТ Б (по твоята методика) – HR по зони чрез "разпределяне" по HR.

    Идея:
      1) Зоните (Z1–Z6) са дефинирани по скорост/интензивност => имаме време в зона:
           T_k = Σ dt_s за сегментите, които са в зона Zk.
      2) Вземаме сегментите с наличен HR (и без speed_spike), сортираме ги по HR ↑.
      3) Разпределяме времето от най-ниските HR сегменти към Z1, после към Z2, ...,
         докато "напълним" T_k секунди за всяка зона.
      4) Ако се наложи да "разрежем" сегмент (частично време), правим частично претегляне.

    Резултат:
      mean_hr_zone(k) = Σ(hr_i * t_i_assigned) / Σ(t_i_assigned)

    Забележка:
      Това НЕ е "директно залепване" HR към скоростни зони по сегмент,
      а конструирана HR-градация, съгласно времето в скоростните зони.
    """
    if zone_order is None:
        zone_order = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]

    df = seg_df.copy()

    # 1) Време по скоростни зони (това е "target" time за HR разпределяне)
    df_t = df.copy()
    if valid_col in df_t.columns:
        df_t = df_t[df_t[valid_col].fillna(True)]
    df_t = df_t.dropna(subset=[zone_col, time_col])

    time_by_zone = (
        df_t.groupby(zone_col)[time_col]
        .sum()
        .to_dict()
    )

    # гарантираме, че всички зони съществуват като ключове
    target = {z: float(time_by_zone.get(z, 0.0)) for z in zone_order}

    # 2) HR кандидати (без спайкове, с HR, с време)
    df_hr = df.copy()
    if spike_col in df_hr.columns:
        df_hr = df_hr[~df_hr[spike_col].fillna(False)]
    if valid_col in df_hr.columns:
        df_hr = df_hr[df_hr[valid_col].fillna(True)]

    df_hr = df_hr.dropna(subset=[hr_col, time_col]).copy()
    if df_hr.empty:
        return pd.DataFrame({"zone": zone_order, "mean_hr_zone": [np.nan] * len(zone_order)})

    # сортиране по HR ↑
    df_hr = df_hr.sort_values(hr_col, ascending=True).reset_index(drop=True)

    # 3) Разпределяне на време по зони
    # ще обходим зоните в ред Z1..Z6 и ще "изядем" време от HR-сегментите
    idx = 0
    remaining_in_row = float(df_hr.at[0, time_col]) if len(df_hr) > 0 else 0.0

    out_rows = []
    for z in zone_order:
        need = float(target.get(z, 0.0))
        if need <= 0:
            out_rows.append({"zone": z, "mean_hr_zone": np.nan})
            continue

        assigned_time = 0.0
        hr_weighted_sum = 0.0

        while need > 1e-9 and idx < len(df_hr):
            hr_val = float(df_hr.at[idx, hr_col])
            take = min(need, remaining_in_row)

            assigned_time += take
            hr_weighted_sum += hr_val * take

            need -= take
            remaining_in_row -= take

            # ако текущият HR ред е изчерпан, минаваме на следващия
            if remaining_in_row <= 1e-9:
                idx += 1
                if idx < len(df_hr):
                    remaining_in_row = float(df_hr.at[idx, time_col])

        mean_hr = (hr_weighted_sum / assigned_time) if assigned_time > 0 else np.nan
        out_rows.append({"zone": z, "mean_hr_zone": mean_hr})

    return pd.DataFrame(out_rows)


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

    for c in [
        "dt_s", "d_m", "slope_pct", "v_kmh_raw",
        "k_glide", "v_glide", "v_flat_eq", "v_flat_eq_cs",
        "delta_v_plus_kmh", "r_kmh", "tau_s", "hr_mean"
    ]:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)

    # booleans safe
    for c in ["valid_basic", "speed_spike"]:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

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
    df = df.drop_duplicates().sort_values(["sport_group", "segment_seconds", "model_key"])
    return df


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="onFlows – Visualization", layout="wide")
st.title("onFlows – Visualization (segments in Supabase)")

init_clients(st)

with st.sidebar:
    st.header("Потребител")
    user_id = st.number_input("user_id (athlete_id)", min_value=1, value=41661616, step=1)

    st.header("Зони (по CS)")
    CS_run = st.number_input("CS run [km/h]", 4.0, 30.0, 12.0, 0.5)
    CS_walk = st.number_input("CS walk [km/h]", 2.0, 20.0, 7.0, 0.5)
    CS_ski = st.number_input("CS ski [km/h]", 5.0, 40.0, 20.0, 0.5)

    use_cs_mod = st.checkbox("Зони по v_flat_eq_cs (иначе по v_flat_eq)", value=True)

acts = fetch_activities(int(user_id))
if acts.empty:
    st.warning("Няма активности за този user_id в таблицата activities.")
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
    st.error("Няма записани сегменти за тази активност в activity_segments.")
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
    st.error("Сегментите не се зареждат (празно).")
    st.stop()

CS = CS_run if sport_group == "run" else (CS_walk if sport_group == "walk" else CS_ski)
CS = float(CS) if CS and CS > 0 else np.nan

# derived
seg = seg.sort_values("seg_idx").reset_index(drop=True)
seg["t_rel_s"] = np.cumsum(seg["dt_s"].fillna(0.0).to_numpy(dtype=float)) - seg["dt_s"].fillna(0.0).to_numpy(dtype=float)

seg["v_base"] = seg["v_flat_eq_cs"] if use_cs_mod and seg["v_flat_eq_cs"].notna().any() else seg["v_flat_eq"]
seg["v_base"] = seg["v_base"].astype(float)

seg["intensity"] = seg["v_base"] / CS if np.isfinite(CS) else np.nan
seg["zone_u"] = zones_from_intensity(seg["intensity"].to_numpy(dtype=float))

# simple load starter
seg["load_u"] = seg["dt_s"].fillna(0.0) * np.nan_to_num(seg["intensity"].to_numpy(dtype=float), nan=0.0) ** 2

# header
c1, c2, c3, c4 = st.columns(4)
c1.metric("sport_group", sport_group)
c2.metric("segments", int(len(seg)))
c3.metric("segment_seconds", seg_seconds)
c4.metric("model_key", model_key)

st.write(f"**Активност:** {act_row.get('name')} | start={act_row.get('start_date')} | distance={act_row.get('distance_m', 0)/1000:.2f} km")

tabs = st.tabs([
    "1) Приравняване (time series)",
    "2) Модели (slope/glide)",
    "3) CS/кислороден дълг",
    "4) Зони + HR (Вариант Б) + натоварване"
])

# -----------------------------
# Tab 1
# -----------------------------
with tabs[0]:
    dfp = seg[["seg_idx", "t_rel_s", "slope_pct", "v_kmh_raw", "v_flat_eq", "v_flat_eq_cs", "hr_mean", "valid_basic", "speed_spike"]].copy()
    dfp["t_min"] = dfp["t_rel_s"] / 60.0

    long = pd.melt(
        dfp,
        id_vars=["seg_idx", "t_min"],
        value_vars=["v_kmh_raw", "v_flat_eq", "v_flat_eq_cs"],
        var_name="metric",
        value_name="v_kmh",
    )

    speed_chart = (
        alt.Chart(long.dropna(subset=["v_kmh"]))
        .mark_line()
        .encode(
            x=alt.X("t_min:Q", title="време [min]"),
            y=alt.Y("v_kmh:Q", title="скорост [km/h]"),
            color=alt.Color("metric:N", title="серия"),
            tooltip=["seg_idx:Q", "t_min:Q", "metric:N", "v_kmh:Q"],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(speed_chart, use_container_width=True)

    slope_chart = (
        alt.Chart(dfp.dropna(subset=["slope_pct"]))
        .mark_line()
        .encode(
            x=alt.X("t_min:Q", title="време [min]"),
            y=alt.Y("slope_pct:Q", title="наклон [%]"),
            tooltip=["seg_idx:Q", "t_min:Q", "slope_pct:Q"],
        )
        .properties(height=180)
        .interactive()
    )
    st.altair_chart(slope_chart, use_container_width=True)

    if dfp["hr_mean"].notna().any():
        hr_chart = (
            alt.Chart(dfp.dropna(subset=["hr_mean"]))
            .mark_line()
            .encode(
                x=alt.X("t_min:Q", title="време [min]"),
                y=alt.Y("hr_mean:Q", title="HR mean [bpm]"),
                tooltip=["seg_idx:Q", "t_min:Q", "hr_mean:Q"],
            )
            .properties(height=180)
            .interactive()
        )
        st.altair_chart(hr_chart, use_container_width=True)

    st.dataframe(seg.head(80), use_container_width=True)


# -----------------------------
# Tab 2
# -----------------------------
with tabs[1]:
    base_col = "v_flat_eq_cs" if (use_cs_mod and seg["v_flat_eq_cs"].notna().any()) else "v_flat_eq"

    m = (
        seg["valid_basic"].fillna(True)
        & seg["slope_pct"].notna()
        & seg["v_kmh_raw"].notna()
        & (seg["v_kmh_raw"] > 0)
        & seg[base_col].notna()
    )
    dfm = seg.loc[m, ["slope_pct", "v_kmh_raw", base_col, "k_glide", "v_glide"]].copy()
    dfm["F_implied"] = dfm[base_col] / dfm["v_kmh_raw"]

    st.subheader("Наклонен модел (имплицитно F = v_eq / v_raw)")
    scat = (
        alt.Chart(dfm)
        .mark_circle(size=45, opacity=0.55)
        .encode(
            x=alt.X("slope_pct:Q", title="наклон [%]"),
            y=alt.Y("F_implied:Q", title="F (implied)"),
            tooltip=["slope_pct:Q", "v_kmh_raw:Q", f"{base_col}:Q", "F_implied:Q"],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(scat, use_container_width=True)

    if sport_group == "ski" and seg["k_glide"].notna().any():
        st.subheader("Плъзгаемост (glide) – по сегменти")
        gdf = seg.loc[seg["k_glide"].notna(), ["slope_pct", "k_glide", "v_kmh_raw", "v_glide"]].copy()

        glide_k = (
            alt.Chart(gdf.dropna(subset=["slope_pct", "k_glide"]))
            .mark_circle(size=45, opacity=0.55)
            .encode(
                x=alt.X("slope_pct:Q", title="наклон [%]"),
                y=alt.Y("k_glide:Q", title="k_glide"),
                tooltip=["slope_pct:Q", "k_glide:Q", "v_kmh_raw:Q", "v_glide:Q"],
            )
            .properties(height=280)
            .interactive()
        )
        st.altair_chart(glide_k, use_container_width=True)

        glide_v = (
            alt.Chart(gdf.dropna(subset=["v_kmh_raw", "v_glide"]))
            .mark_circle(size=45, opacity=0.55)
            .encode(
                x=alt.X("v_kmh_raw:Q", title="v_raw [km/h]"),
                y=alt.Y("v_glide:Q", title="v_glide [km/h]"),
                tooltip=["slope_pct:Q", "k_glide:Q", "v_kmh_raw:Q", "v_glide:Q"],
            )
            .properties(height=280)
            .interactive()
        )
        st.altair_chart(glide_v, use_container_width=True)

    st.caption("Бележка: тук показваме имплицитния модел от резултатите. Ако искаш да рисуваме самите полиноми, трябва да записваме coeffs в DB (model_fits).")


# -----------------------------
# Tab 3
# -----------------------------
with tabs[2]:
    st.subheader("CS модулатор (кислороден дълг / натрупване)")
    cols = ["seg_idx", "t_rel_s", "v_base", "delta_v_plus_kmh", "r_kmh", "tau_s"]
    dfc = seg[cols].copy()
    dfc["t_min"] = dfc["t_rel_s"] / 60.0

    v_chart = (
        alt.Chart(dfc.dropna(subset=["v_base"]))
        .mark_line()
        .encode(
            x=alt.X("t_min:Q", title="време [min]"),
            y=alt.Y("v_base:Q", title="v (eq or eq_cs) [km/h]"),
            tooltip=["seg_idx:Q", "t_min:Q", "v_base:Q"],
        )
        .properties(height=200)
        .interactive()
    )
    st.altair_chart(v_chart, use_container_width=True)

    dv_chart = (
        alt.Chart(dfc.dropna(subset=["delta_v_plus_kmh"]))
        .mark_line()
        .encode(
            x=alt.X("t_min:Q", title="време [min]"),
            y=alt.Y("delta_v_plus_kmh:Q", title="Δv+ [km/h]"),
            tooltip=["seg_idx:Q", "t_min:Q", "delta_v_plus_kmh:Q"],
        )
        .properties(height=200)
        .interactive()
    )
    st.altair_chart(dv_chart, use_container_width=True)

    r_chart = (
        alt.Chart(dfc.dropna(subset=["r_kmh"]))
        .mark_line()
        .encode(
            x=alt.X("t_min:Q", title="време [min]"),
            y=alt.Y("r_kmh:Q", title="r [km/h]"),
            tooltip=["seg_idx:Q", "t_min:Q", "r_kmh:Q"],
        )
        .properties(height=200)
        .interactive()
    )
    st.altair_chart(r_chart, use_container_width=True)

    tau_chart = (
        alt.Chart(dfc.dropna(subset=["tau_s"]))
        .mark_line()
        .encode(
            x=alt.X("t_min:Q", title="време [min]"),
            y=alt.Y("tau_s:Q", title="tau [s]"),
            tooltip=["seg_idx:Q", "t_min:Q", "tau_s:Q"],
        )
        .properties(height=200)
        .interactive()
    )
    st.altair_chart(tau_chart, use_container_width=True)


# -----------------------------
# Tab 4: Zones + HR (Variant B)
# -----------------------------
with tabs[3]:
    st.subheader("Зони (единна система) + пулс по твоята методика (Вариант Б)")

    # speed-zone summary (time/distance/load)
    ztab = (
        seg.groupby("zone_u", dropna=False)
        .agg(
            time_s=("dt_s", "sum"),
            dist_m=("d_m", "sum"),
            load=("load_u", "sum"),
            v_mean=("v_base", "mean"),
        )
        .reset_index()
    )
    ztab["time_min"] = ztab["time_s"] / 60.0
    ztab["dist_km"] = ztab["dist_m"] / 1000.0

    # HR by Variant B allocation
    hr_alloc = compute_zone_hr_sorted_allocation(seg, zone_col="zone_u", hr_col="hr_mean", time_col="dt_s")
    hr_alloc = hr_alloc.rename(columns={"zone": "zone_u"})

    ztab = ztab.merge(hr_alloc, on="zone_u", how="left")

    # order zones
    zone_order = ["Z1","Z2","Z3","Z4","Z5","Z6","Z?"]
    ztab["zone_u"] = pd.Categorical(ztab["zone_u"], categories=zone_order, ordered=True)
    ztab = ztab.sort_values("zone_u")

    st.caption(
        "HR по зона тук се изчислява така: времето по скоростни зони определя целевите секунди за всяка зона, "
        "след което сегментите с наличен HR се сортират по HR↑ и се 'разпределят' последователно по време към Z1..Z6."
    )

    st.dataframe(
        ztab[["zone_u", "time_min", "dist_km", "v_mean", "load", "mean_hr_zone"]],
        use_container_width=True
    )

    zbar = (
        alt.Chart(ztab.dropna(subset=["time_min"]))
        .mark_bar()
        .encode(
            x=alt.X("zone_u:N", title="зона"),
            y=alt.Y("time_min:Q", title="време [min]"),
            tooltip=["zone_u:N", "time_min:Q", "dist_km:Q", "load:Q", "mean_hr_zone:Q"],
        )
        .properties(height=220)
    )
    st.altair_chart(zbar, use_container_width=True)

    # activity totals
    total_time = float(seg["dt_s"].sum())
    total_dist = float(seg["d_m"].sum())
    total_load = float(seg["load_u"].sum())
    vmean = float(np.nanmean(seg["v_base"].to_numpy(dtype=float)))

    st.markdown("### Обобщение за активността")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time [min]", f"{total_time/60:.1f}")
    c2.metric("Distance [km]", f"{total_dist/1000:.2f}")
    c3.metric("Mean v (eq)", f"{vmean:.2f} km/h")
    c4.metric("Load (Σ dt·I²)", f"{total_load:.1f}")

    st.caption("Load тук е проста стартова метрика (Σ dt · intensity²). След това можем да я заменим с твоята финална формула.")
