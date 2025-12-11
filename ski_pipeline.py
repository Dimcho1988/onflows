# ski_pipeline.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from cs_modulator import apply_cs_modulation

# ---------------------------------------------------------
# НАСТРОЙКИ (същите, както в ski Streamlit app-а)
# ---------------------------------------------------------
T_SEG = 7.0            # дължина на сегмента [s]
MIN_D_SEG = 5.0        # минимум хоризонтална дистанция [m]
MIN_T_SEG = 4.0        # минимум продължителност [s]
MAX_ABS_SLOPE = 15.0   # макс. наклон [%]
V_JUMP_KMH = 15.0      # праг за "скачане" на скоростта между сегменти
V_JUMP_MIN = 20.0      # гледаме спайкове само над тази скорост [km/h]

GLIDE_POLY_DEG = 2     # степен на полинома за плъзгаемост
SLOPE_POLY_DEG = 2     # степен на полинома за наклон

# Граници за наклоновия коефициент (както при теб)
F_MIN_SLOPE = 0.7
F_MAX_SLOPE = 1.7


# ---------------------------------------------------------
# ВСПОМОГАТЕЛНИ
# ---------------------------------------------------------
def build_points_from_streams(act_row: dict, streams: dict) -> pd.DataFrame:
    """
    streams: dict като от Supabase activity_streams, но вече key_by_type:
      {
        "time": {"data": [...]},
        "distance": {"data": [...]},
        "altitude": {"data": [...]},
        "heartrate": {"data": [...]},
      }
    act_row: ред от таблицата activities (има start_date).
    """
    time_arr = np.array(streams.get("time", {}).get("data", []), dtype=float)
    dist_arr = np.array(streams.get("distance", {}).get("data", []), dtype=float)
    alt_arr = np.array(streams.get("altitude", {}).get("data", []), dtype=float)
    hr_arr = np.array(streams.get("heartrate", {}).get("data", []), dtype=float)

    n = len(time_arr)
    if n == 0:
        return pd.DataFrame(columns=["time", "dist", "elev", "hr"])

    start = datetime.fromisoformat(act_row["start_date"])
    times = [start + timedelta(seconds=float(t)) for t in time_arr]

    df = pd.DataFrame({
        "time": times,
        "dist": dist_arr if len(dist_arr) == n else np.nan,
        "elev": alt_arr if len(alt_arr) == n else np.nan,
        "hr": hr_arr if len(hr_arr) == n else np.nan,
    })
    df["dist"] = df["dist"].ffill()
    return df


# ---------------------------------------------------------
# СЕГМЕНТИРАНЕ НА 7 s (с hr_mean)
# ---------------------------------------------------------
def build_segments(df_activity: pd.DataFrame, activity_id: int) -> pd.DataFrame:
    if df_activity.empty:
        return pd.DataFrame()

    df_activity = df_activity.sort_values("time").reset_index(drop=True)

    times = df_activity["time"].to_numpy()
    elevs = df_activity["elev"].to_numpy()
    dists = df_activity["dist"].to_numpy()
    hrs = df_activity["hr"].to_numpy()

    n = len(df_activity)
    start_idx = 0
    seg_idx = 0
    seg_rows = []

    while start_idx < n - 1:
        t0 = times[start_idx]

        end_idx = start_idx + 1
        while end_idx < n:
            dt_tmp = (times[end_idx] - t0) / np.timedelta64(1, "s")
            if dt_tmp >= T_SEG:
                break
            end_idx += 1

        if end_idx >= n:
            break

        t1 = times[end_idx]
        dt = (t1 - t0) / np.timedelta64(1, "s")

        d0 = dists[start_idx]
        d1 = dists[end_idx]
        elev0 = elevs[start_idx]
        elev1 = elevs[end_idx]
        d_m = max(0.0, d1 - d0)

        if dt < MIN_T_SEG or d_m < MIN_D_SEG:
            start_idx = end_idx
            continue

        if any(pd.isna([elev0, elev1])):
            slope = np.nan
        else:
            slope = (elev1 - elev0) / d_m * 100.0 if d_m > 0 else np.nan

        v_kmh = (d_m / dt) * 3.6
        hr_mean = float(np.nanmean(hrs[start_idx:end_idx + 1]))

        seg_rows.append({
            "activity_id": activity_id,
            "seg_idx": seg_idx,
            "t_start": pd.to_datetime(t0),
            "t_end": pd.to_datetime(t1),
            "dt_s": float(dt),
            "d_m": float(d_m),
            "slope_pct": float(slope) if not np.isnan(slope) else np.nan,
            "v_kmh_raw": float(v_kmh),
            "hr_mean": hr_mean
        })

        seg_idx += 1
        start_idx = end_idx

    return pd.DataFrame(seg_rows)


# ---------------------------------------------------------
# БАЗОВИ ФИЛТРИ (наклон + speed spikes)
# ---------------------------------------------------------
def apply_basic_filters(segments: pd.DataFrame) -> pd.DataFrame:
    seg = segments.copy()

    valid_slope = seg["slope_pct"].between(-MAX_ABS_SLOPE, MAX_ABS_SLOPE)
    valid_slope &= seg["slope_pct"].notna()
    seg["valid_basic"] = valid_slope

    def mark_speed_spikes(group):
        group = group.sort_values("seg_idx").copy()
        spike = np.zeros(len(group), dtype=bool)
        v = group["v_kmh_raw"].values
        for i in range(1, len(group)):
            dv = abs(v[i] - v[i-1])
            vmax = max(v[i], v[i-1])
            if dv > V_JUMP_KMH and vmax > V_JUMP_MIN:
                spike[i] = True
        group["speed_spike"] = spike
        return group

    seg = seg.groupby("activity_id", group_keys=False).apply(mark_speed_spikes)
    seg["speed_spike"] = seg["speed_spike"].fillna(False)
    seg.loc[seg["speed_spike"], "valid_basic"] = False

    return seg


# ---------------------------------------------------------
# GLIDE МОДЕЛ
# ---------------------------------------------------------
def get_glide_training_segments(seg: pd.DataFrame) -> pd.DataFrame:
    """
    Сегменти за модел на плъзгаемостта:
    - сегментът и предишният са валидни
    - и двата имат наклон <= -5%  (спускания)
    """
    df = seg.copy()
    df["prev_slope"] = df.groupby("activity_id")["slope_pct"].shift(1)
    df["prev_valid"] = df.groupby("activity_id")["valid_basic"].shift(1)

    cond = (
        df["valid_basic"]
        & df["prev_valid"].fillna(False)
        & (df["slope_pct"] <= -5.0)
        & (df["prev_slope"] <= -5.0)
    )

    train = df[cond].copy()
    return train


def fit_glide_poly(train_df: pd.DataFrame):
    if train_df.empty:
        return None
    x = train_df["slope_pct"].values.astype(float)
    y = train_df["v_kmh_raw"].values.astype(float)
    if len(x) <= GLIDE_POLY_DEG:
        return None
    coeffs = np.polyfit(x, y, GLIDE_POLY_DEG)
    return np.poly1d(coeffs)


def compute_glide_coefficients(seg: pd.DataFrame, glide_poly, DAMP_GLIDE: float):
    """
    Връща dict activity_id -> K_glide.
    1) k_raw = V_model / V_real
    2) клипваме k_raw в [0.90, 1.25]
    3) омекотяваме към 1 с коефициент DAMP_GLIDE:
       K = 1 + α * (k_clipped - 1)
    """
    train = get_glide_training_segments(seg)
    if glide_poly is None or train.empty:
        return {}

    coeffs_dict = {}
    for aid, g in train.groupby("activity_id"):
        s_mean = g["slope_pct"].mean()
        v_real = g["v_kmh_raw"].mean()
        if v_real <= 0:
            continue
        v_model = float(glide_poly(s_mean))
        if v_model <= 0:
            continue

        k_raw = v_model / v_real
        k_clipped = max(0.9, min(1.25, k_raw))
        k_final = 1.0 + DAMP_GLIDE * (k_clipped - 1.0)
        coeffs_dict[aid] = k_final

    return coeffs_dict


def apply_glide_modulation(seg: pd.DataFrame, glide_coeffs: dict) -> pd.DataFrame:
    seg = seg.copy()
    seg["k_glide"] = seg["activity_id"].map(glide_coeffs).fillna(1.0)
    seg["v_glide"] = seg["v_kmh_raw"] * seg["k_glide"]
    return seg


# ---------------------------------------------------------
# SLOPE МОДЕЛ (както при теб – с F(0)=1)
# ---------------------------------------------------------
def compute_flat_ref_speeds(seg_glide: pd.DataFrame) -> dict:
    flat_refs = {}
    for aid, g in seg_glide.groupby("activity_id"):
        mask_flat = g["slope_pct"].between(-1.0, 1.0) & g["valid_basic"]
        g_flat = g[mask_flat]
        if g_flat.empty:
            continue
        v_flat = g_flat["v_glide"].mean()
        if v_flat > 0:
            flat_refs[aid] = v_flat
    return flat_refs


def get_slope_training_data(seg_glide: pd.DataFrame, flat_refs: dict) -> pd.DataFrame:
    df = seg_glide.copy()
    df["V_flat_ref"] = df["activity_id"].map(flat_refs)
    mask = (
        df["valid_basic"]
        & df["slope_pct"].between(-15.0, 15.0)
        & df["V_flat_ref"].notna()
        & (df["v_glide"] > 0)
    )
    train = df[mask].copy()
    if train.empty:
        return pd.DataFrame(columns=["slope_pct", "F"])
    train["F"] = train["V_flat_ref"] / train["v_glide"]
    return train[["slope_pct", "F"]]


def fit_slope_poly(train_df: pd.DataFrame):
    if train_df.empty:
        return None
    x = train_df["slope_pct"].values.astype(float)
    y = train_df["F"].values.astype(float)
    if len(x) <= SLOPE_POLY_DEG:
        return None
    coeffs = np.polyfit(x, y, SLOPE_POLY_DEG)
    raw_poly = np.poly1d(coeffs)
    # корекция F(0)=1
    F0 = float(raw_poly(0.0))
    offset = F0 - 1.0
    coeffs_corr = raw_poly.coefficients.copy()
    coeffs_corr[-1] -= offset
    return np.poly1d(coeffs_corr)


def apply_slope_modulation(seg_glide: pd.DataFrame, slope_poly, V_crit: float) -> pd.DataFrame:
    df = seg_glide.copy()
    if slope_poly is None:
        df["v_flat_eq"] = df["v_glide"]
        return df

    slopes = df["slope_pct"].values.astype(float)
    F_vals = slope_poly(slopes)
    F_vals = np.clip(F_vals, F_MIN_SLOPE, F_MAX_SLOPE)

    # |slope| <= 1% -> F = 1
    mask_mid = np.abs(slopes) <= 1.0
    F_vals[mask_mid] = 1.0
    # спускане <-1% -> F<=1
    mask_down = slopes < -1.0
    F_vals[mask_down] = np.minimum(F_vals[mask_down], 1.0)
    # изкачване >1% -> F>=1
    mask_up = slopes > 1.0
    F_vals[mask_up] = np.maximum(F_vals[mask_up], 1.0)

    v_flat_eq = df["v_glide"].values * F_vals

    # силни спускания под -3% -> 70% от V_crit
    if V_crit is not None and V_crit > 0:
        idx_below = df["slope_pct"] < -3.0
        v_flat_eq[idx_below] = 0.7 * V_crit

    df["v_flat_eq"] = v_flat_eq
    return df


# ---------------------------------------------------------
# ЧИСТЕНЕ НА v_flat_eq ПРЕДИ CS
# ---------------------------------------------------------
def clean_speed_for_cs(g: pd.DataFrame, v_max_cs=50.0) -> np.ndarray:
    v = g["v_flat_eq"].to_numpy(dtype=float)
    v = np.clip(v, 0.0, v_max_cs)

    if "speed_spike" in g.columns:
        is_spike = g["speed_spike"].to_numpy(dtype=bool)
    else:
        is_spike = np.zeros_like(v, dtype=bool)

    if not is_spike.any():
        return v

    v_clean = v.copy()
    idx = np.arange(len(v_clean))
    good = ~is_spike
    if good.sum() <= 1:
        return v_clean

    v_clean[is_spike] = np.interp(idx[is_spike], idx[good], v_clean[good])
    return v_clean


# ---------------------------------------------------------
# ГЛАВНА ФУНКЦИЯ ЗА СКИ АКТИВНОСТ
# ---------------------------------------------------------
def process_ski_activity(
    act_row: dict,
    streams: dict,
    V_crit: float,
    DAMP_GLIDE: float,
    CS: float,
    tau_min: float,
    k_par: float,
    q_par: float,
    gamma_cs: float,
) -> pd.DataFrame:
    """
    Основен pipeline за ски активност:

      1) Strava streams -> points
      2) 7s сегменти
      3) базови филтри (наклон + speed spikes)
      4) glide модел (v_glide)
      5) slope модел (v_flat_eq)
      6) CS модел върху v_flat_eq -> v_flat_eq_cs

    Връща DataFrame със сегментите и колони:
      activity_id, seg_idx, t_start, t_end, dt_s, d_m,
      slope_pct, v_kmh_raw, valid_basic, speed_spike,
      k_glide, v_glide, v_flat_eq, v_flat_eq_cs,
      time_s, delta_v_plus_kmh, r_kmh, tau_s,
      hr_mean
    """
    activity_id = act_row["id"]

    # 1) points
    points = build_points_from_streams(act_row, streams)
    if points.empty:
        return pd.DataFrame()

    # 2) сегменти
    segments = build_segments(points, activity_id=activity_id)
    if segments.empty:
        return pd.DataFrame()

    # 3) базови филтри
    segments_f = apply_basic_filters(segments)

    # 4) GLIDE
    train_glide = get_glide_training_segments(segments_f)
    glide_poly = fit_glide_poly(train_glide)
    if glide_poly is None:
        glide_coeffs = {}
    else:
        glide_coeffs = compute_glide_coefficients(segments_f, glide_poly, DAMP_GLIDE)
    seg_glide = apply_glide_modulation(segments_f, glide_coeffs)

    # 5) SLOPE
    flat_refs = compute_flat_ref_speeds(seg_glide)
    slope_train = get_slope_training_data(seg_glide, flat_refs)
    slope_poly = fit_slope_poly(slope_train)
    seg_slope = apply_slope_modulation(seg_glide, slope_poly, V_crit)

    # 6) CS върху v_flat_eq
    seg_slope = seg_slope.sort_values("t_start").reset_index(drop=True)
    seg_slope["time_s"] = seg_slope["dt_s"].cumsum() - seg_slope["dt_s"]

    v_clean = clean_speed_for_cs(seg_slope, v_max_cs=50.0)
    dt_arr = seg_slope["dt_s"].to_numpy(dtype=float)

    out_cs = apply_cs_modulation(
        v=v_clean,
        dt=dt_arr,
        CS=CS,
        tau_min=tau_min,
        k_par=k_par,
        q_par=q_par,
        gamma=gamma_cs,
    )

    seg_slope["v_flat_eq_cs"] = out_cs["v_mod"]
    seg_slope["delta_v_plus_kmh"] = out_cs["delta_v_plus"]
    seg_slope["r_kmh"] = out_cs["r"]
    seg_slope["tau_s"] = out_cs["tau_s"]

    # уверяваме се, че имаме всички нужни колони
    needed_cols_default = {
        "activity_id": activity_id,
        "k_glide": np.nan,
        "v_glide": np.nan,
        "v_flat_eq": np.nan,
        "v_flat_eq_cs": np.nan,
        "delta_v_plus_kmh": 0.0,
        "r_kmh": 0.0,
        "tau_s": tau_min,
        "valid_basic": True,
        "speed_spike": False,
    }
    for col, default_val in needed_cols_default.items():
        if col not in seg_slope.columns:
            seg_slope[col] = default_val

    return seg_slope
