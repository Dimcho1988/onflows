import numpy as np
import pandas as pd
from datetime import timedelta
from dateutil.parser import isoparse

from cs_modulator import apply_cs_modulation


MAX_ABS_SLOPE = 30.0
MIN_SPEED_FOR_MODEL = 3.0
MAX_SPEED_FOR_MODEL = 30.0

SLOPE_POLY_DEG = 2
F_MIN_SLOPE = 0.7
F_MAX_SLOPE = 1.7

T_SEG = 30.0
MIN_D_SEG = 10.0
MIN_T_SEG = 15.0


def build_points_from_streams(act_row: dict, streams: dict) -> pd.DataFrame:
    time_arr = np.array(streams.get("time", {}).get("data", []), dtype=float)
    dist_arr = np.array(streams.get("distance", {}).get("data", []), dtype=float)
    alt_arr = np.array(streams.get("altitude", {}).get("data", []), dtype=float)
    hr_arr = np.array(streams.get("heartrate", {}).get("data", []), dtype=float)

    n = len(time_arr)
    if n == 0:
        return pd.DataFrame(columns=["time", "dist", "elev", "hr"])

    start = isoparse(act_row["start_date"])
    times = [start + timedelta(seconds=float(t)) for t in time_arr]

    dist = dist_arr if len(dist_arr) == n else np.full(n, np.nan)
    elev = alt_arr if len(alt_arr) == n else np.full(n, np.nan)
    hr = hr_arr if len(hr_arr) == n else np.full(n, np.nan)

    df = pd.DataFrame({"time": times, "dist": dist, "elev": elev, "hr": hr})
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)

    df["dist"] = pd.to_numeric(df["dist"], errors="coerce").ffill()
    return df


def build_segments_30s(df, activity_id):
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("time").reset_index(drop=True)
    times = df["time"].to_numpy(dtype="datetime64[ns]")
    elevs = df["elev"].to_numpy(dtype=float)
    dists = df["dist"].to_numpy(dtype=float)
    hrs = df["hr"].to_numpy(dtype=float)

    n = len(df)
    start_idx = 0
    seg_idx = 0
    rows = []

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

        slope = (elev1 - elev0) / d_m * 100.0 if (d_m > 0 and np.isfinite(elev0) and np.isfinite(elev1)) else np.nan
        v_kmh = (d_m / dt) * 3.6
        hr_mean = float(np.nanmean(hrs[start_idx:end_idx + 1]))

        rows.append({
            "activity_id": activity_id,
            "seg_idx": seg_idx,
            "t_start": pd.to_datetime(t0, utc=True),
            "t_end": pd.to_datetime(t1, utc=True),
            "dt_s": float(dt),
            "d_m": float(d_m),
            "slope_pct": float(slope) if np.isfinite(slope) else np.nan,
            "v_kmh_raw": float(v_kmh),
            "hr_mean": hr_mean,
        })

        seg_idx += 1
        start_idx = end_idx

    return pd.DataFrame(rows)


def apply_basic_filters(segments: pd.DataFrame) -> pd.DataFrame:
    seg = segments.copy()
    valid_slope = seg["slope_pct"].between(-MAX_ABS_SLOPE, MAX_ABS_SLOPE) & seg["slope_pct"].notna()
    valid_speed = seg["v_kmh_raw"].between(0.0, MAX_SPEED_FOR_MODEL * 1.2)
    seg["valid_basic"] = valid_slope & valid_speed

    seg["speed_spike"] = False
    for aid, g in seg.groupby("activity_id"):
        g = g.sort_values("seg_idx")
        v = g["v_kmh_raw"].values
        spike = np.zeros(len(g), dtype=bool)
        for i in range(1, len(g)):
            dv = abs(v[i] - v[i - 1])
            vmax = max(v[i], v[i - 1])
            if dv > 15.0 and vmax > 10.0:
                spike[i] = True
        seg.loc[g.index, "speed_spike"] = spike
    seg.loc[seg["speed_spike"], "valid_basic"] = False
    return seg


def compute_global_flat_speed(seg_f: pd.DataFrame) -> float | None:
    df = seg_f.copy()
    mask_flat = (
        df["valid_basic"]
        & df["slope_pct"].between(-1.0, 1.0)
        & (df["v_kmh_raw"] > MIN_SPEED_FOR_MODEL)
        & (df["v_kmh_raw"] < MAX_SPEED_FOR_MODEL)
    )
    flat = df.loc[mask_flat]
    if flat.empty:
        return None
    return float(flat["v_kmh_raw"].mean())


def get_slope_training_data(seg_f: pd.DataFrame, V0: float) -> pd.DataFrame:
    if V0 is None or V0 <= 0:
        return pd.DataFrame(columns=["slope_pct", "F_raw"])

    df = seg_f.copy()
    mask = (
        df["valid_basic"]
        & df["slope_pct"].between(-MAX_ABS_SLOPE, MAX_ABS_SLOPE)
        & (df["v_kmh_raw"] > MIN_SPEED_FOR_MODEL)
        & (df["v_kmh_raw"] < MAX_SPEED_FOR_MODEL)
    )
    train = df.loc[mask, ["slope_pct", "v_kmh_raw"]].copy()
    if train.empty:
        return pd.DataFrame(columns=["slope_pct", "F_raw"])

    train["F_raw"] = V0 / train["v_kmh_raw"]
    return train


def fit_slope_poly(train_df: pd.DataFrame):
    if train_df.empty:
        return None
    x = train_df["slope_pct"].values.astype(float)
    y = train_df["F_raw"].values.astype(float)
    if len(x) <= 2:
        return None
    coeffs = np.polyfit(x, y, SLOPE_POLY_DEG)
    raw_poly = np.poly1d(coeffs)
    F0 = float(raw_poly(0.0))
    offset = F0 - 1.0
    coeffs_corr = raw_poly.coefficients.copy()
    coeffs_corr[-1] -= offset
    return np.poly1d(coeffs_corr)


def compute_slope_F(slopes: np.ndarray, slope_poly, alpha_slope: float) -> np.ndarray:
    slopes = np.asarray(slopes, dtype=float)
    if slope_poly is None:
        return np.ones_like(slopes, dtype=float)

    F_model = slope_poly(slopes)
    F = np.clip(F_model, F_MIN_SLOPE, F_MAX_SLOPE)

    abs_s = np.abs(slopes)
    mask_mid = abs_s <= 1.0
    F[mask_mid] = 1.0
    mask_down = slopes < -1.0
    F[mask_down] = np.minimum(F[mask_down], 1.0)
    mask_up = slopes > 1.0
    F[mask_up] = np.maximum(F[mask_up], 1.0)

    F = 1.0 + alpha_slope * (F - 1.0)
    return F


def clean_speed_for_cs(g: pd.DataFrame, v_max_cs=30.0) -> np.ndarray:
    v = g["v_flat_eq"].to_numpy(dtype=float)
    v = np.clip(v, 0.0, v_max_cs)
    is_spike = g["speed_spike"].to_numpy(dtype=bool) if "speed_spike" in g.columns else np.zeros_like(v, dtype=bool)
    if not is_spike.any():
        return v
    v_clean = v.copy()
    idx = np.arange(len(v_clean))
    good = ~is_spike
    if good.sum() <= 1:
        return v_clean
    v_clean[is_spike] = np.interp(idx[is_spike], idx[good], v_clean[good])
    return v_clean


def process_run_walk_activity(
    act_row: dict, streams: dict,
    alpha_slope: float,
    V_crit: float, CS: float,
    tau_min: float, k_par: float, q_par: float, gamma_cs: float
) -> pd.DataFrame:
    points = build_points_from_streams(act_row, streams)
    seg = build_segments_30s(points, activity_id=act_row["id"])
    if seg.empty:
        return seg

    seg = apply_basic_filters(seg)
    V0 = compute_global_flat_speed(seg)
    slope_train = get_slope_training_data(seg, V0)
    slope_poly = fit_slope_poly(slope_train)

    seg["v_flat_eq"] = seg["v_kmh_raw"]
    if slope_poly is not None:
        F_vals = compute_slope_F(seg["slope_pct"].values, slope_poly, alpha_slope)
        v_flat_eq = seg["v_kmh_raw"].values * F_vals

        # Run/walk: cap only (not equalize)
        if V_crit and V_crit > 0:
            idx_below = seg["slope_pct"] < -5.0
            v_flat_eq[idx_below] = np.minimum(v_flat_eq[idx_below], 0.8 * V_crit)

        seg["v_flat_eq"] = v_flat_eq

    seg = seg.sort_values("t_start").reset_index(drop=True)
    v_clean = clean_speed_for_cs(seg, v_max_cs=MAX_SPEED_FOR_MODEL)
    dt_arr = seg["dt_s"].to_numpy(dtype=float)

    out_cs = apply_cs_modulation(
        v=v_clean,
        dt=dt_arr,
        CS=CS,
        tau_min=tau_min,
        k_par=k_par,
        q_par=q_par,
        gamma=gamma_cs,
    )

    seg["v_flat_eq_cs"] = out_cs["v_mod"]
    seg["delta_v_plus_kmh"] = out_cs["delta_v_plus"]
    seg["r_kmh"] = out_cs["r"]
    seg["tau_s"] = out_cs["tau_s"]

    # run/walk doesn't use glide but keep columns for DB compatibility
    seg["k_glide"] = 1.0
    seg["v_glide"] = seg["v_kmh_raw"]

    return seg
