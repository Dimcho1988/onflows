# ski_pipeline.py
import numpy as np
import pandas as pd
from datetime import timedelta
from dateutil.parser import isoparse

from cs_modulator import apply_cs_modulation


# =========================
# 7s settings (internal)
# =========================
T_SEG_7 = 7.0
MIN_D_SEG_7 = 5.0
MIN_T_SEG_7 = 4.0
MAX_ABS_SLOPE_7 = 15.0

V_JUMP_KMH = 15.0
V_JUMP_MIN = 20.0

GLIDE_POLY_DEG = 2
SLOPE_POLY_DEG = 2

# Slope model constraints (F)
F_MIN_SLOPE = 0.7
F_MAX_SLOPE = 1.7
SLOPE_TRAIN_MAX_ABS = 15.0
SLOPE_EPS_FLAT = 1.0  # |slope|<=1% is "flat reference", excluded from slope fit

# =========================
# OUT settings (30s output)
# =========================
T_SEG_OUT = 30.0
MIN_D_SEG_OUT = 10.0
MIN_T_SEG_OUT = 15.0


# =========================
# Build points
# =========================
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
    df["elev"] = pd.to_numeric(df["elev"], errors="coerce")
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce")
    return df


# =========================
# Segmentation fixed seconds
# =========================
def build_segments_fixed_seconds(
    df: pd.DataFrame,
    activity_id: int,
    T_SEG: float,
    MIN_T: float,
    MIN_D: float
) -> pd.DataFrame:
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
        e0 = elevs[start_idx]
        e1 = elevs[end_idx]

        d_m = max(0.0, float(d1 - d0))
        if dt < MIN_T or d_m < MIN_D:
            start_idx = end_idx
            continue

        slope = np.nan
        if d_m > 0 and np.isfinite(e0) and np.isfinite(e1):
            slope = (float(e1 - e0) / d_m) * 100.0

        v_kmh = (d_m / float(dt)) * 3.6
        hr_mean = float(np.nanmean(hrs[start_idx:end_idx + 1]))

        rows.append({
            "activity_id": int(activity_id),
            "seg_idx": int(seg_idx),
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


# =========================
# Basic filters + speed spikes
# =========================
def apply_basic_filters_7s(segments: pd.DataFrame) -> pd.DataFrame:
    seg = segments.copy()

    valid_slope = seg["slope_pct"].between(-MAX_ABS_SLOPE_7, MAX_ABS_SLOPE_7) & seg["slope_pct"].notna()
    seg["valid_basic"] = valid_slope

    seg["speed_spike"] = False
    for aid, g in seg.groupby("activity_id"):
        g = g.sort_values("seg_idx")
        v = g["v_kmh_raw"].to_numpy(dtype=float)
        spike = np.zeros(len(g), dtype=bool)
        for i in range(1, len(g)):
            dv = abs(v[i] - v[i - 1])
            vmax = max(v[i], v[i - 1])
            if dv > V_JUMP_KMH and vmax > V_JUMP_MIN:
                spike[i] = True
        seg.loc[g.index, "speed_spike"] = spike

    seg.loc[seg["speed_spike"], "valid_basic"] = False
    seg["speed_spike"] = seg["speed_spike"].fillna(False).astype(bool)
    seg["valid_basic"] = seg["valid_basic"].fillna(False).astype(bool)
    return seg


# =========================
# GLIDE training (consecutive downhills)
# =========================
def get_glide_training_segments(seg7: pd.DataFrame) -> pd.DataFrame:
    df = seg7.copy()
    df = df.sort_values(["activity_id", "seg_idx"]).reset_index(drop=True)

    df["prev_slope"] = df.groupby("activity_id")["slope_pct"].shift(1)
    df["prev_valid"] = df.groupby("activity_id")["valid_basic"].shift(1)

    cond = (
        df["valid_basic"]
        & df["prev_valid"].fillna(False)
        & (df["slope_pct"] <= -5.0)
        & (df["prev_slope"] <= -5.0)
    )
    return df.loc[cond].copy()


def fit_glide_poly(train_df: pd.DataFrame):
    if train_df is None or train_df.empty:
        return None
    x = train_df["slope_pct"].to_numpy(dtype=float)
    y = train_df["v_kmh_raw"].to_numpy(dtype=float)
    if len(x) <= GLIDE_POLY_DEG:
        return None
    coeffs = np.polyfit(x, y, GLIDE_POLY_DEG)
    return np.poly1d(coeffs)


def compute_k_glide_for_activity(seg7: pd.DataFrame, glide_poly, DAMP_GLIDE: float) -> float:
    """
    Per-activity K_glide (както в TCX app):
      - взимаме downhill consecutive 7s сегменти от ТАЗИ активност
      - k_raw = v_model(s_mean) / v_real_mean
      - clip k_raw в [0.90, 1.25]
      - k_final = 1 + DAMP_GLIDE*(k_clip-1)
    """
    if glide_poly is None:
        return 1.0

    train = get_glide_training_segments(seg7)
    if train.empty:
        return 1.0

    s_mean = float(train["slope_pct"].mean())
    v_real = float(train["v_kmh_raw"].mean())
    if not np.isfinite(v_real) or v_real <= 0:
        return 1.0

    v_model = float(glide_poly(s_mean))
    if not np.isfinite(v_model) or v_model <= 0:
        return 1.0

    k_raw = v_model / v_real
    k_clip = max(0.90, min(1.25, k_raw))
    k_final = 1.0 + float(DAMP_GLIDE) * (k_clip - 1.0)
    return float(k_final)


def apply_glide_modulation_7s(seg7: pd.DataFrame, k_glide: float) -> pd.DataFrame:
    seg = seg7.copy()
    seg["k_glide"] = float(k_glide)
    seg["v_glide"] = seg["v_kmh_raw"] * seg["k_glide"]
    return seg


# =========================
# Aggregate 7s -> OUT (30s)
# =========================
def aggregate_7s_to_out(seg7: pd.DataFrame, activity_id: int) -> pd.DataFrame:
    if seg7 is None or seg7.empty:
        return pd.DataFrame()

    df = seg7.sort_values("t_start").reset_index(drop=True).copy()

    rows = []
    buf = []
    t_sum = 0.0
    seg_idx_out = 0

    def wavg(series: pd.Series, w: np.ndarray) -> float:
        s = series.to_numpy(dtype=float)
        mask = np.isfinite(s) & np.isfinite(w) & (w > 0)
        if mask.sum() == 0:
            return np.nan
        return float(np.sum(s[mask] * w[mask]) / np.sum(w[mask]))

    def flush(buf_rows, seg_idx_local):
        if not buf_rows:
            return None
        g = pd.DataFrame(buf_rows)

        dt = float(g["dt_s"].sum())
        d = float(g["d_m"].sum())
        if dt < MIN_T_SEG_OUT or d < MIN_D_SEG_OUT:
            return None

        t_start = g["t_start"].iloc[0]
        t_end = g["t_end"].iloc[-1]

        w_t = g["dt_s"].to_numpy(dtype=float)
        w_d = g["d_m"].to_numpy(dtype=float)

        slope = wavg(g["slope_pct"], w_d)
        v_raw = (d / dt) * 3.6
        v_glide = wavg(g["v_glide"], w_t)
        k_glide = wavg(g["k_glide"], w_t)
        hr_mean = wavg(g["hr_mean"], w_t)

        valid_basic = bool(g["valid_basic"].all())
        speed_spike = bool(g["speed_spike"].any())

        return {
            "activity_id": int(activity_id),
            "seg_idx": int(seg_idx_local),
            "t_start": t_start,
            "t_end": t_end,
            "dt_s": float(dt),
            "d_m": float(d),
            "slope_pct": float(slope) if np.isfinite(slope) else np.nan,
            "v_kmh_raw": float(v_raw),
            "valid_basic": bool(valid_basic),
            "speed_spike": bool(speed_spike),
            "k_glide": float(k_glide) if np.isfinite(k_glide) else 1.0,
            "v_glide": float(v_glide) if np.isfinite(v_glide) else float(v_raw),
            "hr_mean": float(hr_mean) if np.isfinite(hr_mean) else np.nan,
        }

    for _, r in df.iterrows():
        buf.append(r.to_dict())
        t_sum += float(r["dt_s"])
        if t_sum >= T_SEG_OUT:
            out = flush(buf, seg_idx_out)
            if out is not None:
                rows.append(out)
                seg_idx_out += 1
            buf = []
            t_sum = 0.0

    out = flush(buf, seg_idx_out)
    if out is not None:
        rows.append(out)

    return pd.DataFrame(rows)


# =========================
# SLOPE model (V_flat_ref from flat ONLY; train excludes flat)
# =========================
def compute_global_flat_ref_speed_out(seg_out: pd.DataFrame) -> float | None:
    """
    Връща V_flat_ref от OUT сегменти (30s):
      - валидни
      - |slope| <= 1%
      - използваме v_glide (след плъзгаемост)
    """
    if seg_out is None or seg_out.empty:
        return None

    df = seg_out.copy()
    m = (
        df["valid_basic"]
        & (~df["speed_spike"])
        & df["slope_pct"].notna()
        & (df["slope_pct"].abs() <= SLOPE_EPS_FLAT)
        & df["v_glide"].notna()
        & (df["v_glide"] > 0)
    )
    flat = df.loc[m, "v_glide"]
    if flat.empty:
        return None

    V0 = float(flat.mean())
    if not np.isfinite(V0) or V0 <= 0:
        return None
    return V0


def get_slope_training_data_out(seg_out: pd.DataFrame, V_flat_ref: float) -> pd.DataFrame:
    """
    Training за F(slope):
      - валидни
      - |slope| <= 15%
      - ИЗКЛЮЧВАМЕ flat |slope|<=1% (flat е само за V_flat_ref)
      - F = V_flat_ref / v_glide
    """
    if V_flat_ref is None or V_flat_ref <= 0:
        return pd.DataFrame(columns=["slope_pct", "F"])

    df = seg_out.copy()
    m = (
        df["valid_basic"]
        & (~df["speed_spike"])
        & df["slope_pct"].between(-SLOPE_TRAIN_MAX_ABS, SLOPE_TRAIN_MAX_ABS)
        & (df["slope_pct"].abs() > SLOPE_EPS_FLAT)
        & df["v_glide"].notna()
        & (df["v_glide"] > 0)
    )
    train = df.loc[m, ["slope_pct", "v_glide"]].copy()
    if train.empty:
        return pd.DataFrame(columns=["slope_pct", "F"])

    train["F"] = float(V_flat_ref) / train["v_glide"]
    return train[["slope_pct", "F"]]


def fit_slope_poly(train_df: pd.DataFrame):
    """
    Fit poly(F vs slope) + shift to enforce F(0)=1 (както в app-а).
    """
    if train_df is None or train_df.empty:
        return None

    x = train_df["slope_pct"].to_numpy(dtype=float)
    y = train_df["F"].to_numpy(dtype=float)
    if len(x) <= SLOPE_POLY_DEG:
        return None

    coeffs = np.polyfit(x, y, SLOPE_POLY_DEG)
    raw_poly = np.poly1d(coeffs)

    # shift to F(0)=1
    F0 = float(raw_poly(0.0))
    if not np.isfinite(F0):
        return None
    offset = F0 - 1.0
    coeffs_corr = raw_poly.coefficients.copy()
    coeffs_corr[-1] -= offset
    return np.poly1d(coeffs_corr)


def apply_slope_modulation_out(seg_out: pd.DataFrame, slope_poly, V_crit: float) -> pd.DataFrame:
    df = seg_out.copy()
    if slope_poly is None:
        df["v_flat_eq"] = df["v_glide"]
        return df

    slopes = df["slope_pct"].to_numpy(dtype=float)
    F_vals = slope_poly(slopes)

    # 1) clip
    F_vals = np.clip(F_vals, F_MIN_SLOPE, F_MAX_SLOPE)

    # 2) shape constraints (same as app)
    mask_mid = np.abs(slopes) <= SLOPE_EPS_FLAT
    F_vals[mask_mid] = 1.0
    mask_down = slopes < -SLOPE_EPS_FLAT
    F_vals[mask_down] = np.minimum(F_vals[mask_down], 1.0)
    mask_up = slopes > SLOPE_EPS_FLAT
    F_vals[mask_up] = np.maximum(F_vals[mask_up], 1.0)

    v_flat_eq = df["v_glide"].to_numpy(dtype=float) * F_vals

    # 3) strong downhills cap
    if V_crit is not None and V_crit > 0:
        idx = (df["valid_basic"]) & (df["slope_pct"] < -3.0)
        v_flat_eq[idx.to_numpy(dtype=bool)] = 0.7 * float(V_crit)

    df["v_flat_eq"] = v_flat_eq
    return df


# =========================
# CS helpers
# =========================
def clean_speed_for_cs(seg_out: pd.DataFrame, v_max_cs=50.0) -> np.ndarray:
    v = seg_out["v_flat_eq"].to_numpy(dtype=float)
    v = np.clip(v, 0.0, float(v_max_cs))

    is_spike = seg_out["speed_spike"].to_numpy(dtype=bool) if "speed_spike" in seg_out.columns else np.zeros_like(v, dtype=bool)
    if not is_spike.any():
        return v

    v_clean = v.copy()
    idx = np.arange(len(v_clean))
    good = ~is_spike
    if good.sum() <= 1:
        return v_clean

    v_clean[is_spike] = np.interp(idx[is_spike], idx[good], v_clean[good])
    return v_clean


# =========================
# Main entry: process one ski activity
# =========================
def process_ski_activity_30s(
    act_row: dict,
    streams: dict,
    V_crit: float,
    DAMP_GLIDE: float,
    CS: float,
    tau_min: float,
    k_par: float,
    q_par: float,
    gamma_cs: float,
    glide_poly_override=None,   # rolling from last-30 (7s history)
    slope_poly_override=None,   # rolling from last-30 (30s history)
):
    """
    Returns:
      seg30 (OUT=30s): includes v_kmh_raw, k_glide, v_glide, v_flat_eq, v_flat_eq_cs, cs fields
      seg7  (7s): for glide history storage (raw + hr_mean + valid flags)
    """
    activity_id = int(act_row["id"])

    points = build_points_from_streams(act_row, streams)
    if points.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 1) 7s segments
    seg7 = build_segments_fixed_seconds(points, activity_id, T_SEG_7, MIN_T_SEG_7, MIN_D_SEG_7)
    if seg7.empty:
        return pd.DataFrame(), pd.DataFrame()

    seg7 = apply_basic_filters_7s(seg7)

    # 2) glide poly: rolling override else local
    glide_poly = glide_poly_override
    if glide_poly is None:
        train_glide_local = get_glide_training_segments(seg7)
        glide_poly = fit_glide_poly(train_glide_local)

    # 3) K_glide for THIS activity from its 7s downhills
    k_glide = compute_k_glide_for_activity(seg7, glide_poly, float(DAMP_GLIDE))
    seg7_mod = apply_glide_modulation_7s(seg7, k_glide)

    # 4) aggregate 7s -> 30s OUT
    seg30 = aggregate_7s_to_out(seg7_mod, activity_id=activity_id)
    if seg30.empty:
        return seg30, seg7  # store raw 7s (without glide cols) is OK

    # 5) slope poly: rolling override else local (using NEW logic: flat only for V0; exclude flat in fit)
    slope_poly = slope_poly_override
    if slope_poly is None:
        V0 = compute_global_flat_ref_speed_out(seg30)
        if V0 is not None:
            slope_train = get_slope_training_data_out(seg30, V0)
            slope_poly = fit_slope_poly(slope_train)

    seg30 = apply_slope_modulation_out(seg30, slope_poly, float(V_crit) if V_crit is not None else None)

    # 6) CS modulation on 30s
    seg30 = seg30.sort_values("t_start").reset_index(drop=True)
    v_clean = clean_speed_for_cs(seg30, v_max_cs=50.0)
    dt_arr = seg30["dt_s"].to_numpy(dtype=float)

    out_cs = apply_cs_modulation(
        v=v_clean,
        dt=dt_arr,
        CS=float(CS),
        tau_min=float(tau_min),
        k_par=float(k_par),
        q_par=float(q_par),
        gamma=float(gamma_cs),
    )

    seg30["v_flat_eq_cs"] = out_cs["v_mod"]
    seg30["delta_v_plus_kmh"] = out_cs["delta_v_plus"]
    seg30["r_kmh"] = out_cs["r"]
    seg30["tau_s"] = out_cs["tau_s"]

    return seg30, seg7
