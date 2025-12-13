# slope_last30.py
import numpy as np
import pandas as pd


# -----------------------------
# helpers
# -----------------------------
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _shift_poly_to_F0_eq_1(poly: np.poly1d) -> np.poly1d:
    F0 = float(poly(0.0))
    offset = F0 - 1.0
    coeffs = poly.coefficients.copy()
    coeffs[-1] -= offset
    return np.poly1d(coeffs)


def fit_slope_poly_from_last30_activities(
    supabase,
    user_id: int,
    sport_group: str,
    segment_seconds: int,
    # важно: ако имаш различни model_key (unknown vs run_slope_v2),
    # искаме да вземем ВСИЧКИ последни 30 активности независимо от model_key
    # -> model_key_filter може да е None (препоръчително)
    model_key_filter: str | None = None,
    n_activities: int = 30,
    deg: int = 2,
    slope_range=(-15.0, 15.0),
    flat_range=(-1.0, 1.0),
    v_min=3.0,
    v_max=30.0,
    min_total_rows: int = 80,
    min_flat_per_activity: int = 2,
) -> np.poly1d | None:
    """
    LAST-30 ACTIVITIES fit (не last-30 сегмента):

    За всяка активност i:
      - V0_i = mean(v_raw) за flat сегментите |slope|<=1% (валидни, без спайк)
      - F_raw = V0_i / v_raw за всички валидни сегменти в slope_range
    После обединяваме всички активности и fit-ваме poly(F_raw ~ slope).

    Връща np.poly1d с F(0)=1 или None.
    """

    # 1) последни N активности (само ids)
    res = (
        supabase.table("activities")
        .select("id,start_date")
        .eq("user_id", int(user_id))
        .eq("sport_group", sport_group)
        .order("start_date", desc=True)
        .limit(int(n_activities))
        .execute()
    )
    acts = res.data or []
    ids = [a.get("id") for a in acts if a.get("id") is not None]
    if len(ids) < 5:
        return None

    # 2) сегменти за тези активности
    q = (
        supabase.table("activity_segments")
        .select("activity_id,slope_pct,v_kmh_raw,valid_basic,speed_spike,segment_seconds,model_key")
        .in_("activity_id", ids)
        .eq("sport_group", sport_group)
        .eq("segment_seconds", int(segment_seconds))
    )

    # по избор: филтър по model_key (по подразбиране НЕ)
    if model_key_filter and model_key_filter != "unknown":
        q = q.eq("model_key", model_key_filter)

    res2 = q.execute()
    df = pd.DataFrame(res2.data or [])
    if df.empty:
        return None

    # normalize types
    df["slope_pct"] = pd.to_numeric(df["slope_pct"], errors="coerce")
    df["v_kmh_raw"] = pd.to_numeric(df["v_kmh_raw"], errors="coerce")
    if "valid_basic" in df.columns:
        df["valid_basic"] = df["valid_basic"].fillna(True).astype(bool)
    else:
        df["valid_basic"] = True
    if "speed_spike" in df.columns:
        df["speed_spike"] = df["speed_spike"].fillna(False).astype(bool)
    else:
        df["speed_spike"] = False

    # базов валиден маск
    base = (
        df["valid_basic"]
        & (~df["speed_spike"])
        & df["slope_pct"].between(float(slope_range[0]), float(slope_range[1]))
        & (df["v_kmh_raw"] > float(v_min))
        & (df["v_kmh_raw"] < float(v_max))
    )
    dfv = df.loc[base, ["activity_id", "slope_pct", "v_kmh_raw"]].copy()
    if dfv.empty:
        return None

    # 3) V0 per activity от flat
    flat = dfv[dfv["slope_pct"].between(float(flat_range[0]), float(flat_range[1]))].copy()
    if flat.empty:
        return None

    v0_map = (
        flat.groupby("activity_id")["v_kmh_raw"]
        .mean()
        .rename("V0")
        .reset_index()
    )

    # require min flat segments per activity (optional)
    flat_counts = flat.groupby("activity_id").size().rename("n_flat").reset_index()
    v0_map = v0_map.merge(flat_counts, on="activity_id", how="left")
    v0_map = v0_map[v0_map["n_flat"] >= int(min_flat_per_activity)]
    if v0_map.empty:
        return None

    dfv = dfv.merge(v0_map[["activity_id", "V0"]], on="activity_id", how="inner")
    if dfv.empty:
        return None

    # 4) F_raw = V0_i / v_raw
    dfv["F_raw"] = dfv["V0"] / dfv["v_kmh_raw"]
    dfv = dfv.replace([np.inf, -np.inf], np.nan).dropna(subset=["slope_pct", "F_raw"])
    if len(dfv) < int(min_total_rows):
        return None

    x = dfv["slope_pct"].to_numpy(dtype=float)
    y = dfv["F_raw"].to_numpy(dtype=float)
    if len(x) <= deg + 5:
        return None

    # 5) polyfit + shift F(0)=1
    coeffs = np.polyfit(x, y, int(deg))
    poly = np.poly1d(coeffs)
    poly = _shift_poly_to_F0_eq_1(poly)
    return poly
