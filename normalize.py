import numpy as np
import pandas as pd
from typing import Tuple, Dict


def _zone_label(value: float, zone_bounds: Dict[str, Tuple[float, float]]) -> str:
    if np.isnan(value):
        return "NA"
    for z, (lo, hi) in zone_bounds.items():
        if lo <= value <= hi:
            return z
    return "NA"


def add_percent_columns(df: pd.DataFrame,
                        cs_run_kmh: float,
                        cp_bike_w: float,
                        hrmax: float,
                        zones_cfg: Dict[str, Dict[str, Tuple[float, float]]]
                        ) -> pd.DataFrame:
    """
    Добавя:
      - pct_cs: velocity_kmh / CS
      - pct_cp: watts / CP
      - pct_hr: hr / HRmax
      - hr_zone / cs_zone / cp_zone (Z1–Z5)
      - confidence флаг за налични канали (в df.attrs["channels_present"])
    """
    out = df.copy()

    # проценти
    out["pct_cs"] = np.where(
        ("velocity_kmh" in out.columns) & (cs_run_kmh > 0),
        out.get("velocity_kmh", np.nan) / cs_run_kmh,
        np.nan
    )
    out["pct_cp"] = np.where(
        ("watts" in out.columns) & (cp_bike_w > 0),
        out.get("watts", np.nan) / cp_bike_w,
        np.nan
    )
    out["pct_hr"] = np.where(
        ("hr" in out.columns) & (hrmax > 0),
        out.get("hr", np.nan) / hrmax,
        np.nan
    )

    # зони
    hr_bounds = zones_cfg.get("hr", {})
    cs_bounds = zones_cfg.get("pct_cs", {})
    cp_bounds = zones_cfg.get("pct_cp", {})

    out["hr_zone"] = [ _zone_label(v, hr_bounds) for v in out["pct_hr"].values ]
    out["cs_zone"] = [ _zone_label(v, cs_bounds) for v in out["pct_cs"].values ]
    out["cp_zone"] = [ _zone_label(v, cp_bounds) for v in out["pct_cp"].values ]

    # confidence
    ch = out.attrs.get("channels_present", {})
    out.attrs["confidence"] = {
        "hr": bool(ch.get("hr", False)),
        "cs": bool(ch.get("velocity", False)),
        "cp": bool(ch.get("watts", False))
    }
    return out
