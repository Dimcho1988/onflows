import pandas as pd
import numpy as np
from typing import Dict

def infer_modalities(df: pd.DataFrame) -> str:
    """
    Много груба евристика: ако има watts → колоездене; иначе, ако има velocity → бягане.
    """
    if "watts" in df.columns and df["watts"].notna().any():
        return "bike"
    if "velocity_ms" in df.columns and df["velocity_ms"].notna().any():
        return "run"
    return "unknown"


def suggest_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    """
    Елементарни предложения:
      - HRmax ~ 99-ти персентил на hr
      - CS_run_kmh ~ 95-ти персентил на velocity_kmh
      - CP_bike_w ~ 95-ти персентил на watts
    (Само като ориентир в MVP.)
    """
    out = {}
    if "hr" in df.columns and df["hr"].notna().any():
        out["HRmax"] = float(np.nanpercentile(df["hr"], 99))
    if "velocity_kmh" in df.columns and df["velocity_kmh"].notna().any():
        out["CS_run_kmh"] = float(np.nanpercentile(df["velocity_kmh"], 95))
    if "watts" in df.columns and df["watts"].notna().any():
        out["CP_bike_w"] = float(np.nanpercentile(df["watts"], 95))
    return out
