import numpy as np
import pandas as pd
from typing import Dict, Tuple


def compute_tiz(df: pd.DataFrame,
                mode: str,
                zones_cfg: Dict[str, Dict[str, Tuple[float, float]]]
                ) -> pd.Series:
    """
    Връща минути в Z1–Z5 според избрания режим:
      - mode="pct_cs" → използва cs_zone
      - mode="pct_cp" → използва cp_zone
      - mode="hr"     → използва hr_zone
    """
    if mode == "pct_cs":
        col = "cs_zone"
    elif mode == "pct_cp":
        col = "cp_zone"
    else:
        col = "hr_zone"

    valid_zones = ["Z1","Z2","Z3","Z4","Z5"]
    seconds_per_zone = {z: 0 for z in valid_zones}
    if col in df.columns:
        counts = df[col].value_counts()
        for z in valid_zones:
            seconds_per_zone[z] = int(counts.get(z, 0))
    tiz_minutes = {z: seconds_per_zone[z] / 60.0 for z in valid_zones}
    return pd.Series(tiz_minutes)


def compute_daily_stress(df: pd.DataFrame) -> float:
    """
    Проста метрика: за всяка секунда вземи най-добрия наличен % (pct_cs, pct_cp, pct_hr)
    и сумирай квадрата му. Резултатът е „стрес“ за активността.
    """
    p = df[["pct_cs","pct_cp","pct_hr"]].copy()
    best = p.max(axis=1, skipna=True)
    best = best.fillna(0.0)
    stress = float(np.square(best).sum())
    return stress


def aggregate_daily_stress(activities_meta: pd.DataFrame) -> pd.Series:
    """
    Очаква DataFrame с колони ["date","stress"] и агрегиран по дата.
    """
    s = activities_meta.groupby("date")["stress"].sum().sort_index()
    return s


def compute_acwr(stress_by_day: pd.Series,
                 acute_days: int = 7,
                 chronic_days: int = 28) -> pd.Series:
    """
    ACWR = (средно за последните 7 дни) / (средно за последните 28 дни).
    Връща серия по дата с ACWR.
    """
    acute = stress_by_day.rolling(acute_days, min_periods=1).mean()
    chronic = stress_by_day.rolling(chronic_days, min_periods=1).mean().replace(0, np.nan)
    acwr = acute / chronic
    return acwr.fillna(0.0)


def compute_hr_drift(df: pd.DataFrame) -> float:
    """
    Decoupling (HR drift) между 1-ва и 2-ра половина:
      ratio = mean(HR/velocity) в 2-ра половина / 1-ва половина - 1
    Нужни: hr и velocity_ms.
    """
    if not {"hr","velocity_ms"}.issubset(df.columns):
        return np.nan
    valid = df[["hr","velocity_ms"]].dropna()
    if len(valid) < 120:
        return np.nan
    n = len(valid)
    first = valid.iloc[:n//2]
    second = valid.iloc[n//2:]
    # избягваме деление на нула
    first_ratio = (first["hr"] / first["velocity_ms"].clip(lower=0.1)).mean()
    second_ratio = (second["hr"] / second["velocity_ms"].clip(lower=0.1)).mean()
    return float((second_ratio / first_ratio) - 1.0)
