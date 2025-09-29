import time
import math
import requests
import pandas as pd
from typing import Dict, Any, List, Optional

STRAVA_BASE = "https://www.strava.com/api/v3"


class StravaClient:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.access_token}"})

    def get_athlete_activities(self, per_page: int = 10) -> List[Dict[str, Any]]:
        url = f"{STRAVA_BASE}/athlete/activities"
        resp = self.session.get(url, params={"per_page": per_page})
        resp.raise_for_status()
        return resp.json()

    def get_activity_streams(self, activity_id: int,
                             keys: Optional[List[str]] = None,
                             key_by_type: bool = True) -> Dict[str, Any]:
        if keys is None:
            keys = ["time","heartrate","velocity_smooth","watts","altitude","distance","cadence"]
        url = f"{STRAVA_BASE}/activities/{activity_id}/streams"
        resp = self.session.get(url, params={"keys": ",".join(keys), "key_by_type": str(key_by_type).lower()})
        resp.raise_for_status()
        return resp.json()


def _safe_get(streams: Dict[str, Any], key: str) -> Optional[List[Any]]:
    obj = streams.get(key)
    if isinstance(obj, dict):
        return obj.get("data")
    return None


def build_timeseries_1hz(streams: Dict[str, Any]) -> pd.DataFrame:
    """
    Приема JSON от /activities/{id}/streams?key_by_type=true
    Връща 1 Hz DataFrame (секунда по секунда) с наличните канали.
    """
    time_s = _safe_get(streams, "time")
    if not time_s:
        # Някои активности нямат time stream; опит за fallback по дължина на друг ключ
        for fallback in ["distance","velocity_smooth","watts","heartrate","altitude","cadence"]:
            arr = _safe_get(streams, fallback)
            if arr:
                time_s = list(range(len(arr)))
                break
    if not time_s:
        # крайна защита
        return pd.DataFrame()

    max_t = int(time_s[-1]) if len(time_s) else 0
    idx = pd.Index(range(max_t + 1), name="second")
    df = pd.DataFrame(index=idx)

    # map ключове -> колони
    mapping = {
        "heartrate": "hr",
        "velocity_smooth": "velocity_ms",
        "watts": "watts",
        "altitude": "altitude_m",
        "distance": "distance_m",
        "cadence": "cadence_spm"
    }

    for key, col in mapping.items():
        arr = _safe_get(streams, key)
        if arr is None:
            df[col] = math.nan
            continue
        # Стриймовете са по sample; ще ги „запишем“ по известните секунди
        series = pd.Series(arr, index=pd.Index(time_s[:len(arr)], name="second"), name=col)
        # Reindex към 1 Hz и запълни с forward-fill, за да няма зъбци при по-рядко семплиране
        df[col] = series.reindex(df.index).ffill()

    # производни
    if "velocity_ms" in df.columns:
        df["velocity_kmh"] = df["velocity_ms"] * 3.6

    # дата/час ако можем (Strava не дава тук absolute timestamps в streams)
    # ще оставим относителен second; датата се взема от activity meta, ако я подадеш отделно

    # маркирай наличност/увереност (за Normalize)
    df.attrs["channels_present"] = {
        "hr": df["hr"].notna().any() if "hr" in df.columns else False,
        "velocity": df["velocity_ms"].notna().any() if "velocity_ms" in df.columns else False,
        "watts": df["watts"].notna().any() if "watts" in df.columns else False,
    }

    return df
