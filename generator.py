import math
import pandas as pd
from typing import Dict, Any, List


def _minutes_by_zone(total_minutes: int, dist: Dict[str, float]) -> Dict[str, int]:
    raw = {z: total_minutes * p for z, p in dist.items()}
    # закръгляне
    zmins = {z: int(round(v)) for z, v in raw.items()}
    # нормализирай до тотал
    diff = total_minutes - sum(zmins.values())
    if diff != 0:
        # коригирай Z1
        zmins["Z1"] = zmins.get("Z1", 0) + diff
    return zmins


def generate_plan(athlete_state: Dict[str, Any],
                  indices: Dict[str, Any],
                  base_calendar: List[str],
                  cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Връща седмичен план: колони [Ден, Зона, Мин, Таргет].
    - Адаптация: ако ACWR > cfg["acwr_high"] → -volume% и -1 зона за Z4/Z5.
    """
    hrmax = athlete_state["HRmax"]
    cs = athlete_state["CS_run_kmh"]
    cp = athlete_state["CP_bike_w"]
    acwr_latest = indices.get("acwr_latest", 1.0)

    weekly_target = cfg["generator"]["weekly_minutes_target"]
    dist = cfg["generator"]["distribution_pct"].copy()

    if acwr_latest > cfg["generator"]["acwr_high"]:
        weekly_target = int(round(weekly_target * (1 - cfg["generator"]["acwr_reduce_volume_pct"])))
        # понижи интензитета: свали минути от Z4/Z5 към Z1
        move = 0
        for z in ["Z5","Z4"]:
            cut = int(round(weekly_target * 0.04))  # по 4% от новия обем
            dist[z] = max(0.0, dist[z] - cut / weekly_target)
            move += cut / weekly_target
        dist["Z1"] = min(1.0, dist["Z1"] + move)

    zmins = _minutes_by_zone(weekly_target, dist)

    # дневно разпределение (прост пример)
    days = base_calendar  # напр. ["Пон","Вто","Сря","Чет","Пет","Съб","Нед"]
    day_share = [0.18, 0.18, 0.14, 0.18, 0.14, 0.18, 0.00]  # Неделя почивка (пример)
    day_minutes = [int(round(weekly_target * s)) for s in day_share]

    rows = []
    # опростено: всеки ден миксира по малко от зоните, като тежките зони повече в Вто/Чет/Съб
    heavy_days = {1, 3, 5}  # 0=Пон
    z_order = ["Z1","Z2","Z3","Z4","Z5"]

    # конструирай таргет низове: бягане (%CS→km/h + HR), колело (%CP→W + HR)
    def target_for_zone(z: str) -> str:
        # примерни прагове
        zone_pct_hr = {
            "Z1": (0.60, 0.75),
            "Z2": (0.76, 0.80),
            "Z3": (0.81, 0.88),
            "Z4": (0.89, 0.95),
            "Z5": (0.96, 1.00),
        }
        lo_hr, hi_hr = zone_pct_hr.get(z, (0.6, 0.75))
        hr_range = f"HR {int(hrmax*lo_hr)}–{int(hrmax*hi_hr)} bpm"

        zone_pct_cs = {
            "Z1": (0.60, 0.75),
            "Z2": (0.76, 0.80),
            "Z3": (0.81, 0.88),
            "Z4": (0.89, 0.95),
            "Z5": (0.96, 1.10),
        }
        lo_cs, hi_cs = zone_pct_cs.get(z, (0.6, 0.75))
        run_speed = f"{cs*lo_cs:.1f}–{cs*hi_cs:.1f} km/h"

        zone_pct_cp = {
            "Z1": (0.55, 0.75),
            "Z2": (0.76, 0.85),
            "Z3": (0.86, 0.95),
            "Z4": (0.96, 1.05),
            "Z5": (1.06, 1.20),
        }
        lo_cp, hi_cp = zone_pct_cp.get(z, (0.55, 0.75))
        bike_power = f"{int(cp*lo_cp)}–{int(cp*hi_cp)} W"

        return f"Бягане: {run_speed}; Колоездене: {bike_power}; {hr_range}"

    # разсипи минути по дни
    remaining = zmins.copy()
    for d_idx, day in enumerate(days):
        m_today = day_minutes[d_idx]
        if m_today == 0:
            rows.append([day, "Почивка", 0, "—"])
            continue

        # дял на интензитета в тежки дни
        today_z = remaining.copy()
        alloc = {z: 0 for z in z_order}

        # прости правила за „тежест“
        weights = {"Z1": 1, "Z2": 1, "Z3": 1 if d_idx not in heavy_days else 1.3,
                   "Z4": 0.6 if d_idx not in heavy_days else 1.2,
                   "Z5": 0.3 if d_idx not in heavy_days else 0.8}

        total_weight = sum(weights[z] * (today_z[z] + 1) for z in z_order)
        for z in z_order:
            share = weights[z] * (today_z[z] + 1) / total_weight
            alloc[z] = int(round(m_today * share))

        # нормализирай до m_today
        diff = m_today - sum(alloc.values())
        if diff != 0:
            alloc["Z1"] += diff

        for z in z_order:
            take = min(alloc[z], remaining[z])
            if take <= 0:
                continue
            remaining[z] -= take
            rows.append([day, z, take, target_for_zone(z)])

    plan = pd.DataFrame(rows, columns=["Ден","Зона","Мин","Таргет"])
    # слей редове за един ден/зона
    plan = plan.groupby(["Ден","Зона","Таргет"], as_index=False)["Мин"].sum()
    # подредба на дните
    order = {name: i for i, name in enumerate(days)}
    plan["__ord"] = plan["Ден"].map(order)
    plan = plan.sort_values(["__ord","Зона"]).drop(columns="__ord").reset_index(drop=True)
    return plan
