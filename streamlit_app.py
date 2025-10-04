# db.py
from __future__ import annotations
import os
from typing import Optional, Iterable
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, text, BigInteger, Integer, Float, String, Boolean,
    DateTime, UniqueConstraint, Index, ForeignKey
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


# ---------- Engine / Session ----------

def get_db_url() -> str:
    url = os.getenv("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError(
            "SUPABASE_DB_URL липсва. Добави го в .env, напр.\n"
            "SUPABASE_DB_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@.../postgres?sslmode=require"
        )
    return url

_engine = create_engine(
    get_db_url(),
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
)

SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


# ---------- ORM модели ----------

class Base(DeclarativeBase):
    pass


class Activity(Base):
    __tablename__ = "activities"

    # В onFlows приемаме, че activity_id идва от Strava (bigint).
    activity_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    athlete_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    type: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    start_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    elapsed_time: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    moving_time: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    distance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_elevation_gain: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    average_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    average_heartrate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_heartrate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    average_cadence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    average_watts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    device_watts: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("timezone('utc', now())"),
        nullable=False,
    )

    __table_args__ = (
        # Ако вече имаш PK/индекси — това няма да ги дублира.
        Index("ix_activities_start_date", "start_date"),
        Index("ix_activities_athlete_id", "athlete_id"),
    )


class ActivityStream(Base):
    __tablename__ = "activity_streams"

    # Композитен уникален ключ: (activity_id, idx)
    activity_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("activities.activity_id", ondelete="CASCADE"),
        primary_key=True,
    )
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)  # 0..N при 1 Hz

    time_s: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    altitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    heartrate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cadence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    power: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("ix_streams_activity_id", "activity_id"),
        Index("ix_streams_time", "activity_id", "time_s"),
    )


# ---------- Init (по избор, ако искаш автосъздаване) ----------

def init_db(create_missing: bool = True) -> None:
    """Създава липсващи таблици според моделите (няма да трие/променя съществуващи)."""
    if create_missing:
        Base.metadata.create_all(_engine)


# ---------- Помощници за нормализация ----------

def _to_aware(dt_str: Optional[str]) -> Optional[datetime]:
    """ISO8601 -> timezone-aware UTC datetime (ако има вход)."""
    if not dt_str:
        return None
    # Примери от Strava: "2025-10-01T06:48:33Z"
    try:
        if isinstance(dt_str, datetime):
            return dt_str if dt_str.tzinfo else dt_str.replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _normalize_activity_row(a: dict) -> dict:
    """Очаква Strava-подобен dict и връща готов ред за таблица `activities`."""
    return {
        "activity_id": a.get("id"),
        "athlete_id": (a.get("athlete") or {}).get("id"),
        "name": a.get("name"),
        "type": a.get("sport_type") or a.get("type"),
        "start_date": _to_aware(a.get("start_date") or a.get("start_date_local")),
        "elapsed_time": a.get("elapsed_time"),
        "moving_time": a.get("moving_time"),
        "distance": a.get("distance"),
        "total_elevation_gain": a.get("total_elevation_gain"),
        "average_speed": a.get("average_speed"),
        "max_speed": a.get("max_speed"),
        "average_heartrate": a.get("average_heartrate"),
        "max_heartrate": a.get("max_heartrate"),
        "average_cadence": a.get("average_cadence"),
        "average_watts": a.get("average_watts"),
        "device_watts": a.get("device_watts"),
    }


# ---------- Публични функции (това ще ползваш) ----------

def upsert_activities(activities: Iterable[dict]) -> int:
    """
    Upsert на списък с активности в `activities`.
    Връща броя редове, които са вкарани/обновени.
    """
    rows = []
    for a in activities:
        row = _normalize_activity_row(a)
        if row.get("activity_id") is None:
            continue
        rows.append(row)

    if not rows:
        return 0

    stmt = pg_insert(Activity.__table__).values(rows)
    # Ключът е PK activity_id – при конфликт обновяваме последните данни (name, метрики и пр.)
    update_cols = {c.name: stmt.excluded[c.name]
                   for c in Activity.__table__.c
                   if c.name not in ("activity_id", "created_at")}
    stmt = stmt.on_conflict_do_update(
        index_elements=[Activity.__table__.c.activity_id],
        set_=update_cols,
    )

    with SessionLocal() as s:
        res = s.execute(stmt)
        s.commit()
        return res.rowcount or 0


def upsert_activity_streams(
    activity_id: int,
    streams: dict,
    *,
    hz: int = 1,
    chunk_size: int = 10_000,
) -> int:
    """
    Upsert на 1 Hz (или друго hz) данни в `activity_streams` за една активност.

    `streams` е dict с ключове по подобие на Strava:
      - "time": [0,1,2,...] или секунди от старта
      - "latlng": [[lat, lon], ...]     (по избор)
      - "altitude", "distance", "velocity_smooth"/"speed",
        "heart_rate", "cadence", "watts"/"power", "temp"/"temperature"  -> списъци

    Връща общия брой опитани за insert/update редове (не точно „вкарани нови“).
    """
    if not activity_id:
        return 0

    # Взимаме масивите, ако липсват – None
    t = streams.get("time") or []
    latlng = streams.get("latlng") or []
    altitude = streams.get("altitude") or []
    distance = streams.get("distance") or []
    speed = streams.get("velocity_smooth") or streams.get("speed") or []
    hr = streams.get("heartrate") or streams.get("heart_rate") or []
    cad = streams.get("cadence") or []
    pwr = streams.get("watts") or streams.get("power") or []
    tmp = streams.get("temperature") or streams.get("temp") or []

    # Дължина = max по наличните потоци
    max_len = max(
        len(t), len(latlng), len(altitude), len(distance),
        len(speed), len(hr), len(cad), len(pwr), len(tmp)
    )
    if max_len == 0:
        return 0

    def get_or_none(arr, i, sub=None):
        try:
            v = arr[i]
            if sub is not None:
                return v[sub] if v is not None else None
            return v
        except Exception:
            return None

    rows = []
    for i in range(max_len):
        rows.append({
            "activity_id": activity_id,
            "idx": i,                     # секунден индекс при 1 Hz
            "time_s": get_or_none(t, i),

            "lat": get_or_none(latlng, i, 0),
            "lon": get_or_none(latlng, i, 1),

            "altitude": get_or_none(altitude, i),
            "distance": get_or_none(distance, i),
            "speed": get_or_none(speed, i),

            "heartrate": get_or_none(hr, i),
            "cadence": get_or_none(cad, i),
            "power": get_or_none(pwr, i),
            "temperature": get_or_none(tmp, i),
        })

    total = 0
    # Upsert на парчета за големи активности
    with SessionLocal() as s:
        for j in range(0, len(rows), chunk_size):
            chunk = rows[j:j + chunk_size]
            stmt = pg_insert(ActivityStream.__table__).values(chunk)
            # При конфликт на (activity_id, idx) -> не правим нищо (данните не би трябвало да се променят)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=[ActivityStream.__table__.c.activity_id,
                                ActivityStream.__table__.c.idx]
            )
            res = s.execute(stmt)
            total += res.rowcount or 0
        s.commit()
    return total
