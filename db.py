from __future__ import annotations
import os
from typing import Optional, Iterable
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, text, BigInteger, Integer, Float, String, Boolean,
    DateTime, Index, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

# --- .env локално (безопасно е и NO-OP в Streamlit Cloud) ---
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def get_db_url() -> str:
    url = os.getenv("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError(
            "SUPABASE_DB_URL липсва. Пример:\n"
            "postgresql+psycopg2://postgres:PASS@aws-1-us-east-2.pooler.supabase.com:6543/postgres?sslmode=require"
        )
    if url.startswith("postgresql+psycopg://"):
        url = url.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)
    if "sslmode" not in url:
        url += "&sslmode=require" if "?" in url else "?sslmode=require"
    return url


_engine = create_engine(
    get_db_url(),
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


# ---------------- ORM модели ----------------

class Base(DeclarativeBase):
    pass


class Athlete(Base):
    __tablename__ = "athletes"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    strava_athlete_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    firstname: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    lastname: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("timezone('utc', now())"), nullable=False
    )


class StravaToken(Base):
    __tablename__ = "strava_tokens"
    athlete_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)  # вътр. athlete.id или директно strava_athlete_id
    access_token: Mapped[str] = mapped_column(String, nullable=False)
    refresh_token: Mapped[str] = mapped_column(String, nullable=False)
    expires_at: Mapped[int] = mapped_column(Integer, nullable=False)
    scope: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("timezone('utc', now())"), nullable=False
    )


class Activity(Base):
    __tablename__ = "activities"

    # !!! съобразено със съществуващата ти таблица: PK колона се казва "id"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)  # Strava activity id

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
        DateTime(timezone=True), server_default=text("timezone('utc', now())"), nullable=False
    )

    __table_args__ = (
        Index("ix_activities_start_date", "start_date"),
        Index("ix_activities_athlete_id", "athlete_id"),
    )


class ActivityStream(Base):
    __tablename__ = "activity_streams"

    # съобразено със стария ти дизайн: (activity_id, second) е уникален ключ
    activity_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("activities.id", ondelete="CASCADE"), primary_key=True
    )
    second: Mapped[int] = mapped_column(Integer, primary_key=True)  # 0..N при 1 Hz

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
        UniqueConstraint("activity_id", "second", name="ux_streams_activity_second"),
    )


# ------------- Init -------------

def init_db(create_missing: bool = True) -> None:
    if create_missing:
        Base.metadata.create_all(_engine)


# ---------- Helpers ----------

def _to_aware(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        if isinstance(dt_str, datetime):
            return dt_str if dt_str.tzinfo else dt_str.replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _normalize_activity_row(a: dict, fallback_athlete_id: Optional[int]) -> dict:
    return {
        "id": a.get("id"),
        "athlete_id": ((a.get("athlete") or {}).get("id")) or fallback_athlete_id,
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


# ---------- Публични функции ----------

def upsert_athlete_and_tokens(strava_athlete_id: int, firstname: str | None, lastname: str | None,
                              access_token: str, refresh_token: str, expires_at: int, scope: str | None) -> None:
    """
    1) Upsert в athletes по strava_athlete_id
    2) Upsert в strava_tokens по athlete_id (= strava_athlete_id или вътр. id – тук ползваме директно strava_athlete_id)
    """
    with SessionLocal() as s:
        # athletes
        a_stmt = pg_insert(Athlete.__table__).values({
            "strava_athlete_id": strava_athlete_id,
            "firstname": firstname,
            "lastname": lastname,
        }).on_conflict_do_update(
            index_elements=[Athlete.__table__.c.strava_athlete_id],
            set_={"firstname": firstname, "lastname": lastname}
        )
        s.execute(a_stmt)

        # strava_tokens (ползваме ключ athlete_id = STRAVA athlete id, за простота)
        t_stmt = pg_insert(StravaToken.__table__).values({
            "athlete_id": strava_athlete_id,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
            "scope": scope,
        }).on_conflict_do_update(
            index_elements=[StravaToken.__table__.c.athlete_id],
            set_={
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_at": expires_at,
                "scope": scope,
                "updated_at": text("timezone('utc', now())")
            }
        )
        s.execute(t_stmt)
        s.commit()


def upsert_activities(activities: Iterable[dict], *, fallback_athlete_id: Optional[int]) -> int:
    rows = []
    for a in activities:
        row = _normalize_activity_row(a, fallback_athlete_id)
        if row.get("id") is None:
            continue
        rows.append(row)

    if not rows:
        return 0

    stmt = pg_insert(Activity.__table__).values(rows)
    update_cols = {
        c.name: stmt.excluded[c.name]
        for c in Activity.__table__.c
        if c.name not in ("id", "created_at")
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=[Activity.__table__.c.id],
        set_=update_cols,
    )

    with SessionLocal() as s:
        res = s.execute(stmt)
        s.commit()
        return res.rowcount or 0


def upsert_activity_streams(activity_id: int, streams: dict, *, chunk_size: int = 10_000) -> int:
    if not activity_id or not streams:
        return 0

    t = streams.get("time") or []
    latlng = streams.get("latlng") or []
    altitude = streams.get("altitude") or []
    distance = streams.get("distance") or []
    speed = streams.get("velocity_smooth") or streams.get("speed") or []
    hr = streams.get("heartrate") or streams.get("heart_rate") or []
    cad = streams.get("cadence") or []
    pwr = streams.get("watts") or streams.get("power") or []
    tmp = streams.get("temperature") or streams.get("temp") or []

    max_len = max(len(t), len(latlng), len(altitude), len(distance),
                  len(speed), len(hr), len(cad), len(pwr), len(tmp))
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
            "second": i,                         # секунден индекс
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
    with SessionLocal() as s:
        for j in range(0, len(rows), chunk_size):
            chunk = rows[j:j + chunk_size]
            stmt = pg_insert(ActivityStream.__table__).values(chunk)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=[ActivityStream.__table__.c.activity_id,
                                ActivityStream.__table__.c.second]
            )
            res = s.execute(stmt)
            total += res.rowcount or 0
        s.commit()
    return total
