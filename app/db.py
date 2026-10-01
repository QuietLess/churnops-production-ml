"""Prediction log persistence (PostgreSQL in docker/cloud, SQLite locally/tests)."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine, func, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

logger = logging.getLogger(__name__)

JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    prediction_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    customer_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prediction: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16))
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(32), index=True)
    request_features: Mapped[dict] = mapped_column(JSONType)
    source: Mapped[str] = mapped_column(String(32), default="api")
    # Delayed ground truth (did the customer actually churn?), set later via POST /feedback.
    # NULL until a label arrives; performance monitoring only uses labelled rows.
    actual_outcome: Mapped[int | None] = mapped_column(Integer, nullable=True)


def new_prediction_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC)


class PredictionRepository:
    def __init__(self, database_url: str):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        if database_url.startswith("sqlite:///"):
            from pathlib import Path

            Path(database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
        self._session = sessionmaker(self.engine, expire_on_commit=False)

    def init_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            return True
        except Exception:  # noqa: BLE001
            return False

    def log_many(self, rows: list[dict]) -> None:
        with Session(self.engine) as session, session.begin():
            session.add_all(PredictionLog(**row) for row in rows)

    def count(self) -> int:
        with self._session() as session:
            return int(session.scalar(select(func.count()).select_from(PredictionLog)) or 0)

    def recent(self, limit: int = 1000) -> list[PredictionLog]:
        with self._session() as session:
            stmt = select(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(limit)
            return list(session.scalars(stmt))

    def set_outcomes(self, outcomes: Mapping[str, int]) -> list[str]:
        """Attach ground truth to logged predictions. Returns the prediction_ids not found."""
        missing = []
        with Session(self.engine) as session, session.begin():
            for prediction_id, outcome in outcomes.items():
                result = session.execute(
                    update(PredictionLog)
                    .where(PredictionLog.prediction_id == prediction_id)
                    .values(actual_outcome=outcome)
                )
                if result.rowcount == 0:  # type: ignore[attr-defined]  # CursorResult at runtime
                    missing.append(prediction_id)
        return missing

    def labelled(self, limit: int = 1000) -> list[PredictionLog]:
        """Most recent predictions that have a ground-truth outcome."""
        with self._session() as session:
            stmt = (
                select(PredictionLog)
                .where(PredictionLog.actual_outcome.is_not(None))
                .order_by(PredictionLog.timestamp.desc())
                .limit(limit)
            )
            return list(session.scalars(stmt))
