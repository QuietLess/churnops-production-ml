"""Runtime settings from environment variables (never hard-code secrets)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.config import CHAMPION_EXPORT_DIR, MODEL_ALIAS, REGISTERED_MODEL_NAME


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # "models:/churnops-model@champion" (registry) or a local exported MLflow model dir.
    model_uri: str = field(
        default_factory=lambda: os.getenv("MODEL_URI", f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}")
    )
    mlflow_tracking_uri: str | None = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI"))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///data/churnops.db"))
    log_predictions: bool = field(default_factory=lambda: _bool("LOG_PREDICTIONS", True))
    max_batch_size: int = field(default_factory=lambda: int(os.getenv("MAX_BATCH_SIZE", "1000")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @staticmethod
    def exported_model_default() -> str:
        return str(CHAMPION_EXPORT_DIR)
