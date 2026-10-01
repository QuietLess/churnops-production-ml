from __future__ import annotations

from fastapi import HTTPException, Request

from app.db import PredictionRepository
from app.metrics import ApiMetrics
from app.model_service import ModelService
from app.settings import Settings


def get_model_service(request: Request) -> ModelService:
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return service


def get_repository(request: Request) -> PredictionRepository | None:
    return getattr(request.app.state, "repository", None)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_metrics(request: Request) -> ApiMetrics:
    return request.app.state.metrics
