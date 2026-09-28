from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_model_service
from app.model_service import ModelService
from app.schemas import ModelInfoResponse
from src.config import RISK_HIGH_LOWER, RISK_LOW_UPPER

router = APIRouter(tags=["model"])


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(service: ModelService = Depends(get_model_service)) -> ModelInfoResponse:
    md = service.metadata
    return ModelInfoResponse(
        model_name=service.model_name,
        model_alias=service.model_alias,
        model_version=service.model_version,
        model_uri=service.model_uri,
        threshold=service.threshold,
        threshold_strategy=md.get("threshold_strategy"),
        model_family=md.get("model_family"),
        feature_version=md.get("feature_version"),
        risk_bands=md.get("risk_bands", {"low_upper": RISK_LOW_UPPER, "high_lower": RISK_HIGH_LOWER}),
        loaded_at=service.loaded_at,
    )
