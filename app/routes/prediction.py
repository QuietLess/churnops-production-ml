"""Prediction endpoints.

Logging policy: if writing the prediction log fails, the prediction is still returned
(with `logged=false`) and the failure is logged as an error. Serving availability is
prioritised over log completeness; monitoring tolerates small gaps.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import ValidationError

from app.db import PredictionRepository, new_prediction_id, utcnow
from app.dependencies import get_metrics, get_model_service, get_repository, get_settings
from app.metrics import ApiMetrics
from app.model_service import ModelService
from app.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    ExplanationResponse,
    PredictionResponse,
)
from app.settings import Settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["prediction"])


def _run(
    customers: list[CustomerFeatures],
    service: ModelService,
    repo: PredictionRepository | None,
    metrics: ApiMetrics,
    source: str,
) -> list[PredictionResponse]:
    features = [c.features() for c in customers]
    try:
        scored = service.score(features)
    except Exception as exc:
        logger.exception("Inference failed", extra={"event": "inference_error"})
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    now = utcnow()
    results = [
        {"prediction_id": new_prediction_id(), "customerID": c.customerID, **s}
        for c, s in zip(customers, scored, strict=True)
    ]
    metrics.observe_predictions(results, service.model_version, source)
    logged = False
    if repo is not None:
        rows = [
            {
                "prediction_id": r["prediction_id"],
                "timestamp": now,
                "customer_id": r["customerID"],
                "prediction": r["prediction"],
                "probability": r["churn_probability"],
                "risk_level": r["risk_level"],
                "threshold": service.threshold,
                "model_name": service.model_name,
                "model_version": service.model_version,
                "request_features": f,
                "source": source,
            }
            for r, f in zip(results, features, strict=True)
        ]
        try:
            repo.log_many(rows)
            logged = True
        except Exception:
            metrics.log_failures.inc()
            logger.exception("Prediction log write failed", extra={"event": "log_write_failed"})
    return [
        PredictionResponse(
            **r,
            threshold=service.threshold,
            model_name=service.model_name,
            model_alias=service.model_alias,
            model_version=service.model_version,
            logged=logged,
        )
        for r in results
    ]


def _check_size(n: int, settings: Settings) -> None:
    if n > settings.max_batch_size:
        raise HTTPException(413, f"Batch of {n} exceeds MAX_BATCH_SIZE={settings.max_batch_size}")


@router.post("/predict", response_model=PredictionResponse)
def predict(
    customer: CustomerFeatures,
    service: ModelService = Depends(get_model_service),
    repo: PredictionRepository | None = Depends(get_repository),
    metrics: ApiMetrics = Depends(get_metrics),
) -> PredictionResponse:
    return _run([customer], service, repo, metrics, source="api")[0]


@router.post("/batch-predict", response_model=BatchResponse)
def batch_predict(
    batch: BatchRequest,
    service: ModelService = Depends(get_model_service),
    repo: PredictionRepository | None = Depends(get_repository),
    settings: Settings = Depends(get_settings),
    metrics: ApiMetrics = Depends(get_metrics),
) -> BatchResponse:
    _check_size(len(batch.records), settings)
    preds = _run(batch.records, service, repo, metrics, source="batch")
    return BatchResponse(count=len(preds), predictions=preds)


@router.post("/batch-predict/csv", response_model=BatchResponse)
def batch_predict_csv(
    file: UploadFile = File(..., description="CSV with the raw dataset columns"),
    service: ModelService = Depends(get_model_service),
    repo: PredictionRepository | None = Depends(get_repository),
    settings: Settings = Depends(get_settings),
    metrics: ApiMetrics = Depends(get_metrics),
) -> BatchResponse:
    """Upload a CSV in the original dataset layout. A `Churn` column, if present, is ignored."""
    try:
        frame = pd.read_csv(io.BytesIO(file.file.read()))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}") from exc
    if frame.empty:
        raise HTTPException(422, "CSV has no rows")
    _check_size(len(frame), settings)
    frame = frame.drop(columns=["Churn"], errors="ignore")
    if "TotalCharges" in frame:
        frame["TotalCharges"] = pd.to_numeric(
            frame["TotalCharges"].astype("string").str.strip().replace("", pd.NA), errors="coerce"
        )
    frame = frame.astype(object).where(frame.notna(), None)

    customers, errors = [], []
    for i, row in enumerate(frame.to_dict(orient="records")):
        try:
            customers.append(CustomerFeatures.model_validate(row))
        except ValidationError as exc:
            errors.append({"row": i, "errors": exc.errors(include_url=False, include_input=False)})
    if errors:
        raise HTTPException(422, {"message": f"{len(errors)} invalid rows", "rows": errors[:20]})
    preds = _run(customers, service, repo, metrics, source="csv")
    return BatchResponse(count=len(preds), predictions=preds)


@router.post("/explain", response_model=ExplanationResponse)
def explain(
    customer: CustomerFeatures,
    service: ModelService = Depends(get_model_service),
) -> ExplanationResponse:
    if not service.can_explain:
        raise HTTPException(501, "Explanations are not available for this model version")
    return ExplanationResponse(**service.explain(customer.features()))
