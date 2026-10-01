"""Ground-truth feedback: attach the real outcome to a logged prediction.

Churn is only known weeks after scoring. When the CRM learns whether a scored customer
actually left, it posts the outcome here keyed by the `prediction_id` that /predict returned.
Labelled rows feed live performance metrics (/metrics) and src.monitoring.performance.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.db import PredictionRepository
from app.dependencies import get_repository
from app.schemas import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(
    body: FeedbackRequest,
    request: Request,
    repo: PredictionRepository | None = Depends(get_repository),
) -> FeedbackResponse:
    if repo is None:
        raise HTTPException(503, "Prediction logging is disabled; feedback cannot be stored")
    outcomes = {item.prediction_id: item.actual_outcome for item in body.outcomes}
    try:
        not_found = repo.set_outcomes(outcomes)
    except Exception as exc:
        logger.exception("Feedback write failed", extra={"event": "feedback_write_failed"})
        raise HTTPException(503, "Could not store feedback") from exc
    metrics = request.app.state.metrics
    for prediction_id, outcome in outcomes.items():
        if prediction_id not in not_found:
            metrics.feedback.labels(str(outcome)).inc()
    return FeedbackResponse(updated=len(outcomes) - len(not_found), not_found=not_found)
