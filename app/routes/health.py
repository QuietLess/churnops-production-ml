from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request, Response

from app.schemas import HealthResponse

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    """Readiness: 200 only when the model is loaded. DB problems degrade but don't fail."""
    service = getattr(request.app.state, "model_service", None)
    repo = getattr(request.app.state, "repository", None)
    db_status: Literal["ok", "unavailable", "disabled"] = (
        "disabled" if repo is None else ("ok" if repo.ping() else "unavailable")
    )
    if service is None:
        response.status_code = 503
    return HealthResponse(
        status="ok" if service is not None and db_status != "unavailable" else "degraded",
        model_loaded=service is not None,
        model_version=None if service is None else service.model_version,
        database=db_status,
    )
