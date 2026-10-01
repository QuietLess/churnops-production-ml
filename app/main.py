"""ChurnOps inference service.

Run locally:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.db import PredictionRepository
from app.logging_config import configure_logging
from app.metrics import ApiMetrics
from app.model_service import ModelService
from app.routes import feedback, health, model_info, prediction
from app.security import PROTECTED, RateLimiter
from app.settings import Settings

logger = logging.getLogger("churnops.api")

DESCRIPTION = """
Churn **risk** classification for telecom customers (IBM Telco sample data).

The model scores a customer snapshot; it is **not** a time-bound forecast
("will churn in the next 30 days"), because the dataset has no temporal snapshots.
"""


def create_app(
    settings: Settings | None = None,
    model_service: ModelService | None = None,
    repository: PredictionRepository | None = None,
) -> FastAPI:
    """App factory. Tests inject a model service / repository; production loads them."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        # Fail fast: if the champion model cannot be loaded the process must not start.
        app.state.model_service = model_service or ModelService.load(
            settings.model_uri, settings.mlflow_tracking_uri
        )
        app.state.model_service.warm_up()
        svc = app.state.model_service
        app.state.metrics.set_model(svc.model_name, svc.model_alias, svc.model_version)
        if repository is not None:
            app.state.repository = repository
        elif settings.log_predictions:
            repo = PredictionRepository(settings.database_url)
            repo.init_schema()
            app.state.repository = repo
        else:
            app.state.repository = None
        if app.state.repository is not None:
            app.state.metrics.enable_live_performance(
                app.state.repository, settings.performance_window, settings.performance_min_labelled
            )
        if not settings.auth_enabled:
            logger.warning(
                "API_KEYS is not set: scoring endpoints are UNAUTHENTICATED (ok for local dev only)",
                extra={"event": "auth_disabled"},
            )
        logger.info(
            "API ready", extra={"event": "startup", "model_version": app.state.model_service.model_version}
        )
        yield

    app = FastAPI(
        title="ChurnOps API",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.metrics = ApiMetrics()
    app.state.rate_limiter = RateLimiter(settings.rate_limit_per_minute)

    @app.middleware("http")
    async def timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        seconds = time.perf_counter() - start
        elapsed = round(seconds * 1000, 2)
        response.headers["X-Process-Time-Ms"] = str(elapsed)
        # Label by route template, never the raw URL, so unknown paths can't explode cardinality.
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        if path not in {"/metrics", "/docs", "/openapi.json"}:
            app.state.metrics.observe_request(path, request.method, response.status_code, seconds)
        if request.url.path not in {"/health", "/docs", "/openapi.json", "/metrics"}:
            logger.info(
                "request",
                extra={
                    "event": "request",
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": elapsed,
                },
            )
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Prometheus scrape endpoint. Keep it on the internal network (not behind API keys)."""
        return Response(generate_latest(app.state.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    app.include_router(health.router)
    app.include_router(model_info.router)
    app.include_router(prediction.router, dependencies=PROTECTED)
    app.include_router(feedback.router, dependencies=PROTECTED)
    return app


app = create_app()
