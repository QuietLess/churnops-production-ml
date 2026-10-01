"""Prometheus metrics exposed on GET /metrics.

Operational metrics (updated on every request):
  churnops_http_requests_total{path,method,status}
  churnops_http_request_latency_seconds{path}            histogram -> p50/p95/p99 in Grafana
  churnops_predictions_total{model_version,risk_level,source}
  churnops_churn_probability                             histogram of model outputs
  churnops_prediction_log_failures_total
  churnops_feedback_total{actual_outcome}
  churnops_model_info{model_name,model_alias,model_version}   always 1 (lineage label)

Model-quality metrics (computed at scrape time from the prediction log, cached briefly):
  churnops_live_labelled_predictions{model_version}
  churnops_live_roc_auc / _pr_auc / _f1 / _precision / _recall {model_version}
These only appear once enough predictions have received a ground-truth label through
POST /feedback (see `performance_min_labelled`). Without labels there is nothing to measure.

Each app instance owns its own CollectorRegistry, so tests can create many apps.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

logger = logging.getLogger(__name__)

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
PROBABILITY_BUCKETS = tuple(round(0.1 * i, 1) for i in range(1, 11))


class LivePerformanceCollector(Collector):
    """Computes ROC-AUC / PR-AUC / F1 on the latest labelled predictions, per model version."""

    CACHE_SECONDS = 30.0

    def __init__(self, repository, window: int, min_labelled: int):
        self.repository = repository
        self.window = window
        self.min_labelled = min_labelled
        self._cache: tuple[float, list[GaugeMetricFamily]] | None = None
        self._lock = threading.Lock()

    def collect(self) -> Iterable[GaugeMetricFamily]:
        with self._lock:
            if self._cache is None or time.monotonic() - self._cache[0] > self.CACHE_SECONDS:
                self._cache = (time.monotonic(), self._compute())
            return list(self._cache[1])

    def _compute(self) -> list[GaugeMetricFamily]:
        from src.monitoring.performance import performance_by_version

        try:
            rows = self.repository.labelled(limit=self.window)
        except Exception:
            logger.exception("Could not read labelled predictions for live metrics")
            return []
        labelled = GaugeMetricFamily(
            "churnops_live_labelled_predictions",
            "Predictions with ground truth in the live-metrics window",
            labels=["model_version"],
        )
        gauges = {
            name: GaugeMetricFamily(
                f"churnops_live_{name}", f"Live {name} on labelled predictions", labels=["model_version"]
            )
            for name in ("roc_auc", "pr_auc", "f1", "precision", "recall")
        }
        for version, result in performance_by_version(rows, self.min_labelled).items():
            labelled.add_metric([version], result["n_labelled"])
            for name, gauge in gauges.items():
                value = (result.get("metrics") or {}).get(name)
                if value is not None:
                    gauge.add_metric([version], value)
        return [labelled, *gauges.values()]


class ApiMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "churnops_http_requests",
            "HTTP requests",
            ["path", "method", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "churnops_http_request_latency_seconds",
            "Request latency",
            ["path"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.predictions = Counter(
            "churnops_predictions",
            "Predictions served",
            ["model_version", "risk_level", "source"],
            registry=self.registry,
        )
        self.probability = Histogram(
            "churnops_churn_probability",
            "Predicted churn probability",
            buckets=PROBABILITY_BUCKETS,
            registry=self.registry,
        )
        self.log_failures = Counter(
            "churnops_prediction_log_failures", "Failed prediction-log writes", registry=self.registry
        )
        self.feedback = Counter(
            "churnops_feedback", "Ground-truth labels received", ["actual_outcome"], registry=self.registry
        )
        self.model_info = Gauge(
            "churnops_model_info",
            "Serving model lineage (value is always 1)",
            ["model_name", "model_alias", "model_version"],
            registry=self.registry,
        )
        self._live: LivePerformanceCollector | None = None

    def set_model(self, name: str, alias: str | None, version: str) -> None:
        self.model_info.clear()
        self.model_info.labels(name, alias or "", version).set(1)

    def enable_live_performance(self, repository, window: int, min_labelled: int) -> None:
        """Idempotent: a restarted lifespan replaces the collector instead of adding a second one."""
        if self._live is not None:
            self.registry.unregister(self._live)
        self._live = LivePerformanceCollector(repository, window, min_labelled)
        self.registry.register(self._live)

    def observe_request(self, path: str, method: str, status: int, seconds: float) -> None:
        self.requests.labels(path, method, str(status)).inc()
        self.latency.labels(path).observe(seconds)

    def observe_predictions(self, results: list[dict], model_version: str, source: str) -> None:
        for r in results:
            self.predictions.labels(model_version, r["risk_level"], source).inc()
            self.probability.observe(r["churn_probability"])
