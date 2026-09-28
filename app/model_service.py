"""Loads the champion pipeline once and serves predictions / explanations."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.config import CATEGORY_VALUES, RAW_FEATURES
from src.risk import risk_level

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    pass


@dataclass
class ModelService:
    model: object  # sklearn-compatible, exposes predict_proba on raw feature frames
    threshold: float
    model_name: str
    model_version: str
    model_alias: str | None = None
    model_uri: str = "in-memory"
    metadata: dict = field(default_factory=dict)
    loaded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    _explainer: object | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_uri: str, tracking_uri: str | None = None) -> ModelService:
        """Load from the MLflow registry (models:/name@alias) or an exported model dir."""
        import mlflow

        start = time.perf_counter()
        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            model = mlflow.sklearn.load_model(model_uri)
            metadata = dict(mlflow.models.get_model_info(model_uri).metadata or {})
            name, alias, version = cls._lineage(model_uri, metadata)
        except Exception as exc:
            raise ModelLoadError(f"Could not load model from {model_uri!r}: {exc}") from exc
        if "threshold" not in metadata:
            raise ModelLoadError("Model metadata has no 'threshold'; retrain with src.training.train")
        service = cls(
            model=model,
            threshold=float(metadata["threshold"]),
            model_name=name,
            model_version=version,
            model_alias=alias,
            model_uri=model_uri,
            metadata=metadata,
        )
        logger.info(
            "Model loaded",
            extra={
                "event": "model_loaded",
                "model_version": version,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return service

    @staticmethod
    def _lineage(model_uri: str, metadata: dict) -> tuple[str, str | None, str]:
        if model_uri.startswith("models:/"):
            from mlflow import MlflowClient

            ref = model_uri.removeprefix("models:/")
            client = MlflowClient()
            if "@" in ref:
                name, alias = ref.split("@", 1)
                mv = client.get_model_version_by_alias(name, alias)
                return name, alias, str(mv.version)
            name, version = ref.split("/", 1)
            return name, None, version
        # Exported directory: lineage was written into metadata at export time.
        return (
            metadata.get("model_name", "churnops-model"),
            metadata.get("model_alias"),
            str(metadata.get("model_version", "unknown")),
        )

    # --------------------------------------------------------------- inference
    def predict_proba(self, records: list[dict]) -> np.ndarray:
        frame = pd.DataFrame.from_records(records, columns=RAW_FEATURES)
        proba = np.asarray(self.model.predict_proba(frame)[:, 1], dtype=float)
        if not np.all(np.isfinite(proba)):
            raise ValueError("Model produced non-finite probabilities")
        return proba

    def score(self, records: list[dict]) -> list[dict]:
        proba = self.predict_proba(records)
        return [
            {
                "prediction": int(p >= self.threshold),
                "churn_probability": round(float(p), 6),
                "risk_level": risk_level(float(p)),
            }
            for p in proba
        ]

    # ----------------------------------------------------------- explanations
    @property
    def can_explain(self) -> bool:
        return bool(self.metadata.get("explain_background"))

    def _encode(self, frame: pd.DataFrame) -> np.ndarray:
        out = frame[RAW_FEATURES].copy()
        for col, values in CATEGORY_VALUES.items():
            out[col] = out[col].map({v: i for i, v in enumerate(values)})
        return out.astype(float).to_numpy()

    def _decode(self, arr: np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame(arr, columns=RAW_FEATURES)
        for col, values in CATEGORY_VALUES.items():
            frame[col] = [values[round(i)] for i in frame[col]]
        return frame

    def _get_explainer(self):
        if self._explainer is None:
            import shap

            background = self._encode(pd.DataFrame(self.metadata["explain_background"]))
            self._explainer = shap.PermutationExplainer(
                lambda a: self.model.predict_proba(self._decode(a))[:, 1],
                shap.maskers.Independent(background, max_samples=len(background)),
                seed=0,
            )
        return self._explainer

    def warm_up(self) -> None:
        if self.can_explain:
            self.explain(dict(self.metadata["explain_background"][0]))

    def explain(self, record: dict, top_k: int = 5) -> dict:
        """Permutation SHAP over raw features: contributions are in probability units."""
        if not self.can_explain:
            raise RuntimeError("This model version has no explanation background data")
        x = self._encode(pd.DataFrame.from_records([record], columns=RAW_FEATURES))
        sv = self._get_explainer()(x, max_evals=2 * len(RAW_FEATURES) + 1)
        contribs = [
            {"feature": f, "value": record.get(f), "contribution": round(float(v), 6)}
            for f, v in zip(RAW_FEATURES, sv.values[0], strict=True)
        ]
        contribs.sort(key=lambda c: c["contribution"], reverse=True)
        return {
            "churn_probability": round(float(sv.base_values[0] + sv.values[0].sum()), 6),
            "base_value": round(float(sv.base_values[0]), 6),
            "model_version": self.model_version,
            "top_increasing": [c for c in contribs if c["contribution"] > 0][:top_k],
            "top_decreasing": [c for c in reversed(contribs) if c["contribution"] < 0][:top_k],
        }
