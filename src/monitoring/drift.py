"""Data drift job (Evidently) comparing current data against the training reference.

Current data comes either from a SIMULATED batch file or from the API prediction log.

Usage:
  python -m src.monitoring.drift --scenario pricing_shift
  python -m src.monitoring.drift --all-scenarios
  python -m src.monitoring.drift --from-db --last 500

Outputs artifacts/monitoring/YYYY-MM-DD/<HHMMSS>_<name>/{report.html,summary.json}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.config import (
    CATEGORICAL_FEATURES,
    CHAMPION_EXPORT_DIR,
    ID_COL,
    MONITORING_ARTIFACTS_DIR,
    MONITORING_CONFIG,
    MONITORING_DATA_DIR,
    NUMERIC_FEATURES,
    RAW_FEATURES,
    TARGET_COL,
)
from src.data.schema import validate_feature_frame
from src.monitoring.batch_generator import SCENARIOS, SIMULATED_DIR
from src.risk import risk_level

logger = logging.getLogger(__name__)
DATASET_DRIFT_SHARE: float = MONITORING_CONFIG["drift_share"]  # dataset drift if >= this share drifts


def _is_drifted(method: str, value: float, threshold: float) -> bool:
    # Statistical tests report p-values (drift if small); distances drift if large.
    return value < threshold if "p_value" in method.lower() else value >= threshold


def schema_checks(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    return {
        "missing_columns": sorted(set(RAW_FEATURES) - set(current.columns)),
        "unexpected_columns": sorted(set(current.columns) - set(RAW_FEATURES) - {ID_COL, TARGET_COL}),
        "missing_share": {c: round(float(current[c].isna().mean()), 4) for c in RAW_FEATURES if c in current},
        "reference_missing_share": {c: round(float(reference[c].isna().mean()), 4) for c in RAW_FEATURES},
    }


def prediction_stats(probabilities: pd.Series, model_version: str | None) -> dict:
    bands = probabilities.map(risk_level)
    return {
        "n_predictions": len(probabilities),
        "avg_churn_probability": round(float(probabilities.mean()), 4),
        "high_risk_share": round(float((bands == "high").mean()), 4),
        "risk_distribution": {k: int(v) for k, v in bands.value_counts().items()},
        "model_version": model_version,
    }


def run_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    name: str,
    out_root: Path = MONITORING_ARTIFACTS_DIR,
    simulated: bool = True,
    predictions: dict | None = None,
) -> dict:
    schema = schema_checks(reference, current)
    # Contract check before drift: a malformed batch is a data-quality incident, not "drift".
    current_features = validate_feature_frame(current)
    definition = DataDefinition(numerical_columns=NUMERIC_FEATURES, categorical_columns=CATEGORICAL_FEATURES)
    snapshot = Report([DataDriftPreset(drift_share=DATASET_DRIFT_SHARE)]).run(
        current_data=Dataset.from_pandas(current_features, data_definition=definition),
        reference_data=Dataset.from_pandas(reference[RAW_FEATURES], data_definition=definition),
    )
    features = []
    for metric in snapshot.dict()["metrics"]:
        cfg = metric["config"]
        if not cfg.get("type", "").endswith("ValueDrift"):
            continue
        value = float(metric["value"])
        features.append(
            {
                "feature": cfg["column"],
                "method": cfg["method"],
                "score": round(value, 6),
                "threshold": cfg["threshold"],
                "drifted": _is_drifted(cfg["method"], value, cfg["threshold"]),
            }
        )
    features.sort(key=lambda f: (not f["drifted"], -f["score"]))
    n_drifted = sum(f["drifted"] for f in features)

    now = datetime.now(UTC)
    out_dir = out_root / now.strftime("%Y-%m-%d") / f"{now.strftime('%H%M%S')}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_dir / "report.html"))
    summary = {
        "run_at": now.isoformat(),
        "name": name,
        "simulated": simulated,
        "n_reference": len(reference),
        "n_current": len(current),
        "n_features": len(features),
        "n_drifted": n_drifted,
        "share_drifted": round(n_drifted / max(len(features), 1), 4),
        "dataset_drift": n_drifted / max(len(features), 1) >= DATASET_DRIFT_SHARE,
        "features": features,
        "schema": schema,
        "predictions": predictions,
        "report_html": "report.html",  # relative to this summary.json
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Drift %-16s drifted=%d/%d -> %s", name, n_drifted, len(features), out_dir)
    return summary


def _score_with_champion(current: pd.DataFrame, model_uri: str) -> dict | None:
    if not Path(model_uri).exists():
        return None
    import mlflow

    model = mlflow.sklearn.load_model(model_uri)
    version = (mlflow.models.get_model_info(model_uri).metadata or {}).get("model_version")
    proba = pd.Series(model.predict_proba(current[RAW_FEATURES])[:, 1])
    return prediction_stats(proba, version)


def load_from_db(database_url: str, last: int) -> tuple[pd.DataFrame, dict, bool]:
    from app.db import PredictionRepository

    rows = PredictionRepository(database_url).recent(limit=last)
    if not rows:
        raise SystemExit("No predictions logged yet.")
    current = pd.DataFrame([r.request_features for r in rows])
    proba = pd.Series([r.probability for r in rows])
    versions = sorted({r.model_version for r in rows})
    simulated = any((r.customer_id or "").startswith("SIM-") for r in rows)
    return current, prediction_stats(proba, ",".join(versions)), simulated


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", choices=sorted(SCENARIOS))
    group.add_argument("--all-scenarios", action="store_true")
    group.add_argument("--from-db", action="store_true")
    parser.add_argument("--last", type=int, default=500, help="rows from DB (with --from-db)")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "sqlite:///data/churnops.db"))
    parser.add_argument("--model-uri", default=str(CHAMPION_EXPORT_DIR))
    args = parser.parse_args()

    reference = pd.read_csv(MONITORING_DATA_DIR / "reference.csv")
    if args.from_db:
        current, preds, simulated = load_from_db(args.database_url, args.last)
        run_drift(reference, current, "api_log", simulated=simulated, predictions=preds)
        return
    names = list(SCENARIOS) if args.all_scenarios else [args.scenario]
    for name in names:
        path = SIMULATED_DIR / f"{name}.csv"
        if not path.exists():
            raise SystemExit(f"{path} missing. Run `make simulate` first.")
        current = pd.read_csv(path)
        run_drift(reference, current, name, predictions=_score_with_champion(current, args.model_uri))


if __name__ == "__main__":
    main()
