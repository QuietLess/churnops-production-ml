"""M1 baseline: logistic regression, no MLflow. Proves the data + preprocessing path.

Usage:  python -m src.training.baseline
Writes artifacts/baseline_metrics.json and artifacts/baseline_confusion_matrix.png
"""

from __future__ import annotations

import json
import logging

from sklearn.linear_model import LogisticRegression

from src.config import ARTIFACTS_DIR, RANDOM_STATE, RAW_FEATURES, TARGET_COL
from src.data.split import split_data
from src.data.validate import load_validated
from src.features.build import build_pipeline
from src.training.evaluate import compute_metrics, save_evaluation_plots

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    train_df, valid_df, _ = split_data(load_validated())
    pipe = build_pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    pipe.fit(train_df[RAW_FEATURES], train_df[TARGET_COL])
    proba = pipe.predict_proba(valid_df[RAW_FEATURES])[:, 1]
    metrics = compute_metrics(valid_df[TARGET_COL], proba, threshold=0.5)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "baseline_metrics.json").write_text(
        json.dumps({"model": "logistic_regression", "split": "validation", **metrics}, indent=2)
    )
    plots = save_evaluation_plots(valid_df[TARGET_COL], proba, 0.5, ARTIFACTS_DIR, "baseline")
    plots[0].rename(ARTIFACTS_DIR / "baseline_confusion_matrix.png")
    for p in plots[1:]:
        p.unlink()
    logger.info("Baseline (validation): %s", {k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
