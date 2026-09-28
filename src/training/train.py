"""Train candidate models, select a champion candidate and log everything to MLflow.

Usage:  python -m src.training.train [--quick]

Selection protocol (the test set is only touched once, at the very end):
1. Fit every candidate on train, score on validation.
2. Best validation PR-AUC wins; candidates within PR_AUC_TOLERANCE of the best are
   tie-broken by Brier score (better calibration), then by simplicity.
3. Decision threshold = lowest hypothetical cost on validation (FN=5, FP=1).
4. Evaluate the selected pipeline once on test, log it and register a new version.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field

import mlflow
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import (
    ARTIFACTS_DIR,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    FEATURE_VERSION,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MONITORING_DATA_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_FEATURES,
    REGISTERED_MODEL_NAME,
    RISK_HIGH_LOWER,
    RISK_LOW_UPPER,
    TARGET_COL,
)
from src.data.split import split_data
from src.data.validate import load_validated
from src.features.build import build_pipeline, model_feature_names
from src.training.evaluate import compute_metrics, save_evaluation_plots, select_cost_threshold
from src.training.serialization import trusted_types_for

logger = logging.getLogger(__name__)

PR_AUC_TOLERANCE = 0.005
EXPLAIN_BACKGROUND_ROWS = 50
TRAINING_REPORT = ARTIFACTS_DIR / "training"


@dataclass
class Candidate:
    name: str
    family: str
    complexity: int  # lower = simpler; used as final tie-breaker
    params: dict
    make: callable = field(repr=False)


def candidates(quick: bool = False) -> list[Candidate]:
    lr_params = {"C": 1.0, "max_iter": 2000, "class_weight": None}
    rf_params = {"n_estimators": 400, "min_samples_leaf": 5, "max_features": "sqrt"}
    lgbm_grid = [
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 40},
        {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 7, "min_child_samples": 60},
        {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20},
    ]
    if quick:
        rf_params["n_estimators"] = 100
        lgbm_grid = lgbm_grid[:1]

    def lgbm(p):
        return LGBMClassifier(
            **p,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1,
        )

    out = [
        Candidate(
            "logreg",
            "logistic_regression",
            0,
            lr_params,
            lambda: build_pipeline(LogisticRegression(**lr_params, random_state=RANDOM_STATE)),
        ),
        Candidate(
            "random_forest",
            "random_forest",
            2,
            rf_params,
            lambda: build_pipeline(
                RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, n_jobs=-1),
                scale_numeric=False,
            ),
        ),
    ]
    for i, p in enumerate(lgbm_grid, start=1):
        out.append(
            Candidate(
                f"lightgbm_v{i}", "lightgbm", 3, p, lambda p=p: build_pipeline(lgbm(p), scale_numeric=False)
            )
        )
        out.append(
            Candidate(
                f"lightgbm_v{i}_calibrated",
                "lightgbm+sigmoid_calibration",
                4,
                {**p, "calibration": "sigmoid", "calibration_cv": 5},
                lambda p=p: CalibratedClassifierCV(
                    build_pipeline(lgbm(p), scale_numeric=False), method="sigmoid", cv=5
                ),
            )
        )
    return out


def select_best(results: list[dict]) -> dict:
    best_pr = max(r["valid"]["pr_auc"] for r in results)
    close = [r for r in results if r["valid"]["pr_auc"] >= best_pr - PR_AUC_TOLERANCE]
    return min(close, key=lambda r: (round(r["valid"]["brier"], 4), r["complexity"]))


def global_importance(model, X_valid: pd.DataFrame, out_path) -> pd.DataFrame | None:
    """SHAP summary for tree pipelines, coefficient magnitudes for logistic regression."""
    pipe = model.calibrated_classifiers_[0].estimator if hasattr(model, "calibrated_classifiers_") else model
    est = pipe.named_steps["model"]
    X_t = pipe[:-1].transform(X_valid)
    names = model_feature_names(pipe)
    import matplotlib.pyplot as plt

    if isinstance(est, LogisticRegression):
        imp = pd.DataFrame({"feature": names, "importance": np.abs(est.coef_[0])})
    else:
        explainer = shap.TreeExplainer(est)
        values = explainer.shap_values(X_t)
        values = values[1] if isinstance(values, list) else values
        if values.ndim == 3:
            values = values[:, :, 1]
        shap.summary_plot(values, X_t, feature_names=names, show=False, max_display=15)
        plt.gcf().tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close("all")
        imp = pd.DataFrame({"feature": names, "importance": np.abs(values).mean(axis=0)})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def run(quick: bool = False) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    TRAINING_REPORT.mkdir(parents=True, exist_ok=True)

    df = load_validated()
    train_df, valid_df, test_df = split_data(df)
    X_train, y_train = train_df[RAW_FEATURES], train_df[TARGET_COL]
    X_valid, y_valid = valid_df[RAW_FEATURES], valid_df[TARGET_COL]
    X_test, y_test = test_df[RAW_FEATURES], test_df[TARGET_COL]
    logger.info("Split sizes train=%d valid=%d test=%d", len(X_train), len(X_valid), len(X_test))

    # Persist splits + drift reference (training feature distribution)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    MONITORING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df[RAW_FEATURES].to_csv(MONITORING_DATA_DIR / "reference.csv", index=False)

    session = time.strftime("%Y%m%d-%H%M%S")
    results: list[dict] = []
    for cand in candidates(quick):
        start = time.perf_counter()
        model = cand.make()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        proba = model.predict_proba(X_valid)[:, 1]
        valid_metrics = compute_metrics(y_valid, proba, 0.5)
        with mlflow.start_run(run_name=cand.name):
            mlflow.set_tags({"session": session, "stage": "candidate", "family": cand.family})
            mlflow.log_params(
                {
                    **cand.params,
                    "model_family": cand.family,
                    "random_state": RANDOM_STATE,
                    "feature_version": FEATURE_VERSION,
                }
            )
            mlflow.log_metrics({f"valid_{k}": v for k, v in valid_metrics.items()})
            mlflow.log_metric("fit_seconds", fit_seconds)
        logger.info(
            "%-26s valid PR-AUC=%.4f ROC-AUC=%.4f Brier=%.4f",
            cand.name,
            valid_metrics["pr_auc"],
            valid_metrics["roc_auc"],
            valid_metrics["brier"],
        )
        results.append(
            {
                "name": cand.name,
                "family": cand.family,
                "complexity": cand.complexity,
                "params": cand.params,
                "model": model,
                "valid": valid_metrics,
                "valid_proba": proba,
            }
        )

    comparison = pd.DataFrame(
        [
            {"model": r["name"], **{k: round(r["valid"][k], 4) for k in ("pr_auc", "roc_auc", "brier")}}
            for r in results
        ]
    ).sort_values("pr_auc", ascending=False)
    comparison.to_csv(TRAINING_REPORT / "model_comparison.csv", index=False)

    best = select_best(results)
    model = best["model"]
    threshold, val_cost = select_cost_threshold(y_valid, best["valid_proba"])
    valid_final = compute_metrics(y_valid, best["valid_proba"], threshold)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_final = compute_metrics(y_test, test_proba, threshold)  # evaluated ONCE
    logger.info(
        "Selected %s | threshold=%.2f | test PR-AUC=%.4f ROC-AUC=%.4f",
        best["name"],
        threshold,
        test_final["pr_auc"],
        test_final["roc_auc"],
    )

    plots = save_evaluation_plots(y_test, test_proba, threshold, TRAINING_REPORT, "test")
    plots += save_evaluation_plots(y_valid, best["valid_proba"], threshold, TRAINING_REPORT, "valid")
    importance = global_importance(model, X_valid, TRAINING_REPORT / "shap_summary.png")
    importance.to_csv(TRAINING_REPORT / "feature_importance.csv", index=False)

    background = X_train.sample(EXPLAIN_BACKGROUND_ROWS, random_state=RANDOM_STATE)
    metadata = {
        "threshold": threshold,
        "threshold_strategy": f"min_validation_cost(fn={FALSE_NEGATIVE_COST},fp={FALSE_POSITIVE_COST})",
        "selected_candidate": best["name"],
        "model_family": best["family"],
        "feature_version": FEATURE_VERSION,
        "raw_features": RAW_FEATURES,
        "risk_bands": {"low_upper": RISK_LOW_UPPER, "high_lower": RISK_HIGH_LOWER},
        "valid_pr_auc": round(valid_final["pr_auc"], 6),
        "explain_background": json.loads(background.to_json(orient="records")),
    }
    summary = {
        "selected_model": best["name"],
        "threshold": threshold,
        "validation_cost": val_cost,
        "validation": valid_final,
        "test": test_final,
        "candidates": comparison.to_dict(orient="records"),
        "data": {
            "train": len(X_train),
            "valid": len(X_valid),
            "test": len(X_test),
            "churn_rate": round(float(df[TARGET_COL].mean()), 4),
        },
    }

    input_example = X_valid.head(3)
    signature = infer_signature(input_example, model.predict_proba(input_example)[:, 1])
    with mlflow.start_run(run_name=f"{best['name']}_champion_candidate") as run:
        mlflow.set_tags({"session": session, "stage": "selected", "family": best["family"]})
        mlflow.log_params(
            {
                **best["params"],
                "model_family": best["family"],
                "selected_candidate": best["name"],
                "threshold": threshold,
                "threshold_strategy": metadata["threshold_strategy"],
                "feature_version": FEATURE_VERSION,
                "random_state": RANDOM_STATE,
            }
        )
        mlflow.log_metrics({f"valid_{k}": v for k, v in valid_final.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_final.items()})
        for p in [
            *plots,
            TRAINING_REPORT / "model_comparison.csv",
            TRAINING_REPORT / "feature_importance.csv",
        ]:
            mlflow.log_artifact(str(p), artifact_path="evaluation")
        if (TRAINING_REPORT / "shap_summary.png").exists():
            mlflow.log_artifact(str(TRAINING_REPORT / "shap_summary.png"), artifact_path="evaluation")
        mlflow.log_dict({k: v for k, v in summary.items()}, "evaluation/metrics.json")
        info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            metadata=metadata,
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=trusted_types_for(model),
        )
        version = str(info.registered_model_version)
        client = mlflow.MlflowClient()
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME, version, "valid_pr_auc", str(valid_final["pr_auc"])
        )
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "threshold", str(threshold))
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "candidate", best["name"])
        summary["mlflow"] = {
            "run_id": run.info.run_id,
            "model_uri": info.model_uri,
            "registered_model": REGISTERED_MODEL_NAME,
            "version": version,
        }

    (TRAINING_REPORT / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Registered %s version %s. Promote with `make promote`.", REGISTERED_MODEL_NAME, version)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", action="store_true", help="smaller candidate set (for CI)")
    args = parser.parse_args()
    summary = run(quick=args.quick)
    print(json.dumps({k: summary[k] for k in ("selected_model", "threshold", "test")}, indent=2))


if __name__ == "__main__":
    main()
