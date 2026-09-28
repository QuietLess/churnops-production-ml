"""Metrics, cost-based threshold selection and evaluation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FALSE_NEGATIVE_COST, FALSE_POSITIVE_COST


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": float(threshold),
    }


def decision_cost(
    y_true,
    proba,
    threshold: float,
    fn_cost: float = FALSE_NEGATIVE_COST,
    fp_cost: float = FALSE_POSITIVE_COST,
) -> float:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return fn_cost * fn + fp_cost * fp


def select_cost_threshold(
    y_true,
    proba,
    fn_cost: float = FALSE_NEGATIVE_COST,
    fp_cost: float = FALSE_POSITIVE_COST,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return (threshold, cost) minimising validation cost. Ties -> higher threshold."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    best_t, best_cost = 0.5, float("inf")
    for t in grid:
        cost = decision_cost(y_true, proba, float(t), fn_cost, fp_cost)
        if cost <= best_cost:
            best_t, best_cost = float(t), cost
    return best_t, best_cost


def save_evaluation_plots(y_true, proba, threshold: float, out_dir: Path, prefix: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, pred, display_labels=["Stay", "Churn"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix (t={threshold:.2f})")
    paths.append(_save(fig, out_dir / f"{prefix}_confusion_matrix.png"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, proba):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0].legend(loc="lower right")
    prec, rec, _ = precision_recall_curve(y_true, proba)
    axes[1].plot(rec, prec, label=f"AP={average_precision_score(y_true, proba):.3f}")
    axes[1].axhline(y_true.mean(), ls="--", color="grey", label="base rate")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axes[1].legend(loc="upper right")
    paths.append(_save(fig, out_dir / f"{prefix}_roc_pr_curves.png"))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed churn rate", title="Calibration")
    ax.legend(loc="upper left")
    paths.append(_save(fig, out_dir / f"{prefix}_calibration.png"))
    return paths


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
