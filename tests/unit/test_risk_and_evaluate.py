import numpy as np
import pytest

from src.risk import risk_level
from src.training.evaluate import compute_metrics, decision_cost, select_cost_threshold


@pytest.mark.parametrize(
    ("p", "band"),
    [(0.0, "low"), (0.2999, "low"), (0.30, "medium"), (0.60, "medium"), (0.6001, "high"), (1.0, "high")],
)
def test_risk_bands(p, band):
    assert risk_level(p) == band


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_risk_band_rejects_invalid_probability(p):
    with pytest.raises(ValueError):
        risk_level(p)


def test_compute_metrics_keys_and_confusion_counts():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    m = compute_metrics(y, p, threshold=0.5)
    assert {"roc_auc", "pr_auc", "brier", "precision", "recall", "f1"} <= m.keys()
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)


def test_cost_threshold_prefers_recall_when_false_negatives_are_expensive():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, 2000), 0, 1)
    t, cost = select_cost_threshold(y, p, fn_cost=5, fp_cost=1)
    assert t < 0.5
    assert cost == decision_cost(y, p, t, 5, 1)
    assert cost <= decision_cost(y, p, 0.5, 5, 1)


def test_evaluation_plots_are_written(tmp_path):
    from src.training.evaluate import save_evaluation_plots

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.4 * y + rng.uniform(0, 0.6, 300), 0, 1)
    paths = save_evaluation_plots(y, p, 0.5, tmp_path, "t")
    assert len(paths) == 3 and all(x.exists() and x.stat().st_size > 0 for x in paths)
