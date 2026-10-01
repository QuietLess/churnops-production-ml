from types import SimpleNamespace

import numpy as np
import pytest

from src.monitoring.performance import compare_to_baseline, evaluate, performance_by_version

CFG = {"min_labelled": 20, "max_roc_auc_drop": 0.05, "max_f1_drop": 0.10}


def rows(n, version="1", seed=0, informative=True, outcome=None):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n) if outcome is None else np.full(n, outcome)
    noise = rng.uniform(0, 1, n)
    proba = np.clip(0.7 * y + 0.3 * noise, 0, 1) if informative else noise
    return [
        SimpleNamespace(model_version=version, probability=p, threshold=0.5, actual_outcome=int(t))
        for p, t in zip(proba, y, strict=True)
    ]


def test_unlabelled_rows_are_ignored_and_small_groups_flagged():
    unlabelled = SimpleNamespace(model_version="1", probability=0.3, threshold=0.5, actual_outcome=None)
    result = performance_by_version([*rows(10), unlabelled], min_labelled=20)["1"]
    assert result["n_labelled"] == 10
    assert result["status"] == "insufficient_labels" and result["metrics"] is None


def test_single_class_has_no_auc():
    assert performance_by_version(rows(30, outcome=0), 20)["1"]["status"] == "single_class"


def test_metrics_per_version():
    result = performance_by_version(rows(200, "1") + rows(200, "2", informative=False), 20)
    assert set(result) == {"1", "2"}
    assert result["1"]["status"] == "ok"
    assert result["1"]["metrics"]["roc_auc"] > 0.95
    assert result["2"]["metrics"]["roc_auc"] < result["1"]["metrics"]["roc_auc"]
    assert sum(result["1"]["confusion"].values()) == 200


@pytest.mark.parametrize(("live_auc", "degraded"), [(0.83, False), (0.70, True)])
def test_degradation_against_baseline(live_auc, degraded):
    baseline = {"roc_auc": 0.84, "f1": 0.58}
    out = compare_to_baseline({"roc_auc": live_auc, "f1": 0.57}, baseline, CFG)
    assert out["degraded"] is degraded
    assert out["checks"]["roc_auc"]["drop"] == pytest.approx(0.84 - live_auc)


def test_evaluate_lists_degraded_versions():
    baseline = {"roc_auc": 0.84, "f1": 0.58}
    summary = evaluate(rows(200, "good") + rows(200, "bad", informative=False), baseline, CFG)
    assert summary["degraded_versions"] == ["bad"]
    assert "comparison" in summary["versions"]["good"]
