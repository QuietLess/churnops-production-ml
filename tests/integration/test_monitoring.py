import json

import pytest

from src.config import RAW_FEATURES
from src.monitoring.batch_generator import SCENARIOS, generate
from src.monitoring.drift import _is_drifted, run_drift

pytestmark = pytest.mark.integration


def test_batches_are_deterministic_and_labelled(clean_df):
    a = generate(clean_df, "pricing_shift", n=50, seed=1)
    b = generate(clean_df, "pricing_shift", n=50, seed=1)
    assert a.equals(b)
    assert a["customerID"].str.startswith("SIM-pricing_shift-").all()
    assert list(a.columns) == ["customerID", *RAW_FEATURES]


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_every_scenario_generates(clean_df, scenario):
    assert len(generate(clean_df, scenario, n=30)) == 30


def test_pricing_shift_raises_charges(clean_df):
    base = generate(clean_df, "baseline", n=300, seed=5)
    shifted = generate(clean_df, "pricing_shift", n=300, seed=5)
    assert shifted["MonthlyCharges"].mean() > base["MonthlyCharges"].mean() * 1.1


def test_drift_direction_depends_on_method():
    assert _is_drifted("K-S p_value", 0.01, 0.05) is True
    assert _is_drifted("K-S p_value", 0.30, 0.05) is False
    assert _is_drifted("Wasserstein distance (normed)", 0.3, 0.1) is True


def test_drift_job_produces_report_artifact(clean_df, tmp_path):
    reference = clean_df[RAW_FEATURES]
    current = generate(clean_df, "contract_mix", n=300, seed=2)
    summary = run_drift(reference, current, "contract_mix", out_root=tmp_path)
    out_files = list(tmp_path.glob("*/*/"))
    assert (out_files[0] / "report.html").exists()
    saved = json.loads((out_files[0] / "summary.json").read_text())
    assert saved["n_features"] == len(RAW_FEATURES)
    assert saved["simulated"] is True
    assert any(f["feature"] == "Contract" and f["drifted"] for f in summary["features"])
