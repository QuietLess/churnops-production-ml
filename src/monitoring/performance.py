"""Model *performance* monitoring: compare live metrics against the offline test baseline.

Drift monitoring (drift.py) only needs inputs. Performance monitoring needs the truth:
did the customer actually churn? That label arrives later (delayed ground truth) and is
attached to the logged prediction through POST /feedback -> prediction_logs.actual_outcome.

This job reads the most recent labelled predictions, computes ROC-AUC / PR-AUC / F1 /
precision / recall per model version, and flags a version as DEGRADED when it is clearly
worse than its frozen test metrics (artifacts/training/metrics.json).

Usage:
  python -m src.monitoring.performance                       # report
  python -m src.monitoring.performance --fail-on-degradation # exit 1 if degraded (cron / CI)

Outputs artifacts/monitoring/YYYY-MM-DD/<HHMMSS>_performance/summary.json

Caveat: with this static dataset the labels are SIMULATED (they come from the held-out
test split, see batch_generator --feedback). In a real system they would come from the
CRM/billing system once the churn window has passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import ARTIFACTS_DIR, MONITORING_ARTIFACTS_DIR, MONITORING_CONFIG
from src.training.evaluate import compute_metrics

logger = logging.getLogger(__name__)

PERFORMANCE_CONFIG: dict[str, Any] = MONITORING_CONFIG["performance"]
REPORTED_METRICS = ("roc_auc", "pr_auc", "f1", "precision", "recall", "brier")
BASELINE_PATH = ARTIFACTS_DIR / "training" / "metrics.json"


def performance_by_version(rows: Iterable, min_labelled: int) -> dict[str, dict]:
    """Group labelled prediction-log rows by model version and score each group.

    `rows` need: model_version, probability, threshold, actual_outcome (0/1).
    status: ok | insufficient_labels | single_class (AUC undefined with one class).
    """
    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        if row.actual_outcome is not None:
            groups[str(row.model_version)].append(row)

    results: dict[str, dict] = {}
    for version, group in sorted(groups.items()):
        y = [int(r.actual_outcome) for r in group]
        proba = [float(r.probability) for r in group]
        result: dict[str, Any] = {"n_labelled": len(group), "positive_rate": round(sum(y) / len(y), 4)}
        if len(group) < min_labelled:
            result.update(status="insufficient_labels", metrics=None)
        elif len(set(y)) < 2:
            result.update(status="single_class", metrics=None)
        else:
            # The threshold is part of the model version's metadata, so it is constant per group.
            metrics = compute_metrics(y, proba, float(group[0].threshold))
            result.update(
                status="ok",
                metrics={k: round(v, 4) for k, v in metrics.items() if k in REPORTED_METRICS},
                confusion={k: metrics[k] for k in ("tp", "fp", "tn", "fn")},
            )
        results[version] = result
    return results


def compare_to_baseline(live: dict[str, float], baseline: dict[str, float], cfg: dict) -> dict:
    """Degraded when ROC-AUC or F1 falls more than the configured tolerance below the baseline."""
    checks = {}
    for metric, max_drop in (("roc_auc", cfg["max_roc_auc_drop"]), ("f1", cfg["max_f1_drop"])):
        drop = round(baseline[metric] - live[metric], 4)
        checks[metric] = {
            "live": live[metric],
            "baseline": round(baseline[metric], 4),
            "drop": drop,
            "max_drop": max_drop,
            "degraded": drop > max_drop,
        }
    return {"degraded": any(c["degraded"] for c in checks.values()), "checks": checks}


def evaluate(rows: Iterable, baseline: dict[str, float] | None, cfg: dict = PERFORMANCE_CONFIG) -> dict:
    versions = performance_by_version(rows, cfg["min_labelled"])
    for result in versions.values():
        if result["status"] == "ok" and baseline:
            result["comparison"] = compare_to_baseline(result["metrics"], baseline, cfg)
    return {
        "versions": versions,
        "degraded_versions": sorted(
            v for v, r in versions.items() if r.get("comparison", {}).get("degraded")
        ),
    }


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, float] | None:
    """Frozen test-set metrics written by src.training.train."""
    if not path.exists():
        return None
    return json.loads(path.read_text())["test"]


def write_summary(summary: dict, out_root: Path = MONITORING_ARTIFACTS_DIR) -> Path:
    now = datetime.fromisoformat(summary["run_at"])
    out_dir = out_root / now.strftime("%Y-%m-%d") / f"{now.strftime('%H%M%S')}_performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "sqlite:///data/churnops.db"))
    parser.add_argument("--window", type=int, default=PERFORMANCE_CONFIG["window"])
    parser.add_argument("--fail-on-degradation", action="store_true", help="exit 1 if any version degraded")
    args = parser.parse_args()

    from app.db import PredictionRepository

    rows = PredictionRepository(args.database_url).labelled(limit=args.window)
    if not rows:
        raise SystemExit("No labelled predictions yet. Send outcomes with POST /feedback.")
    baseline = load_baseline()
    summary = {
        "run_at": datetime.now(UTC).isoformat(),
        "name": "performance",
        "window": args.window,
        "baseline_source": str(BASELINE_PATH.relative_to(ARTIFACTS_DIR.parent)) if baseline else None,
        "simulated": any((r.customer_id or "").startswith("SIM-") for r in rows),
        **evaluate(rows, baseline),
    }
    out_dir = write_summary(summary)
    for version, result in summary["versions"].items():
        logger.info(
            "model v%s: %s n=%d %s",
            version,
            result["status"],
            result["n_labelled"],
            result.get("metrics") or "",
        )
    logger.info("Performance summary -> %s", out_dir)
    if summary["degraded_versions"]:
        logger.warning(
            "DEGRADED model version(s): %s. Investigate drift, then retrain or roll back "
            "(`python -m src.training.register --rollback`).",
            summary["degraded_versions"],
        )
        if args.fail_on_degradation:
            sys.exit(1)


if __name__ == "__main__":
    main()
