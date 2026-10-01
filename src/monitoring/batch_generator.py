"""Generate SIMULATED "production" batches from the held-out test split.

The IBM dataset is static, so these batches are a production *simulation*, not real
future traffic. Every scenario is deterministic (fixed seeds) and labelled SIM-*.

Usage:
  python -m src.monitoring.batch_generator                     # write all scenarios
  python -m src.monitoring.batch_generator --send http://localhost:8000   # also POST to API
  python -m src.monitoring.batch_generator --send http://localhost:8000 --feedback
      # ...and then POST the true Churn label of each customer to /feedback (simulated
      # delayed ground truth -> live performance metrics). Set CHURNOPS_API_KEY if the
      # API requires a key.

Caveat for --feedback: labels are the original customers' outcomes. In scenarios that
perturb values (pricing_shift) the "true" label would also change in reality; here it does not.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ID_COL,
    MONITORING_CONFIG,
    MONITORING_DATA_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_FEATURES,
    TARGET_COL,
)

logger = logging.getLogger(__name__)
SIMULATED_DIR = MONITORING_DATA_DIR / "simulated"
DEFAULT_BATCH_SIZE: int = MONITORING_CONFIG["simulated_batch_size"]


def _weighted_sample(df: pd.DataFrame, weights: pd.Series, n: int, seed: int) -> pd.DataFrame:
    return df.sample(n=n, replace=True, weights=weights, random_state=seed)


def baseline(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """No intentional drift: a plain resample of unseen customers."""
    return df.sample(n=n, replace=True, random_state=seed)


def pricing_shift(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Monthly prices rise 15-25%; TotalCharges scaled consistently."""
    out = baseline(df, n, seed).copy()
    factor = np.random.default_rng(seed).uniform(1.15, 1.25, size=len(out))
    out["MonthlyCharges"] = (out["MonthlyCharges"] * factor).round(2)
    out["TotalCharges"] = (out["TotalCharges"] * factor).round(2)
    return out


def contract_mix(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Many more month-to-month contracts."""
    w = np.where(df["Contract"] == "Month-to-month", 6.0, 1.0)
    return _weighted_sample(df, pd.Series(w, index=df.index), n, seed)


def maturity_shift(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Customer base skews towards low tenure (new customers)."""
    w = np.where(df["tenure"] <= 12, 8.0, 1.0)
    return _weighted_sample(df, pd.Series(w, index=df.index), n, seed)


def channel_shift(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Payment channel mix moves to electronic check."""
    w = np.where(df["PaymentMethod"] == "Electronic check", 6.0, 1.0)
    return _weighted_sample(df, pd.Series(w, index=df.index), n, seed)


SCENARIOS: dict[str, Callable[[pd.DataFrame, int, int], pd.DataFrame]] = {
    "baseline": baseline,
    "pricing_shift": pricing_shift,
    "contract_mix": contract_mix,
    "maturity_shift": maturity_shift,
    "channel_shift": channel_shift,
}


def generate(
    source: pd.DataFrame,
    scenario: str,
    n: int = DEFAULT_BATCH_SIZE,
    seed: int = RANDOM_STATE,
    keep_target: bool = False,
) -> pd.DataFrame:
    """Build a batch. keep_target=True also carries the true Churn label (for --feedback)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    columns = [*RAW_FEATURES, TARGET_COL] if keep_target else RAW_FEATURES
    batch = SCENARIOS[scenario](source[columns], n, seed).reset_index(drop=True)
    batch.insert(0, ID_COL, [f"SIM-{scenario}-{i:05d}" for i in range(len(batch))])
    return batch


def send_to_api(
    batch: pd.DataFrame,
    api_url: str,
    chunk: int = 200,
    feedback: bool = False,
    api_key: str | None = None,
) -> int:
    """POST the batch to /batch-predict; with feedback=True also POST its labels to /feedback."""
    import httpx

    if feedback and TARGET_COL not in batch:
        raise ValueError("feedback=True needs a batch generated with keep_target=True")
    labels = batch[TARGET_COL].tolist() if TARGET_COL in batch else None
    features = batch.drop(columns=[TARGET_COL], errors="ignore")
    records = features.astype(object).where(features.notna(), None).to_dict(orient="records")
    headers = {"X-API-Key": api_key} if api_key else {}
    sent = 0
    with httpx.Client(base_url=api_url, timeout=60, headers=headers) as client:
        for i in range(0, len(records), chunk):
            resp = client.post("/batch-predict", json={"records": records[i : i + chunk]})
            resp.raise_for_status()
            predictions = resp.json()["predictions"]
            sent += len(predictions)
            if feedback and labels is not None:
                outcomes = [
                    {"prediction_id": p["prediction_id"], "actual_outcome": int(y)}
                    for p, y in zip(predictions, labels[i : i + chunk], strict=True)
                ]
                client.post("/feedback", json={"outcomes": outcomes}).raise_for_status()
    return sent


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), action="append")
    parser.add_argument("--n", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--source", type=Path, default=PROCESSED_DIR / "test.csv")
    parser.add_argument("--send", metavar="API_URL", help="POST batches to a running API")
    parser.add_argument(
        "--feedback", action="store_true", help="with --send: also POST true labels to /feedback"
    )
    args = parser.parse_args()
    if args.feedback and not args.send:
        parser.error("--feedback requires --send")

    source = pd.read_csv(args.source)
    SIMULATED_DIR.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(args.scenario or list(SCENARIOS)):
        batch = generate(source, name, n=args.n, seed=RANDOM_STATE + i, keep_target=args.feedback)
        path = SIMULATED_DIR / f"{name}.csv"
        batch.drop(columns=[TARGET_COL], errors="ignore").to_csv(path, index=False)  # features only
        logger.info("SIMULATED batch %-15s rows=%d -> %s", name, len(batch), path)
        if args.send:
            n_sent = send_to_api(
                batch, args.send, feedback=args.feedback, api_key=os.getenv("CHURNOPS_API_KEY")
            )
            logger.info("Sent %d records to %s (feedback=%s)", n_sent, args.send, args.feedback)


if __name__ == "__main__":
    main()
