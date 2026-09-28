"""Stratified train/validation/test split (the dataset has no time axis).

Usage:  python -m src.data.split
"""

from __future__ import annotations

import json
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, RANDOM_STATE, TARGET_COL, TEST_SIZE, VALID_SIZE
from src.data.validate import load_validated

logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame,
    valid_size: float = VALID_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdout = valid_size + test_size
    train_df, temp_df = train_test_split(
        df, test_size=holdout, stratify=df[TARGET_COL], random_state=random_state
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / holdout,
        stratify=temp_df[TARGET_COL],
        random_state=random_state,
    )
    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = load_validated()
    parts = dict(zip(("train", "valid", "test"), split_data(df), strict=True))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, part in parts.items():
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        summary[name] = {"rows": len(part), "churn_rate": round(float(part[TARGET_COL].mean()), 4)}
    (PROCESSED_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Split summary: %s", summary)


if __name__ == "__main__":
    main()
