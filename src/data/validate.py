"""Raw data loading and validation.

Validation failures raise ``DataValidationError`` with a readable message rather than
bare asserts, so a changed or corrupted dataset stops the pipeline loudly.
The raw file is never modified.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CATEGORY_VALUES,
    EXPECTED_COLUMNS,
    ID_COL,
    RAW_DATA_PATH,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the raw dataset violates the expected contract."""


def to_numeric_charges(series: pd.Series) -> pd.Series:
    """Convert TotalCharges to float: blank strings -> NaN, garbage -> error."""
    as_str = series.astype("string").str.strip()
    blank = as_str.isna() | (as_str == "")
    numeric = pd.to_numeric(as_str.where(~blank), errors="coerce")
    bad = numeric.isna() & ~blank
    if bad.any():
        examples = series[bad].head(3).tolist()
        raise DataValidationError(f"TotalCharges has non-numeric values, e.g. {examples}")
    return numeric.astype(float)


def validate_raw(df: pd.DataFrame) -> dict:
    """Validate the raw dataframe. Returns a small summary for logging."""
    missing = EXPECTED_COLUMNS - set(df.columns)
    extra = set(df.columns) - EXPECTED_COLUMNS
    if missing or extra:
        raise DataValidationError(f"Schema mismatch. missing={sorted(missing)} unexpected={sorted(extra)}")
    if df.empty:
        raise DataValidationError("Dataset is empty")
    if df[ID_COL].isna().any():
        raise DataValidationError(f"{ID_COL} contains null values")
    dupes = df.loc[df[ID_COL].duplicated(), ID_COL]
    if not dupes.empty:
        raise DataValidationError(
            f"{ID_COL} must be unique; {len(dupes)} duplicates e.g. {dupes.head(3).tolist()}"
        )
    target_values = set(df[TARGET_COL].dropna().unique())
    if df[TARGET_COL].isna().any() or not target_values <= {"Yes", "No"}:
        raise DataValidationError(f"{TARGET_COL} must be Yes/No, found {sorted(target_values)}")

    for col, allowed in CATEGORY_VALUES.items():
        unknown = set(df[col].dropna().unique()) - set(allowed)
        if unknown:
            raise DataValidationError(f"{col} has unexpected categories: {sorted(unknown)}")
    if not df["SeniorCitizen"].isin([0, 1]).all():
        raise DataValidationError("SeniorCitizen must be 0/1")

    charges = to_numeric_charges(df["TotalCharges"])
    for col, values in (
        ("tenure", df["tenure"]),
        ("MonthlyCharges", df["MonthlyCharges"]),
        ("TotalCharges", charges),
    ):
        if (values.dropna() < 0).any():
            raise DataValidationError(f"{col} contains negative values")

    summary = {
        "rows": len(df),
        "columns": int(df.shape[1]),
        "churn_rate": float((df[TARGET_COL] == "Yes").mean()),
        "blank_total_charges": int(charges.isna().sum()),
    }
    logger.info("Raw data validated: %s", summary)
    return summary


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a typed copy: numeric TotalCharges and 0/1 target."""
    out = df.copy()
    out["TotalCharges"] = to_numeric_charges(out["TotalCharges"])
    out[TARGET_COL] = (out[TARGET_COL] == "Yes").astype(np.int64)
    return out


def load_validated(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make data` first.")
    raw = pd.read_csv(path)
    validate_raw(raw)
    return clean(raw)
