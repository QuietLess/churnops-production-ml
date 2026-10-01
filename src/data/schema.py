"""Declarative DataFrame contracts (Pandera) for data *after* cleaning.

`validate.validate_raw` guards the raw CSV (string TotalCharges, Yes/No target) with
hand-written, very specific error messages. These schemas guard the typed frames that
flow *between* pipeline stages:

- FEATURE_SCHEMA: the 19 raw model inputs (used for scoring / drift batches)
- TRAINING_SCHEMA: FEATURE_SCHEMA + customerID + 0/1 Churn target (used before splitting)

Both are generated from the single feature contract in src/config.py, so a new category
or column only has to be added in one place. Validation is lazy: all failures are
collected and reported together as a DataValidationError.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from src.config import CATEGORY_VALUES, ID_COL, INTERNET_ADDON_COLS, RAW_FEATURES, TARGET_COL
from src.data.validate import DataValidationError

MAX_ERRORS_SHOWN = 10


def _internet_addons_consistent(df: pd.DataFrame) -> pd.Series:
    """Add-ons are 'No internet service' exactly when InternetService == 'No'."""
    no_internet = df["InternetService"] == "No"
    addon_says_none = df[INTERNET_ADDON_COLS] == "No internet service"
    return addon_says_none.eq(no_internet, axis=0).all(axis=1)


def _phone_lines_consistent(df: pd.DataFrame) -> pd.Series:
    """MultipleLines is 'No phone service' exactly when PhoneService == 'No'."""
    return (df["PhoneService"] == "No") == (df["MultipleLines"] == "No phone service")


def _feature_columns() -> dict[str, pa.Column]:
    columns: dict[str, pa.Column] = {
        col: pa.Column(str, pa.Check.isin(values), nullable=False) for col, values in CATEGORY_VALUES.items()
    }
    columns.update(
        {
            "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
            "tenure": pa.Column(int, pa.Check.ge(0), coerce=True),
            "MonthlyCharges": pa.Column(float, pa.Check.ge(0), coerce=True),
            # Blank in the raw data for brand-new customers (tenure == 0) -> NaN is allowed.
            "TotalCharges": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        }
    )
    return {col: columns[col] for col in RAW_FEATURES}  # keep the contract's column order


_CROSS_FIELD_CHECKS = [
    pa.Check(_internet_addons_consistent, error="internet add-ons inconsistent with InternetService"),
    pa.Check(_phone_lines_consistent, error="MultipleLines inconsistent with PhoneService"),
]

FEATURE_SCHEMA = pa.DataFrameSchema(
    _feature_columns(),
    checks=_CROSS_FIELD_CHECKS,
    strict="filter",  # extra columns (e.g. customerID in a batch) are dropped, not fed to the model
    name="churn_features",
)

TRAINING_SCHEMA = pa.DataFrameSchema(
    {
        ID_COL: pa.Column(str, unique=True, nullable=False),
        **_feature_columns(),
        TARGET_COL: pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
    },
    checks=_CROSS_FIELD_CHECKS,
    strict=True,  # the training table must have exactly these columns
    name="churn_training_table",
)


def _validate(schema: pa.DataFrameSchema, df: pd.DataFrame) -> pd.DataFrame:
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases.copy()
        frame_level = cases["schema_context"] == "DataFrameSchema"
        # Structural problems (missing / unexpected column): failure_case is the column name.
        structural = frame_level & cases["index"].isna()
        lines = [f"table: {row.check} {row.failure_case!r}" for row in cases[structural].itertuples()]
        # Cross-field checks are reported once per row, not once per cell of that row.
        cases.loc[frame_level, "column"] = "<row>"
        for (column, check), group in cases[~structural].groupby(["column", "check"], sort=False):
            rows = sorted({int(i) for i in group["index"].dropna()})
            example = "" if column == "<row>" else f", e.g. {group['failure_case'].iloc[0]!r}"
            lines.append(f"{column}: {check} -> {len(rows) or len(group)} row(s){example}, rows {rows[:5]}")
        raise DataValidationError(
            f"{schema.name} failed {len(lines)} check(s):\n  " + "\n  ".join(lines[:MAX_ERRORS_SHOWN])
        ) from exc


def validate_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a cleaned training table (output of validate.clean). Returns the coerced frame."""
    return _validate(TRAINING_SCHEMA, df)


def validate_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a batch of model inputs. Returns only the RAW_FEATURES columns, coerced."""
    return _validate(FEATURE_SCHEMA, df)
