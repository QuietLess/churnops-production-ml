import numpy as np
import pandas as pd
import pytest

from src.data.validate import DataValidationError, clean, to_numeric_charges, validate_raw


def test_valid_data_passes(raw_df):
    summary = validate_raw(raw_df)
    assert summary["rows"] == len(raw_df)
    assert 0 < summary["churn_rate"] < 1


def test_duplicate_customer_id_rejected(raw_df):
    df = pd.concat([raw_df, raw_df.head(1)], ignore_index=True)
    with pytest.raises(DataValidationError, match="unique"):
        validate_raw(df)


def test_missing_column_rejected(raw_df):
    with pytest.raises(DataValidationError, match="missing"):
        validate_raw(raw_df.drop(columns=["tenure"]))


def test_invalid_target_rejected(raw_df):
    df = raw_df.copy()
    df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DataValidationError, match="Churn"):
        validate_raw(df)


def test_negative_tenure_rejected(raw_df):
    df = raw_df.copy()
    df.loc[0, "tenure"] = -3
    with pytest.raises(DataValidationError, match="tenure"):
        validate_raw(df)


def test_unknown_category_rejected(raw_df):
    df = raw_df.copy()
    df.loc[0, "Contract"] = "Lifetime"
    with pytest.raises(DataValidationError, match="Contract"):
        validate_raw(df)


def test_total_charges_blanks_become_nan():
    out = to_numeric_charges(pd.Series(["10.5", " ", "", "7"]))
    assert out.iloc[0] == 10.5 and out.iloc[3] == 7.0
    assert out.iloc[1:3].isna().all()


def test_total_charges_garbage_raises():
    with pytest.raises(DataValidationError, match="non-numeric"):
        to_numeric_charges(pd.Series(["10", "abc"]))


def test_clean_encodes_target_and_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    cleaned = clean(raw_df)
    assert set(cleaned["Churn"].unique()) <= {0, 1}
    assert cleaned["TotalCharges"].dtype == np.float64
    pd.testing.assert_frame_equal(raw_df, before)
