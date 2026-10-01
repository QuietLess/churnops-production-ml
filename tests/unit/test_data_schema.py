import numpy as np
import pytest

from src.config import RAW_FEATURES
from src.data.schema import validate_feature_frame, validate_training_frame
from src.data.validate import DataValidationError


def test_clean_training_frame_passes(clean_df):
    out = validate_training_frame(clean_df)
    assert out.shape == clean_df.shape
    assert out["TotalCharges"].isna().sum() == clean_df["TotalCharges"].isna().sum()  # blanks allowed


def test_all_problems_reported_together(clean_df):
    bad = clean_df.copy()
    bad.loc[0, "tenure"] = -3
    bad.loc[1, "Contract"] = "Weekly"
    bad.loc[2, "Churn"] = 2
    with pytest.raises(DataValidationError) as err:
        validate_training_frame(bad)
    message = str(err.value)
    assert "tenure" in message and "Contract" in message and "Churn" in message
    assert "'Weekly'" in message


def test_cross_field_rule_reported_per_row(clean_df):
    bad = clean_df.copy()
    row = bad.index[bad["InternetService"] != "No"][0]
    bad.loc[row, "InternetService"] = "No"  # but its add-ons still say Yes/No
    with pytest.raises(DataValidationError, match=r"<row>: internet add-ons .* rows \[" + str(row)):
        validate_training_frame(bad)


def test_training_table_rejects_extra_columns_and_duplicate_ids(clean_df):
    with pytest.raises(DataValidationError, match="column_in_schema"):
        validate_training_frame(clean_df.assign(leak=1))
    dupes = clean_df.copy()
    dupes.loc[1, "customerID"] = dupes.loc[0, "customerID"]
    with pytest.raises(DataValidationError, match="customerID"):
        validate_training_frame(dupes)


def test_feature_frame_drops_non_features_and_coerces(clean_df):
    batch = clean_df.head(10).copy()
    batch["tenure"] = batch["tenure"].astype(float)
    out = validate_feature_frame(batch)  # customerID and Churn are filtered out
    assert list(out.columns) == RAW_FEATURES
    assert out["tenure"].dtype == np.int64


def test_feature_frame_missing_column_fails(clean_df):
    with pytest.raises(DataValidationError, match="MonthlyCharges"):
        validate_feature_frame(clean_df.drop(columns=["MonthlyCharges"]))
