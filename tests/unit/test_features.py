import numpy as np
import pandas as pd

from src.features.build import FeatureEngineer


def _row(**overrides):
    base = {
        "tenure": 10,
        "TotalCharges": 500.0,
        "PhoneService": "Yes",
        "InternetService": "DSL",
        "Contract": "Month-to-month",
        "TechSupport": "Yes",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_engineered_features_values():
    out = FeatureEngineer().transform(_row())
    assert out.loc[0, "service_count"] == 4  # phone + internet + security + support
    assert out.loc[0, "avg_charge_per_tenure"] == 50.0
    assert out.loc[0, "is_month_to_month"] == 1.0
    assert out.loc[0, "has_support"] == 1.0


def test_zero_tenure_does_not_divide_by_zero():
    out = FeatureEngineer().transform(_row(tenure=0, TotalCharges=30.0))
    assert out.loc[0, "avg_charge_per_tenure"] == 30.0


def test_blank_total_charges_become_nan():
    out = FeatureEngineer().transform(_row(TotalCharges=" "))
    assert np.isnan(out.loc[0, "TotalCharges"])


def test_feature_engineering_is_deterministic_and_pure(clean_df):
    fe = FeatureEngineer()
    before = clean_df.copy()
    a, b = fe.transform(clean_df), fe.transform(clean_df)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(clean_df, before)
