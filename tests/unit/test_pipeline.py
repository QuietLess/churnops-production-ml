import numpy as np
import pandas as pd

from src.config import RAW_FEATURES, TARGET_COL
from src.data.split import split_data
from src.features.build import model_feature_names


def test_split_is_stratified_disjoint_and_complete(clean_df):
    train, valid, test = split_data(clean_df)
    assert len(train) + len(valid) + len(test) == len(clean_df)
    ids = [set(x["customerID"]) for x in (train, valid, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    rate = clean_df[TARGET_COL].mean()
    for part in (train, valid, test):
        assert abs(part[TARGET_COL].mean() - rate) < 0.05


def test_split_is_reproducible(clean_df):
    a = split_data(clean_df)[0]["customerID"].tolist()
    b = split_data(clean_df)[0]["customerID"].tolist()
    assert a == b


def test_preprocessing_outputs_finite_values(fitted_pipeline, clean_df):
    transformed = fitted_pipeline[:-1].transform(clean_df[RAW_FEATURES])
    assert np.isfinite(np.asarray(transformed, dtype=float)).all()


def test_no_id_or_target_leakage(fitted_pipeline):
    names = model_feature_names(fitted_pipeline)
    assert not any("customerID" in n or "Churn" in n for n in names)


def test_customer_id_does_not_change_prediction(fitted_pipeline, clean_df):
    X = clean_df[["customerID", *RAW_FEATURES]].head(10)
    X2 = X.assign(customerID="SOMETHING-ELSE")
    np.testing.assert_allclose(fitted_pipeline.predict_proba(X), fitted_pipeline.predict_proba(X2))


def test_unknown_category_is_handled_by_encoder(fitted_pipeline, clean_df):
    X = clean_df[RAW_FEATURES].head(1).copy()
    X["PaymentMethod"] = "Crypto"
    p = fitted_pipeline.predict_proba(X)[:, 1]
    assert 0 <= p[0] <= 1


def test_prediction_shape_and_range(fitted_pipeline, clean_df):
    p = fitted_pipeline.predict_proba(clean_df[RAW_FEATURES])
    assert p.shape == (len(clean_df), 2)
    assert ((p >= 0) & (p <= 1)).all()


def test_missing_total_charges_is_imputed(fitted_pipeline, clean_df):
    X = clean_df[RAW_FEATURES].head(1).copy()
    X["TotalCharges"] = np.nan
    assert np.isfinite(fitted_pipeline.predict_proba(pd.DataFrame(X))).all()
