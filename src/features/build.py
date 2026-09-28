"""Feature engineering + preprocessing, shared by training and inference.

The whole object graph (engineering -> preprocessing -> estimator) is serialized as
one sklearn Pipeline, so the API never re-implements preprocessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, INTERNET_ADDON_COLS, NUMERIC_FEATURES

ENGINEERED_NUMERIC = ["service_count", "avg_charge_per_tenure", "is_month_to_month", "has_support"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless, deterministic feature engineering on raw business fields.

    - service_count: number of active services (phone, internet, add-ons)
    - avg_charge_per_tenure: TotalCharges / max(tenure, 1)
    - is_month_to_month: Contract == "Month-to-month"
    - has_support: TechSupport == "Yes"

    Also coerces TotalCharges to numeric (blank -> NaN) so raw-looking input is safe.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        total = pd.to_numeric(
            out["TotalCharges"].astype("string").str.strip().replace("", pd.NA),
            errors="coerce",
        ).astype(float)
        out["TotalCharges"] = total
        active = (out[INTERNET_ADDON_COLS] == "Yes").sum(axis=1)
        active += (out["PhoneService"] == "Yes").astype(int)
        active += (out["InternetService"] != "No").astype(int)
        out["service_count"] = active.astype(float)
        tenure = pd.to_numeric(out["tenure"], errors="coerce").astype(float)
        out["avg_charge_per_tenure"] = total / np.maximum(tenure, 1.0)
        out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(float)
        out["has_support"] = (out["TechSupport"] == "Yes").astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.asarray(base + ENGINEERED_NUMERIC, dtype=object)


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    numeric_steps: list = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES + ENGINEERED_NUMERIC),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # customerID (and anything unexpected) never reaches the model
        verbose_feature_names_out=True,
    )


def build_pipeline(estimator, scale_numeric: bool = True) -> Pipeline:
    return Pipeline(
        [
            ("features", FeatureEngineer()),
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def model_feature_names(pipeline: Pipeline) -> list[str]:
    """Names of the columns the estimator actually sees (after preprocessing)."""
    return [str(n) for n in pipeline.named_steps["preprocess"].get_feature_names_out()]
