"""Shared fixtures. Tests use synthetic, schema-valid data so CI needs no download."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from app.db import PredictionRepository
from app.main import create_app
from app.model_service import ModelService
from app.schemas import EXAMPLE_CUSTOMER
from app.settings import Settings
from src.config import INTERNET_ADDON_COLS, RAW_FEATURES, TARGET_COL
from src.features.build import build_pipeline


def make_raw_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic data in the RAW dataset format (Churn Yes/No, TotalCharges as strings)."""
    rng = np.random.default_rng(seed)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    tenure = rng.integers(0, 73, n)
    monthly = np.round(rng.uniform(18, 120, n), 2)
    total = np.where(tenure == 0, " ", np.round(monthly * np.maximum(tenure, 1), 2).astype(str))
    df = pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(n)],
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n)),
            "InternetService": internet,
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
    )
    for col in INTERNET_ADDON_COLS:
        df[col] = np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))
    logit = -1.2 + 1.5 * (contract == "Month-to-month") - 0.03 * tenure + 0.6 * (internet == "Fiber optic")
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    df["Churn"] = np.where(churn, "Yes", "No")
    return df[["customerID", *RAW_FEATURES, "Churn"]]


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_raw_df()


@pytest.fixture(scope="session")
def clean_df(raw_df) -> pd.DataFrame:
    from src.data.validate import clean

    return clean(raw_df)


@pytest.fixture(scope="session")
def fitted_pipeline(clean_df):
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(clean_df[RAW_FEATURES], clean_df[TARGET_COL])
    return pipe


@pytest.fixture(scope="session")
def model_service(fitted_pipeline, clean_df) -> ModelService:
    background = clean_df[RAW_FEATURES].head(20)
    return ModelService(
        model=fitted_pipeline,
        threshold=0.4,
        model_name="churnops-model",
        model_version="test",
        model_alias="champion",
        metadata={
            "threshold": 0.4,
            "model_family": "logistic_regression",
            "explain_background": background.astype(object)
            .where(background.notna(), None)
            .to_dict(orient="records"),
        },
    )


@pytest.fixture()
def repository(tmp_path) -> PredictionRepository:
    repo = PredictionRepository(f"sqlite:///{tmp_path / 'test.db'}")
    repo.init_schema()
    return repo


@pytest.fixture()
def client(model_service, repository):
    app = create_app(settings=Settings(), model_service=model_service, repository=repository)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def customer() -> dict:
    return dict(EXAMPLE_CUSTOMER)
