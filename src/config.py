"""Central project configuration.

Everything that training, serving and monitoring must agree on lives here, so the
feature contract is defined exactly once.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# --- Paths -----------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "telco_churn.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MONITORING_DATA_DIR = DATA_DIR / "monitoring"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MONITORING_ARTIFACTS_DIR = ARTIFACTS_DIR / "monitoring"
CHAMPION_EXPORT_DIR = ARTIFACTS_DIR / "champion_model"

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

# --- Reproducibility -------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.15

# --- Data contract ---------------------------------------------------------
ID_COL = "customerID"
TARGET_COL = "Churn"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

YES_NO = ["No", "Yes"]
INTERNET_ADDON = ["No", "No internet service", "Yes"]

CATEGORY_VALUES: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": INTERNET_ADDON,
    "OnlineBackup": INTERNET_ADDON,
    "DeviceProtection": INTERNET_ADDON,
    "TechSupport": INTERNET_ADDON,
    "StreamingTV": INTERNET_ADDON,
    "StreamingMovies": INTERNET_ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}
CATEGORICAL_FEATURES = list(CATEGORY_VALUES)

INTERNET_ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Raw input fields the model consumes (order matters for signatures / CSV export).
RAW_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
EXPECTED_COLUMNS = {ID_COL, TARGET_COL, *RAW_FEATURES}

FEATURE_VERSION = "v1"

# --- Decision policy (portfolio assumptions, NOT real telecom economics) ----
FALSE_NEGATIVE_COST = 5.0
FALSE_POSITIVE_COST = 1.0
RISK_LOW_UPPER = 0.30  # p < 0.30 -> low
RISK_HIGH_LOWER = 0.60  # p > 0.60 -> high, otherwise medium

# --- MLflow ----------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "churnops_training")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "churnops-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
