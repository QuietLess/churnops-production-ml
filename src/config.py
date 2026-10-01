"""Central project configuration.

Everything that training, serving and monitoring must agree on lives here, so the
feature contract is defined exactly once.

Two kinds of settings:
- The *feature contract* (columns, allowed categories) is code: the API schema is typed against it.
- *Tunable values* (hyperparameters, split sizes, costs, monitoring thresholds) come from
  configs/config.yaml, or from the file named by the CHURNOPS_CONFIG env var.
Deployment settings (MLflow URI, model name/alias) can additionally be overridden by env vars.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config. Relative paths are resolved against the project root."""
    path = Path(path or os.getenv("CHURNOPS_CONFIG") or DEFAULT_CONFIG_PATH)
    if not path.is_absolute():
        path = ROOT_DIR / path
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CONFIG = load_config()
TRAINING_CONFIG: dict[str, Any] = CONFIG["training"]
MONITORING_CONFIG: dict[str, Any] = CONFIG["monitoring"]

# --- Paths -----------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "telco_churn.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MONITORING_DATA_DIR = DATA_DIR / "monitoring"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MONITORING_ARTIFACTS_DIR = ARTIFACTS_DIR / "monitoring"
CHAMPION_EXPORT_DIR = ARTIFACTS_DIR / "champion_model"

RAW_DATA_URL: str = CONFIG["data"]["raw_url"]

# --- Reproducibility -------------------------------------------------------
RANDOM_STATE: int = CONFIG["data"]["random_state"]
TEST_SIZE: float = CONFIG["data"]["test_size"]
VALID_SIZE: float = CONFIG["data"]["valid_size"]

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

FEATURE_VERSION: str = CONFIG["features"]["version"]

# --- Decision policy (portfolio assumptions, NOT real telecom economics) ----
_policy = CONFIG["decision_policy"]
FALSE_NEGATIVE_COST: float = _policy["false_negative_cost"]
FALSE_POSITIVE_COST: float = _policy["false_positive_cost"]
RISK_LOW_UPPER: float = _policy["risk_low_upper"]  # p < low_upper -> low
RISK_HIGH_LOWER: float = _policy["risk_high_lower"]  # p > high_lower -> high, otherwise medium

# --- MLflow (env vars override the YAML for deployment) ----------------------
_mlflow = CONFIG["mlflow"]
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", _mlflow["experiment"])
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", _mlflow["registered_model_name"])
MODEL_ALIAS = os.getenv("MODEL_ALIAS", _mlflow["model_alias"])
