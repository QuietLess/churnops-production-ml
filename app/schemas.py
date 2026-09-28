"""Request/response contracts. The API accepts raw business fields, not encoded columns.

Unknown categories are REJECTED here (422) rather than silently ignored by the encoder;
this is a deliberate, tested choice. Cross-field consistency (e.g. no internet add-ons
without internet service) is also enforced, mirroring the training data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import INTERNET_ADDON_COLS

YesNo = Literal["No", "Yes"]
InternetAddon = Literal["No", "No internet service", "Yes"]

EXAMPLE_CUSTOMER = {
    "customerID": "DEMO-0001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 8,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.55,
    "TotalCharges": 720.40,
}


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_CUSTOMER})

    customerID: str | None = Field(  # noqa: N815 - matches dataset column name
        default=None, max_length=64, description="Optional trace ID; never used as a feature."
    )
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=1000, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"
    ]
    MonthlyCharges: float = Field(ge=0, le=10_000)
    TotalCharges: float | None = Field(
        default=None, ge=0, le=1_000_000, description="May be null for brand-new customers"
    )

    @model_validator(mode="after")
    def check_service_consistency(self) -> CustomerFeatures:
        no_internet = self.InternetService == "No"
        for col in INTERNET_ADDON_COLS:
            value = getattr(self, col)
            if no_internet and value != "No internet service":
                raise ValueError(f"{col} must be 'No internet service' when InternetService='No'")
            if not no_internet and value == "No internet service":
                raise ValueError(f"{col}='No internet service' requires InternetService='No'")
        no_phone = self.PhoneService == "No"
        if no_phone != (self.MultipleLines == "No phone service"):
            raise ValueError("MultipleLines='No phone service' must match PhoneService='No'")
        return self

    def features(self) -> dict:
        return self.model_dump(exclude={"customerID"})


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[CustomerFeatures] = Field(min_length=1)


class PredictionResponse(BaseModel):
    prediction_id: str
    customerID: str | None = None  # noqa: N815
    prediction: Literal[0, 1]
    churn_probability: float = Field(ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    threshold: float
    model_name: str
    model_alias: str | None
    model_version: str
    logged: bool = Field(description="False if the prediction log write failed (non-fatal)")


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class Contribution(BaseModel):
    feature: str
    value: str | float | int | None
    contribution: float = Field(description="Change in churn probability attributed by the model")


class ExplanationResponse(BaseModel):
    churn_probability: float
    base_value: float
    model_version: str
    top_increasing: list[Contribution]
    top_decreasing: list[Contribution]
    note: str = (
        "Contributions describe how the model used each feature for this prediction. "
        "They are not causal effects."
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    database: Literal["ok", "unavailable", "disabled"]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_alias: str | None
    model_version: str
    model_uri: str
    threshold: float
    threshold_strategy: str | None
    model_family: str | None
    feature_version: str | None
    risk_bands: dict
    loaded_at: str
