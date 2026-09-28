import pytest
from pydantic import ValidationError

from app.schemas import CustomerFeatures


def test_example_customer_is_valid(customer):
    c = CustomerFeatures(**customer)
    assert "customerID" not in c.features()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenure", -1),
        ("MonthlyCharges", -5),
        ("TotalCharges", -1),
        ("SeniorCitizen", 2),
        ("Contract", "Lifetime"),
        ("gender", "Unknown"),
    ],
)
def test_invalid_values_rejected(customer, field, value):
    customer[field] = value
    with pytest.raises(ValidationError):
        CustomerFeatures(**customer)


def test_extra_fields_rejected(customer):
    customer["Churn"] = "Yes"
    with pytest.raises(ValidationError):
        CustomerFeatures(**customer)


def test_addons_require_internet(customer):
    customer["InternetService"] = "No"
    with pytest.raises(ValidationError, match="No internet service"):
        CustomerFeatures(**customer)


def test_multiple_lines_consistent_with_phone(customer):
    customer["PhoneService"] = "No"
    with pytest.raises(ValidationError, match="No phone service"):
        CustomerFeatures(**customer)


def test_total_charges_may_be_null(customer):
    customer["TotalCharges"] = None
    assert CustomerFeatures(**customer).TotalCharges is None
