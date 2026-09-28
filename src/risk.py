"""Risk band labelling shared by the API and the dashboard.

Bands are UI labels only (documented project assumptions); the class decision
uses the model's cost-based threshold, which is a separate concept.
"""

from __future__ import annotations

from src.config import RISK_HIGH_LOWER, RISK_LOW_UPPER


def risk_level(probability: float) -> str:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be within [0, 1], got {probability}")
    if probability < RISK_LOW_UPPER:
        return "low"
    if probability > RISK_HIGH_LOWER:
        return "high"
    return "medium"
