"""Pydantic Request and Response Schemas for PhishGuard API.

Defines input validation rules and structured JSON output models compatible with Pydantic V2.
"""

from typing import List
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """URL prediction request schema."""

    url: str = Field(
        ...,
        title="Target Website URL",
        description="Full or raw URL string to evaluate for phishing indicators.",
        json_schema_extra={"example": "http://verify-bank-account-update.com/login"},
        min_length=1,
        max_length=4096,
    )


class PredictionResponse(BaseModel):
    """URL prediction response schema."""

    url: str = Field(..., description="Normalized URL evaluated by the system.")
    prediction: str = Field(
        ...,
        description="Predicted safety category ('legitimate' or 'phishing').",
        json_schema_extra={"example": "phishing"},
    )
    confidence: float = Field(
        ...,
        description="Model predicted probability score for phishing class (range 0.0000 to 1.0000).",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.9421},
    )
    risk_level: str = Field(
        ...,
        description="Application-level risk severity rating ('LOW', 'MEDIUM', 'HIGH').",
        json_schema_extra={"example": "HIGH"},
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="List of supporting URL-level structural, protocol, or keyword risk indicators.",
    )


class HealthResponse(BaseModel):
    """API health check response schema."""

    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model: str = Field(..., json_schema_extra={"example": "xgboost"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    feature_schema_loaded: bool = Field(..., json_schema_extra={"example": True})
