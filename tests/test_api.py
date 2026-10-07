"""Integration tests for FastAPI REST API endpoints (api/main.py).

Tests GET /health and POST /predict endpoints using FastAPI TestClient with lifespan support.
Includes strict URL format rejection API tests for Phase 9.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Fixture providing TestClient inside lifespan context manager."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test GET /health returns operational diagnostics."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"] == "xgboost"
    assert data["model_loaded"] is True
    assert data["feature_schema_loaded"] is True


def test_predict_endpoint_valid_phishing_url(client):
    """Test POST /predict with a valid phishing URL payload."""
    payload = {"url": "http://verify-paypal-account.com/login"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["url"] == "http://verify-paypal-account.com/login"
    assert data["prediction"] == "phishing"
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(data["risk_factors"], list)


def test_predict_endpoint_legitimate_url(client):
    """Test POST /predict with a standard benign URL."""
    payload = {"url": "https://google.com"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["url"] == "https://google.com"
    assert data["prediction"] == "legitimate"


def test_predict_endpoint_legitimate_ip_portal(client):
    """Test POST /predict with legitimate institutional IP portal."""
    payload = {"url": "http://110.172.151.102/LMS/index.html"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "legitimate"


@pytest.mark.parametrize("invalid_url", [
    "http://.com",
    "https://.com",
    "http://",
    "https://",
    "http://.",
    "http://..",
    "http:///example.com",
])
def test_predict_endpoint_invalid_urls(client, invalid_url):
    """Test POST /predict with invalid URL payloads returns 400 Bad Request."""
    payload = {"url": invalid_url}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "INVALID URL" in data["detail"]


def test_predict_endpoint_empty_url(client):
    """Test POST /predict with an empty URL payload returns 400 or 422."""
    payload = {"url": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code in [400, 422]


def test_predict_endpoint_missing_url(client):
    """Test POST /predict with missing url key returns 422 Unprocessable Entity."""
    payload = {}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
