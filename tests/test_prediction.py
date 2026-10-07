"""Unit tests for production prediction service (src/prediction.py).

Tests URL input validation, model inference, confidence range, risk level mapping,
evidence-based risk factor generation, and schema alignment.
"""

import pytest
from src.prediction import URLPredictor, predictor


from tests.test_features import EXPECTED_FEATURE_KEYS


@pytest.fixture
def loaded_predictor():
    """Fixture to ensure predictor artifacts are loaded."""
    p = URLPredictor()
    p.load_artifacts()
    return p


def test_predictor_artifacts_loading(loaded_predictor):
    """Test that model and feature schema artifacts load correctly."""
    assert loaded_predictor.is_loaded is True
    assert loaded_predictor.model is not None
    assert len(loaded_predictor.feature_names) == 38


def test_predict_valid_https_url(loaded_predictor):
    """Test prediction on standard HTTPS URL."""
    res = loaded_predictor.predict("https://www.google.com")
    assert isinstance(res, dict)
    assert res["url"] == "https://www.google.com"
    assert res["prediction"] in ["legitimate", "phishing"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(res["risk_factors"], list)


def test_predict_http_url(loaded_predictor):
    """Test prediction on HTTP URL."""
    res = loaded_predictor.predict("http://example.com/index.html")
    assert res["prediction"] in ["legitimate", "phishing"]
    assert "URL uses insecure HTTP protocol instead of HTTPS" in res["risk_factors"]


def test_predict_url_without_scheme(loaded_predictor):
    """Test prediction on URL lacking scheme prefix."""
    res = loaded_predictor.predict("sub.domain.co.uk/path")
    assert res["prediction"] in ["legitimate", "phishing"]
    assert "URL lacks an explicit protocol scheme" in res["risk_factors"]


def test_predict_ip_based_url(loaded_predictor):
    """Test prediction on IP-based URL."""
    res = loaded_predictor.predict("http://192.168.1.1:8080/admin")
    assert res["prediction"] in ["legitimate", "phishing"]
    assert any("IP address used instead of a domain name" in rf for rf in res["risk_factors"])
    assert any("Non-standard port number detected" in rf for rf in res["risk_factors"])


def test_predict_suspicious_url(loaded_predictor):
    """Test prediction on highly suspicious phishing-like URL."""
    res = loaded_predictor.predict("http://verify-bank-account-update.com/login")
    assert res["prediction"] == "phishing"
    assert res["confidence"] >= 0.70
    assert res["risk_level"] == "HIGH"
    assert len(res["risk_factors"]) >= 3


def test_predict_very_long_url(loaded_predictor):
    """Test prediction on long URL (> 75 chars)."""
    url = "https://example.com/" + ("a" * 100)
    res = loaded_predictor.predict(url)
    assert any("Unusually long URL length" in rf for rf in res["risk_factors"])


def test_predict_empty_url(loaded_predictor):
    """Test that empty URL raises ValueError."""
    with pytest.raises(ValueError, match="INVALID URL"):
        loaded_predictor.predict("")


def test_predict_none_input(loaded_predictor):
    """Test that None input raises ValueError."""
    with pytest.raises(ValueError, match="INVALID URL"):
        loaded_predictor.predict(None)


def test_schema_alignment(loaded_predictor):
    """Test that feature names in predictor match training feature order."""
    assert loaded_predictor.feature_names == EXPECTED_FEATURE_KEYS
