"""Unit tests for URL feature extraction module (src/feature_extraction.py).

Verifies feature calculations, edge cases, protocol handling, IP address detection,
suspicious keyword detection, unicode support, long URLs, empty inputs, and schema consistency.
"""

import pytest
from src.feature_extraction import extract_features, SUSPICIOUS_KEYWORDS


EXPECTED_FEATURE_KEYS = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "number_of_dots",
    "number_of_hyphens",
    "number_of_underscores",
    "number_of_slashes",
    "number_of_digits",
    "number_of_letters",
    "number_of_special_characters",
    "number_of_question_marks",
    "number_of_equals",
    "number_of_at_symbols",
    "number_of_percent_symbols",
    "domain_length",
    "number_of_subdomains",
    "has_ip_address",
    "has_port",
    "uses_https",
    "uses_http",
    "has_login_keyword",
    "has_verify_keyword",
    "has_account_keyword",
    "has_secure_keyword",
    "has_password_keyword",
    "has_update_keyword",
    "has_banking_keyword",
    "suspicious_keyword_count",
    "hostname_letter_ratio",
    "hostname_digit_ratio",
    "hostname_hyphen_ratio",
    "hostname_special_character_count",
    "hostname_entropy",
    "consecutive_digit_count",
    "consecutive_hyphen_count",
    "domain_token_count",
    "tld_length",
]


def test_feature_schema_consistency():
    """Test that extract_features returns exact expected 38 feature keys."""
    res = extract_features("https://www.example.com")
    assert isinstance(res, dict)
    assert len(res) == 38
    assert list(res.keys()) == EXPECTED_FEATURE_KEYS


def test_normal_https_url():
    """Test feature extraction for standard HTTPS URL."""
    url = "https://www.google.com/search"
    res = extract_features(url)
    assert res["uses_https"] == 1
    assert res["uses_http"] == 0
    assert res["has_ip_address"] == 0
    assert res["has_port"] == 0
    assert res["number_of_dots"] == 2
    assert res["number_of_slashes"] == 3
    assert res["tld_length"] == 3
    assert res["domain_token_count"] == 3


def test_http_url():
    """Test feature extraction for standard HTTP URL."""
    url = "http://example.com/index.html"
    res = extract_features(url)
    assert res["uses_https"] == 0
    assert res["uses_http"] == 1
    assert res["hostname_length"] == 11
    assert res["path_length"] == 11


def test_url_without_scheme():
    """Test feature extraction for URL lacking protocol scheme."""
    url = "sub.domain.co.uk/path/page.html"
    res = extract_features(url)
    assert res["uses_https"] == 0
    assert res["uses_http"] == 0
    assert res["has_ip_address"] == 0
    assert res["number_of_subdomains"] >= 1


def test_ip_based_url():
    """Test feature extraction for IP address URL."""
    url = "http://192.168.1.1/admin"
    res = extract_features(url)
    assert res["has_ip_address"] == 1
    assert res["uses_http"] == 1
    assert res["number_of_subdomains"] == 0
    assert res["consecutive_digit_count"] == 3
    assert res["hostname_digit_ratio"] > 0.5


def test_url_with_port():
    """Test feature extraction for URL specifying explicit port."""
    url = "http://example.com:8080/test"
    res = extract_features(url)
    assert res["has_port"] == 1
    assert res["hostname_length"] == 11


def test_url_with_query_parameters():
    """Test feature extraction for URL containing query parameters."""
    url = "https://example.com/search?q=test&page=1"
    res = extract_features(url)
    assert res["query_length"] == 13
    assert res["number_of_question_marks"] == 1
    assert res["number_of_equals"] == 2


def test_suspicious_keywords_detection():
    """Test detection of suspicious security keywords."""
    url = "http://verify-bank-account-update.com/login"
    res = extract_features(url)
    assert res["has_login_keyword"] == 1
    assert res["has_verify_keyword"] == 1
    assert res["has_account_keyword"] == 1
    assert res["has_update_keyword"] == 1
    assert res["has_banking_keyword"] == 1
    assert res["suspicious_keyword_count"] >= 5


def test_new_features_entropy_and_ratios():
    """Test feature calculations for character ratios, entropy, and consecutive sequences."""
    url = "http://x8k9z2-login-12345.info/verify"
    res = extract_features(url)
    assert res["hostname_entropy"] > 3.0
    assert res["hostname_digit_ratio"] > 0.0
    assert res["hostname_hyphen_ratio"] > 0.0
    assert res["consecutive_digit_count"] == 5
    assert res["consecutive_hyphen_count"] == 1
    assert res["tld_length"] == 4


def test_special_characters_in_url():
    """Test feature extraction with special characters (@, %, _)."""
    url = "http://example.com/@user_name/profile?key=10%20val"
    res = extract_features(url)
    assert res["number_of_at_symbols"] == 1
    assert res["number_of_underscores"] == 1
    assert res["number_of_percent_symbols"] == 1


def test_very_long_url():
    """Test feature extraction on a very long URL (5000+ chars)."""
    url = "https://example.com/" + ("a" * 5000)
    res = extract_features(url)
    assert res["url_length"] == 5020
    assert res["uses_https"] == 1


def test_empty_and_invalid_inputs():
    """Test strict format validation rejecting empty strings, None, and malformed URLs."""
    with pytest.raises(ValueError, match="INVALID URL"):
        extract_features("")

    with pytest.raises(ValueError, match="INVALID URL"):
        extract_features(None)

    for invalid_url in ["http://", "https://", "http://.com", "https://.com", "http://.", "http://..", "http:///example.com"]:
        with pytest.raises(ValueError, match="INVALID URL"):
            extract_features(invalid_url)


def test_unicode_url():
    """Test feature extraction on URL containing unicode characters."""
    url = "http://exámple.com/ñáó"
    res = extract_features(url)
    assert res["uses_http"] == 1
    assert res["url_length"] > 0
    assert len(res) == 38

