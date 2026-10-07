"""Phase 9 Comprehensive Benchmark & Regression Test Suite for PhishGuard.

Tests production prediction pipeline against:
- Invalid URLs (Strict Validation, Part A & Part I)
- Legitimate URLs across 6 categories (Part C & Part J)
- Phishing URLs across 12 categories (Part D & Part K)
- IP Address handling (Part G)
- Security checks (Part L)
"""

import pytest
from src.prediction import predictor, get_prediction
from src.feature_extraction import validate_url_format


# ==============================================================================
# PART A & PART I — INVALID URL REGRESSION BENCHMARK
# ==============================================================================
INVALID_URLS_BENCHMARK = [
    "http://",
    "https://",
    "http://.com",
    "https://.com",
    "http://.",
    "http://..",
    "http:///example.com",
    "http://example.com:abc",
    "http://example.com:999999",
    "http://.example.com",
    "http://example..com",
    "http://example.com.",
]


# ==============================================================================
# PART C & PART J — DIVERSE LEGITIMATE BENCHMARK
# ==============================================================================
LEGITIMATE_DOMAIN_ONLY = [
    "https://google.com",
    "https://github.com",
    "https://microsoft.com",
    "https://apple.com",
    "https://amazon.com",
    "https://wikipedia.org",
    "https://python.org",
    "https://youtube.com",
    "https://linkedin.com",
    "https://stackoverflow.com",
]

LEGITIMATE_HTTPS_PATHS = [
    "https://github.com/login",
    "https://www.microsoft.com/en-us",
    "https://www.python.org/downloads/",
    "https://en.wikipedia.org/wiki/Computer",
]

LEGITIMATE_QUERY_URLS = [
    "https://www.google.com/search?q=python",
    "https://www.youtube.com/results?search_query=test",
]

LEGITIMATE_SUBDOMAINS = [
    "https://docs.python.org",
    "https://developer.mozilla.org",
    "https://aws.amazon.com",
]

LEGITIMATE_HTTP_URLS = [
    "http://example.com",
    "http://gnu.org",
    "http://w3.org",
]

LEGITIMATE_IP_URLS = [
    "http://110.172.151.102/LMS/index.html",
    "http://192.168.1.1/portal/index.html",
    "http://10.0.0.1/dashboard",
    "http://172.16.0.1/index.html",
]


# ==============================================================================
# PART D & PART K — DIVERSE PHISHING BENCHMARK
# ==============================================================================
PHISHING_BENCHMARK = [
    "http://verify-bank-account-update.com/login",
    "http://secure-login-account-verify.com/signin",
    "http://account-password-reset.example/verify",
    "http://paypal-verification-security.com/login",
    "https://account-update-verification-bank.com/signin",
    "http://110.172.151.102/login",
    "http://192.168.1.1/verify-account",
    "http://10.0.0.1/update-password",
    "http://login-security-check.example/account/update",
    "http://my-secure-bank-account-verify-login-update.example",
    "http://bank-login-99887766554433.example/login",
    "http://login.account.security.verify.bank.example.com/login",
    "http://login-verify.example/signin?account=123&verify=true&session=xyz",
    "http://admin:secret@phish-bank-login.example/login",
    "http://very-long-phishing-domain-name-security-update-verify-account-bank-login-portal-secure-identity-confirm-access.example/login/verify/account/update/index.html",
]


# ==============================================================================
# TEST IMPLEMENTATION
# ==============================================================================

@pytest.mark.parametrize("url", INVALID_URLS_BENCHMARK)
def test_invalid_urls_rejected_before_xgboost(url):
    """Verify malformed URLs fail strict validation and never reach ML model."""
    is_valid, reason = validate_url_format(url)
    assert not is_valid, f"Url validation allowed invalid URL: {url}"

    with pytest.raises(ValueError) as exc_info:
        get_prediction(url)
    assert "INVALID URL" in str(exc_info.value)


@pytest.mark.parametrize("url", LEGITIMATE_DOMAIN_ONLY)
def test_legitimate_domain_only_benchmark(url):
    """Verify domain-only legitimate benchmark URLs pass as legitimate."""
    res = get_prediction(url)
    assert res["prediction"] == "legitimate", f"False positive on domain-only URL: {url}"


@pytest.mark.parametrize("url", LEGITIMATE_HTTPS_PATHS)
def test_legitimate_https_paths_benchmark(url):
    """Verify HTTPS legitimate URLs with paths pass as legitimate."""
    res = get_prediction(url)
    assert res["prediction"] == "legitimate", f"False positive on HTTPS path URL: {url}"


@pytest.mark.parametrize("url", LEGITIMATE_QUERY_URLS)
def test_legitimate_query_benchmark(url):
    """Verify legitimate query URLs pass as legitimate."""
    res = get_prediction(url)
    assert res["prediction"] == "legitimate", f"False positive on query URL: {url}"


@pytest.mark.parametrize("url", LEGITIMATE_SUBDOMAINS)
def test_legitimate_subdomains_benchmark(url):
    """Verify legitimate subdomains pass as legitimate."""
    res = get_prediction(url)
    assert res["prediction"] == "legitimate", f"False positive on subdomain URL: {url}"


@pytest.mark.parametrize("url", LEGITIMATE_HTTP_URLS)
def test_legitimate_http_benchmark(url):
    """Verify legitimate HTTP URLs pass as legitimate."""
    res = get_prediction(url)
    assert res["prediction"] == "legitimate", f"False positive on HTTP URL: {url}"


@pytest.mark.parametrize("url", LEGITIMATE_IP_URLS)
def test_legitimate_ip_benchmark(url):
    """Verify legitimate institutional/portal IP URLs pass as legitimate."""
    res = get_prediction(url)
    assert res["prediction"] == "legitimate", f"False positive on IP URL: {url}"


@pytest.mark.parametrize("url", PHISHING_BENCHMARK)
def test_phishing_benchmark(url):
    """Verify phishing benchmark URLs are correctly detected as phishing."""
    res = get_prediction(url)
    assert res["prediction"] == "phishing", f"False negative on phishing URL: {url}"
    assert res["risk_level"] in ["MEDIUM", "HIGH"]
