"""Deterministic URL Feature Extraction Module for PhishGuard.

Extracts lexical, domain, protocol/security, and keyword features from raw URL strings.
Designed to be robust against empty strings, missing schemes, IP addresses, ports,
special characters, unicode, and extreme URL lengths without network dependencies.
"""

from pathlib import Path
import ipaddress
import re
from urllib.parse import urlparse
import pandas as pd


SUSPICIOUS_KEYWORDS = [
    "login",
    "signin",
    "sign-in",
    "verify",
    "verification",
    "account",
    "secure",
    "security",
    "update",
    "password",
    "banking",
    "bank",
    "confirm",
    "authentication",
    "paypal",
    "credential",
    "wallet",
]


import math
from collections import Counter


def calculate_entropy(s: str) -> float:
    """Calculate Shannon entropy of character distribution in a string."""
    if not s:
        return 0.0
    counts = Counter(s.lower())
    length = len(s)
    return round(-sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values()), 4)


def validate_url_format(url: str) -> tuple:
    """Strictly validate input URL structure before feature extraction/ML model.
    Returns (is_valid: bool, reason: str).
    """
    if url is None or not isinstance(url, str):
        return False, "Input URL must be a non-empty string."
    cleaned = url.strip()
    if not cleaned:
        return False, "URL string cannot be empty or whitespace."
    if len(cleaned) > 16384:
        return False, "URL length exceeds maximum allowable threshold of 16384 characters."

    url_lower = cleaned.lower()

    # Reject scheme-only or malformed authority prefixes
    if "://" in url_lower:
        scheme, remainder = url_lower.split("://", 1)
        if scheme not in ("http", "https"):
            return False, "Invalid URL scheme."
        if not remainder or remainder.startswith("/"):
            return False, "Empty or malformed hostname after scheme."
        parse_target = cleaned
    else:
        parse_target = "http://" + cleaned

    try:
        parsed = urlparse(parse_target)
    except Exception:
        return False, "Malformed URL structure."

    netloc = parsed.netloc
    if not netloc:
        return False, "Empty hostname."

    # Strip userinfo (user:pass@host) if present for netloc validation
    if "@" in netloc:
        netloc = netloc.split("@")[-1]

    # Validate port if present
    if ":" in netloc:
        if netloc.startswith("["):
            end_bracket = netloc.find("]")
            if end_bracket == -1:
                return False, "Malformed IPv6 address."
            hostname = netloc[:end_bracket + 1]
            port_part = netloc[end_bracket + 1:]
            if port_part:
                if not port_part.startswith(":"):
                    return False, "Malformed port separator."
                port_str = port_part[1:]
                if not port_str.isdigit() or not (1 <= int(port_str) <= 65535):
                    return False, "Invalid port number."
        else:
            parts = netloc.split(":")
            if len(parts) > 2:
                return False, "Multiple port separators in netloc."
            hostname = parts[0]
            port_str = parts[1]
            if not port_str.isdigit() or not (1 <= int(port_str) <= 65535):
                return False, "Invalid port number."
    else:
        hostname = netloc

    clean_host = hostname.strip("[]")
    if not clean_host:
        return False, "Empty hostname."

    # Check for leading or trailing dots in hostname
    if clean_host.startswith(".") or clean_host.endswith("."):
        return False, "Hostname cannot start or end with a dot."

    # Check for consecutive dots in hostname
    if ".." in clean_host:
        return False, "Hostname cannot contain consecutive dots."

    # Check if IP address
    is_ip = False
    try:
        ipaddress.ip_address(clean_host)
        is_ip = True
    except ValueError:
        pass

    if not is_ip:
        labels = clean_host.split(".")
        if len(labels) < 2 and clean_host.lower() != "localhost":
            return False, "Hostname must contain at least a domain and TLD."

        for label in labels:
            if not label:
                return False, "Empty domain label."
            if label.startswith("-") or label.endswith("-"):
                return False, f"Domain label cannot start or end with hyphen: '{label}'."
            # Check allowed characters: alnum, hyphen, or non-ASCII (for unicode/IDN)
            if not any(c.isalnum() or c == "-" or ord(c) > 127 for c in label):
                return False, f"Invalid domain label: '{label}'."

        tld = labels[-1]
        # TLD validation: alphabetic, IDN xn--, or non-ASCII
        if not (tld.isalpha() or tld.lower().startswith("xn--") or any(ord(c) > 127 for c in tld)):
            return False, f"Invalid TLD: '{tld}'."

    return True, ""


def extract_features(url: str) -> dict:
    """Extract 38 deterministic numerical features from a given URL string."""
    is_valid, reason = validate_url_format(url)
    if not is_valid:
        raise ValueError(f"INVALID URL: {reason if reason else 'Please enter a valid website URL.'}")

    url_str = url.strip()
    url_lower = url_str.lower()

    # Protocol detection
    uses_https = 1 if url_lower.startswith("https://") else 0
    uses_http = 1 if url_lower.startswith("http://") else 0

    # Parse URL components using urllib.parse
    if uses_https or uses_http or "://" in url_lower:
        parse_target = url_str
    else:
        parse_target = "//" + url_str

    try:
        parsed = urlparse(parse_target)
        netloc = parsed.netloc or ""
        path = parsed.path or ""
        query = parsed.query or ""
    except Exception:
        netloc = ""
        path = ""
        query = ""

    # Split netloc into userinfo, hostname, and port if present
    clean_netloc = netloc.split("@")[-1] if "@" in netloc else netloc
    has_port = 0
    if ":" in clean_netloc:
        parts = clean_netloc.split(":")
        hostname = parts[0]
        # Verify if second part is a valid numeric port
        if len(parts) > 1 and parts[1].isdigit():
            has_port = 1
    else:
        hostname = clean_netloc

    # Remove brackets from IPv6 hostnames if present
    clean_hostname = hostname.strip("[]")

    # IP Address detection
    has_ip_address = 0
    if clean_hostname:
        try:
            ipaddress.ip_address(clean_hostname)
            has_ip_address = 1
        except ValueError:
            # Fallback regex for standard IPv4
            if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", clean_hostname):
                has_ip_address = 1

    # Subdomain count and main domain length calculation
    domain_length = len(clean_hostname)
    number_of_subdomains = 0

    if clean_hostname and not has_ip_address:
        # Strip trailing dot if present
        host_parts = clean_hostname.rstrip(".").split(".")

        # Handle www prefix
        if host_parts and host_parts[0].lower() == "www":
            host_parts = host_parts[1:]

        if len(host_parts) > 2:
            number_of_subdomains = len(host_parts) - 2
        else:
            number_of_subdomains = 0

    # Lexical features
    url_length = len(url_str)
    hostname_length = len(clean_hostname)
    path_length = len(path)
    query_length = len(query)

    number_of_dots = url_str.count(".")
    number_of_hyphens = url_str.count("-")
    number_of_underscores = url_str.count("_")
    number_of_slashes = url_str.count("/")
    number_of_digits = sum(c.isdigit() for c in url_str)
    number_of_letters = sum(c.isalpha() for c in url_str)
    number_of_special_characters = sum(not c.isalnum() for c in url_str)
    number_of_question_marks = url_str.count("?")
    number_of_equals = url_str.count("=")
    number_of_at_symbols = url_str.count("@")
    number_of_percent_symbols = url_str.count("%")

    # Suspicious keyword features
    has_login_keyword = 1 if any(kw in url_lower for kw in ["login", "signin", "sign-in"]) else 0
    has_verify_keyword = 1 if any(kw in url_lower for kw in ["verify", "verification"]) else 0
    has_account_keyword = 1 if "account" in url_lower else 0
    has_secure_keyword = 1 if any(kw in url_lower for kw in ["secure", "security"]) else 0
    has_password_keyword = 1 if "password" in url_lower else 0
    has_update_keyword = 1 if "update" in url_lower else 0
    has_banking_keyword = 1 if any(kw in url_lower for kw in ["banking", "bank"]) else 0

    suspicious_keyword_count = sum(url_lower.count(kw) for kw in SUSPICIOUS_KEYWORDS)

    # --- 9 Defensible URL-String-Only Features ---
    host_len = len(clean_hostname)
    host_letters = sum(c.isalpha() for c in clean_hostname)
    hostname_letter_ratio = round(host_letters / host_len, 4) if host_len > 0 else 0.0

    host_digits = sum(c.isdigit() for c in clean_hostname)
    hostname_digit_ratio = round(host_digits / host_len, 4) if host_len > 0 else 0.0

    host_hyphens = clean_hostname.count("-")
    hostname_hyphen_ratio = round(host_hyphens / host_len, 4) if host_len > 0 else 0.0

    hostname_special_character_count = sum(not c.isalnum() and c != "." for c in clean_hostname)
    hostname_entropy = calculate_entropy(clean_hostname)

    consecutive_digit_count = 0
    curr_d = 0
    for c in url_str:
        if c.isdigit():
            curr_d += 1
            consecutive_digit_count = max(consecutive_digit_count, curr_d)
        else:
            curr_d = 0

    consecutive_hyphen_count = 0
    curr_h = 0
    for c in url_str:
        if c == "-":
            curr_h += 1
            consecutive_hyphen_count = max(consecutive_hyphen_count, curr_h)
        else:
            curr_h = 0

    if clean_hostname and not has_ip_address:
        host_parts_tld = clean_hostname.lower().rstrip(".").split(".")
        domain_token_count = len(host_parts_tld)
        tld_length = len(host_parts_tld[-1]) if host_parts_tld and host_parts_tld[-1] else 0
    else:
        domain_token_count = 0
        tld_length = 0

    return {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "query_length": query_length,
        "number_of_dots": number_of_dots,
        "number_of_hyphens": number_of_hyphens,
        "number_of_underscores": number_of_underscores,
        "number_of_slashes": number_of_slashes,
        "number_of_digits": number_of_digits,
        "number_of_letters": number_of_letters,
        "number_of_special_characters": number_of_special_characters,
        "number_of_question_marks": number_of_question_marks,
        "number_of_equals": number_of_equals,
        "number_of_at_symbols": number_of_at_symbols,
        "number_of_percent_symbols": number_of_percent_symbols,
        "domain_length": domain_length,
        "number_of_subdomains": number_of_subdomains,
        "has_ip_address": has_ip_address,
        "has_port": has_port,
        "uses_https": uses_https,
        "uses_http": uses_http,
        "has_login_keyword": has_login_keyword,
        "has_verify_keyword": has_verify_keyword,
        "has_account_keyword": has_account_keyword,
        "has_secure_keyword": has_secure_keyword,
        "has_password_keyword": has_password_keyword,
        "has_update_keyword": has_update_keyword,
        "has_banking_keyword": has_banking_keyword,
        "suspicious_keyword_count": suspicious_keyword_count,
        "hostname_letter_ratio": hostname_letter_ratio,
        "hostname_digit_ratio": hostname_digit_ratio,
        "hostname_hyphen_ratio": hostname_hyphen_ratio,
        "hostname_special_character_count": hostname_special_character_count,
        "hostname_entropy": hostname_entropy,
        "consecutive_digit_count": consecutive_digit_count,
        "consecutive_hyphen_count": consecutive_hyphen_count,
        "domain_token_count": domain_token_count,
        "tld_length": tld_length,
    }


def generate_feature_dataset(
    cleaned_dataset_path: str = "dataset/processed/cleaned_urls.csv",
    output_feature_path: str = "dataset/processed/features.csv",
) -> pd.DataFrame:
    """Load cleaned URLs, apply extract_features, and save features.csv."""
    in_path = Path(cleaned_dataset_path)
    out_path = Path(output_feature_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {in_path}")

    print(f"Loading cleaned dataset from {in_path}...")
    df = pd.read_csv(in_path)
    print(f"Extracting features for {len(df)} URLs...")

    urls = df["url"].astype(str).tolist()

    # Extract features sequentially for maximum Windows compatibility
    feature_dicts = [extract_features(url) for url in urls]

    features_df = pd.DataFrame(feature_dicts)

    # Combine original URL, features, and label
    final_df = pd.concat([df[["url"]], features_df, df[["label"]]], axis=1)

    print(f"Saving feature dataset to {out_path}...")
    final_df.to_csv(out_path, index=False)
    print(f"Successfully generated {out_path} with shape {final_df.shape}")

    return final_df


if __name__ == "__main__":
    generate_feature_dataset()
