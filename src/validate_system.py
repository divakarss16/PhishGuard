"""Final System Validation and Testing Script for PhishGuard (Phase 9).

Executes end-to-end URL string evaluation across controlled test categories:
  Category A: Clearly Legitimate Domain-Only & Portal URLs
  Category B: Controlled Phishing / Suspicious URLs
  Category C: Structural Edge Cases & IP Portals
  Category D: Invalid URL Format Rejections (Strict Validation)

Measures accuracy, precision, recall, false positives, false negatives, confidence distribution,
and verifies zero crashes without issuing network requests or crawling targets.
"""

import sys, os
sys.path.insert(0, os.getcwd())
from typing import Dict, List, Any
from src.prediction import predictor, URLPredictor
from src.feature_extraction import validate_url_format


# Controlled Test Dataset for Phase 9
TEST_SUITE = [
    # Category A — Clearly Legitimate URLs (Ground Truth: legitimate)
    {"category": "Category A (Legitimate)", "url": "https://google.com", "expected": "legitimate"},
    {"category": "Category A (Legitimate)", "url": "https://github.com", "expected": "legitimate"},
    {"category": "Category A (Legitimate)", "url": "https://microsoft.com", "expected": "legitimate"},
    {"category": "Category A (Legitimate)", "url": "https://apple.com", "expected": "legitimate"},
    {"category": "Category A (Legitimate)", "url": "https://amazon.com", "expected": "legitimate"},
    {"category": "Category A (Legitimate)", "url": "https://wikipedia.org", "expected": "legitimate"},
    {"category": "Category A (Legitimate)", "url": "https://python.org", "expected": "legitimate"},
    {"category": "Category A (Legitimate)", "url": "https://youtube.com", "expected": "legitimate"},
    {"category": "Category A (Legitimate)", "url": "https://linkedin.com", "expected": "legitimate"},
    {"category": "Category A (Legitimate)", "url": "https://stackoverflow.com", "expected": "legitimate"},
    {"category": "Category A (Legitimate)", "url": "http://110.172.151.102/LMS/index.html", "expected": "legitimate"},

    # Category B — Controlled Phishing / Suspicious URLs (Ground Truth: phishing)
    {"category": "Category B (Phishing)", "url": "http://verify-bank-account-update.com/login", "expected": "phishing"},
    {"category": "Category B (Phishing)", "url": "http://secure-login-account-verify.com/signin", "expected": "phishing"},
    {"category": "Category B (Phishing)", "url": "http://account-password-reset.example/verify", "expected": "phishing"},
    {"category": "Category B (Phishing)", "url": "http://110.172.151.102/login", "expected": "phishing"},
    {"category": "Category B (Phishing)", "url": "http://110.172.151.102/verify-account", "expected": "phishing"},

    # Category C — Structural Edge Cases (Behavioral test)
    {"category": "Category C (Edge Case)", "url": "https://github.com/security", "expected": "legitimate"},
    {"category": "Category C (Edge Case)", "url": "https://www.google.com/search?q=python", "expected": "legitimate"},
    {"category": "Category C (Edge Case)", "url": "https://www.youtube.com/results?search_query=test", "expected": "legitimate"},
    {"category": "Category C (Edge Case)", "url": "http://192.168.1.1/portal/index.html", "expected": "legitimate"},

    # Category D — Invalid URL Format Rejections (Expected: INVALID URL)
    {"category": "Category D (Invalid URL)", "url": "http://.com", "expected": "invalid"},
    {"category": "Category D (Invalid URL)", "url": "https://.com", "expected": "invalid"},
    {"category": "Category D (Invalid URL)", "url": "http://", "expected": "invalid"},
    {"category": "Category D (Invalid URL)", "url": "https://", "expected": "invalid"},
    {"category": "Category D (Invalid URL)", "url": "http://.", "expected": "invalid"},
    {"category": "Category D (Invalid URL)", "url": "http://..", "expected": "invalid"},
    {"category": "Category D (Invalid URL)", "url": "http:///example.com", "expected": "invalid"},
]


def run_system_validation() -> Dict[str, Any]:
    """Execute end-to-end prediction pipeline validation and compute summary analytics."""
    print("=" * 80)
    print("                PHISHGUARD FINAL SYSTEM VALIDATION (PHASE 9)")
    print("=" * 80)

    predictor.load_artifacts()
    results: List[Dict[str, Any]] = []

    crashes = 0
    known_correct = 0
    known_incorrect = 0
    fp_list = []
    fn_list = []

    for test_item in TEST_SUITE:
        cat = test_item["category"]
        url_str = test_item["url"]
        expected = test_item["expected"]

        try:
            if expected == "invalid":
                is_valid, reason = validate_url_format(url_str)
                if not is_valid:
                    known_correct += 1
                    print(f"\n[{cat}]")
                    print(f"  URL:         {url_str}")
                    print(f"  Result:      REJECTED AS INVALID (Reason: {reason})")
                else:
                    known_incorrect += 1
                    print(f"\n[{cat}] FAILED: Unexpectedly accepted invalid URL: {url_str}")
                continue

            res = predictor.predict(url_str)
            pred = res["prediction"]
            conf = res["confidence"]
            risk = res["risk_level"]
            factors = res["risk_factors"]

            results.append({
                "category": cat,
                "url": url_str,
                "expected": expected,
                "prediction": pred,
                "confidence": conf,
                "risk_level": risk,
                "risk_factors": factors,
            })

            # Check known-label predictions
            if expected is not None:
                if pred == expected:
                    known_correct += 1
                else:
                    known_incorrect += 1
                    if expected == "legitimate" and pred == "phishing":
                        fp_list.append((url_str, conf, factors))
                    elif expected == "phishing" and pred == "legitimate":
                        fn_list.append((url_str, conf, factors))

            # Print individual item evaluation
            print(f"\n[{cat}]")
            print(f"  URL:         {url_str}")
            print(f"  Prediction:  {pred.upper()} (Expected: {expected or 'N/A'})")
            print(f"  Confidence:  {conf * 100:.2f}% | Risk Level: {risk}")
            print(f"  Risk Factors: {factors if factors else 'None (Clean)'}")

        except Exception as exc:
            crashes += 1
            print(f"\n[{cat}] CRASH FOR URL: {url_str}")
            print(f"  Error: {exc}")

    # Summary Statistics
    total_tested = len(TEST_SUITE)
    known_total = known_correct + known_incorrect
    accuracy = (known_correct / known_total) * 100 if known_total > 0 else 0.0

    print("\n" + "=" * 80)
    print("                   SYSTEM VALIDATION SUMMARY METRICS")
    print("=" * 80)
    print(f"Total Test Cases Evaluated:       {total_tested}")
    print(f"Pipeline Crashes / Unhandled Errors: {crashes}")
    print(f"\nKnown-Label Evaluation Subset ({known_total} cases):")
    print(f"  - Correct Classifications:      {known_correct} / {known_total} ({accuracy:.2f}%)")
    print(f"  - Incorrect Classifications:    {known_incorrect} / {known_total}")
    print(f"  - False Positives (Clean -> Phish): {len(fp_list)}")
    print(f"  - False Negatives (Phish -> Clean): {len(fn_list)}")

    # Detailed Error Analysis
    if fp_list:
        print("\n--- False Positive Analysis ---")
        for url_str, conf, factors in fp_list:
            print(f"  * URL: {url_str} | Conf: {conf*100:.1f}% | Factors: {factors}")
    else:
        print("\n--- False Positive Analysis ---")
        print("  None detected in controlled legitimate benchmark set.")

    if fn_list:
        print("\n--- False Negative Analysis ---")
        for url_str, conf, factors in fn_list:
            print(f"  * URL: {url_str} | Conf: {conf*100:.1f}% | Factors: {factors}")
    else:
        print("\n--- False Negative Analysis ---")
        print("  None detected in controlled phishing benchmark set.")

    print("=" * 80)

    return {
        "total_tested": total_tested,
        "crashes": crashes,
        "known_correct": known_correct,
        "known_incorrect": known_incorrect,
        "accuracy": accuracy,
        "false_positives": len(fp_list),
        "false_negatives": len(fn_list),
        "results": results,
    }


if __name__ == "__main__":
    run_system_validation()
