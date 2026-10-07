"""Production Prediction Pipeline for PhishGuard.

Loads pre-trained XGBoost model and feature schema to predict URL risk, confidence scores,
risk levels, and evidence-based risk factors without network dependencies or retraining.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import joblib
import pandas as pd
from src.feature_extraction import extract_features, validate_url_format, SUSPICIOUS_KEYWORDS

# Centralized Risk Threshold Configuration
RISK_THRESHOLDS = {
    "LOW_MAX": 0.40,
    "MEDIUM_MAX": 0.70,
}

LABEL_MAP = {0: "legitimate", 1: "phishing"}


class URLPredictor:
    """Production prediction service for PhishGuard."""

    def __init__(
        self,
        model_path: str = "models/phishing_model.pkl",
        feature_names_path: str = "models/feature_names.pkl",
    ):
        self.model_path = Path(model_path)
        self.feature_names_path = Path(feature_names_path)
        self.model: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None
        self.is_loaded: bool = False

    def load_artifacts(self) -> None:
        """Load trained model and feature schema into memory."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model artifact missing at {self.model_path}")
        if not self.feature_names_path.exists():
            raise FileNotFoundError(f"Feature schema artifact missing at {self.feature_names_path}")

        self.model = joblib.load(self.model_path)
        self.feature_names = joblib.load(self.feature_names_path)
        self.is_loaded = True

    def validate_url(self, url: str) -> str:
        """Validate input URL string using strict URL format rules."""
        is_valid, reason = validate_url_format(url)
        if not is_valid:
            raise ValueError("INVALID URL: Please enter a valid website URL.")
        return url.strip()

    def compute_risk_level(self, confidence: float) -> str:
        """Determine application risk level based on model predicted probability score."""
        if confidence < RISK_THRESHOLDS["LOW_MAX"]:
            return "LOW"
        elif confidence < RISK_THRESHOLDS["MEDIUM_MAX"]:
            return "MEDIUM"
        else:
            return "HIGH"

    def extract_risk_factors(self, features: dict, url_str: str) -> List[str]:
        """Extract supporting URL-level risk indicators from feature dictionary."""
        risk_factors: List[str] = []
        url_lower = url_str.lower()

        # Protocol factors
        if features.get("uses_http", 0) == 1:
            risk_factors.append("URL uses insecure HTTP protocol instead of HTTPS")
        elif features.get("uses_https", 0) == 0:
            risk_factors.append("URL lacks an explicit protocol scheme")

        # Domain & Host factors
        if features.get("has_ip_address", 0) == 1:
            risk_factors.append("Raw IP address used instead of a domain name")
        if features.get("has_port", 0) == 1:
            risk_factors.append("Non-standard port number detected in URL")
        if features.get("number_of_subdomains", 0) > 1:
            risk_factors.append(f"Multiple subdomains detected ({features['number_of_subdomains']} subdomains)")

        # Lexical & Structural factors
        if features.get("url_length", 0) > 75:
            risk_factors.append(f"Unusually long URL length ({features['url_length']} characters)")
        if features.get("number_of_dots", 0) > 3:
            risk_factors.append(f"Excessive dot count detected ({features['number_of_dots']} dots)")
        if features.get("number_of_hyphens", 0) > 2:
            risk_factors.append(f"Multiple hyphens detected ({features['number_of_hyphens']} hyphens)")
        if features.get("number_of_at_symbols", 0) > 0:
            risk_factors.append("@ symbol detected in URL (potential credential/redirect trick)")
        if features.get("number_of_special_characters", 0) > 10:
            risk_factors.append(f"High special character count ({features['number_of_special_characters']} characters)")

        # Hostname structural & entropy factors
        if features.get("hostname_entropy", 0.0) > 4.2:
            risk_factors.append(f"High hostname character entropy detected ({features['hostname_entropy']:.2f})")
        if features.get("hostname_digit_ratio", 0.0) > 0.2:
            risk_factors.append(f"High digit ratio in hostname ({features['hostname_digit_ratio']*100:.1f}%)")
        if features.get("hostname_hyphen_ratio", 0.0) > 0.15:
            risk_factors.append(f"High hyphen ratio in hostname ({features['hostname_hyphen_ratio']*100:.1f}%)")
        if features.get("consecutive_digit_count", 0) > 4:
            risk_factors.append(f"Long sequence of consecutive digits detected ({features['consecutive_digit_count']} digits)")

        # Keyword factors
        detected_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url_lower]
        for kw in detected_keywords:
            risk_factors.append(f"Suspicious security keyword detected: '{kw}'")

        return risk_factors

    def predict(self, url: str) -> Dict[str, Any]:
        """Perform URL feature extraction, model prediction, risk scoring, and evidence breakdown."""
        if not self.is_loaded:
            self.load_artifacts()

        clean_url = self.validate_url(url)

        # 1. Extract 29 features using identical feature extractor
        raw_features = extract_features(clean_url)

        # 2. Reorder features to match exact trained feature schema
        feature_df = pd.DataFrame([raw_features])[self.feature_names]

        # 3. Model inference
        pred_class = int(self.model.predict(feature_df)[0])
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feature_df)[0]
            confidence = float(probs[1])  # Phishing probability score
        else:
            confidence = 1.0 if pred_class == 1 else 0.0

        # 4. Map class and risk level
        prediction_label = LABEL_MAP.get(pred_class, "phishing" if pred_class == 1 else "legitimate")
        risk_level = self.compute_risk_level(confidence)

        # 5. Extract evidence-based risk factors
        risk_factors = self.extract_risk_factors(raw_features, clean_url)

        return {
            "url": clean_url,
            "prediction": prediction_label,
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
        }


# Singleton instance for application reuse
predictor = URLPredictor()


def get_prediction(url: str) -> Dict[str, Any]:
    """Helper function to get prediction from default predictor instance."""
    return predictor.predict(url)
