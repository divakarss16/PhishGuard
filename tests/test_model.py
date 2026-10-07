"""Unit tests for ML Model Training, Evaluation, and Serialization (Phase 4).

Verifies feature matrix shape, label encoding, train/test split integrity,
model fitting, prediction validity, artifact serialization, and feature schema alignment.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.feature_extraction import extract_features


@pytest.fixture
def feature_df():
    """Fixture to load dataset/processed/features.csv."""
    path = Path("dataset/processed/features.csv")
    assert path.exists(), f"Feature dataset file missing at {path}"
    return pd.read_csv(path)


def test_training_dataset_loads(feature_df):
    """Test that feature dataset loads and is non-empty."""
    assert len(feature_df) > 0, "Feature dataset contains no rows!"
    assert "url" in feature_df.columns, "Expected 'url' column for traceability"
    assert "label" in feature_df.columns, "Expected 'label' target column"


def test_feature_matrix_columns(feature_df):
    """Test that feature matrix X excludes 'url' and 'label' and has 38 features."""
    feature_cols = [c for c in feature_df.columns if c not in ["url", "label"]]
    assert len(feature_cols) == 38, f"Expected 38 numerical features, got {len(feature_cols)}"
    assert "url" not in feature_cols, "'url' column leaked into feature matrix X!"
    assert "label" not in feature_cols, "'label' column leaked into feature matrix X!"


def test_label_classes(feature_df):
    """Test that target labels contain only legitimate and phishing."""
    unique_labels = set(feature_df["label"].unique())
    assert unique_labels == {"legitimate", "phishing"}, f"Unexpected labels: {unique_labels}"


def test_train_test_split_non_empty(feature_df):
    """Test that train/test split yields non-empty stratified subsets."""
    feature_cols = [c for c in feature_df.columns if c not in ["url", "label"]]
    X = feature_df[feature_cols]
    y = feature_df["label"].map({"legitimate": 0, "phishing": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    assert len(X_train) > 0 and len(X_test) > 0
    assert len(X_train) + len(X_test) == len(feature_df)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_model_training_and_prediction():
    """Test fitting a classifier on a sample and making valid predictions."""
    X_sample = np.random.rand(50, 38)
    y_sample = np.random.choice([0, 1], size=50)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_sample, y_sample)

    preds = clf.predict(X_sample[:5])
    probs = clf.predict_proba(X_sample[:5])

    assert len(preds) == 5
    assert set(preds).issubset({0, 1})
    assert probs.shape == (5, 2)


def test_saved_model_artifact_loading():
    """Test that models/phishing_model.pkl exists and can be loaded with joblib."""
    model_path = Path("models/phishing_model.pkl")
    assert model_path.exists(), f"Serialized model file missing at {model_path}"

    loaded_model = joblib.load(model_path)
    assert loaded_model is not None
    assert hasattr(loaded_model, "predict")


def test_loaded_model_prediction():
    """Test that loaded model can predict on a sample feature vector from extract_features."""
    model_path = Path("models/phishing_model.pkl")
    feature_names_path = Path("models/feature_names.pkl")

    loaded_model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)

    raw_features = extract_features("http://verify-bank-account.com/login")
    sample_df = pd.DataFrame([raw_features])[feature_names]

    pred = loaded_model.predict(sample_df)
    prob = loaded_model.predict_proba(sample_df)[:, 1]

    assert len(pred) == 1
    assert pred[0] in [0, 1]
    assert 0.0 <= prob[0] <= 1.0


def test_prediction_feature_schema_matches():
    """Test that prediction feature schema matches trained feature names order."""
    feature_names_path = Path("models/feature_names.pkl")
    feature_names = joblib.load(feature_names_path)

    sample_features = extract_features("https://google.com")
    assert list(sample_features.keys()) == feature_names
