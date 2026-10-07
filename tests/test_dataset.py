"""Unit tests for dataset loading, inspection, and verification."""

from pathlib import Path
import pytest
import pandas as pd


@pytest.fixture
def dataset_path():
    """Returns path to raw dataset file."""
    return Path("dataset/raw/phishing_urls.csv")


def test_dataset_file_exists(dataset_path):
    """Test that raw dataset file exists in dataset/raw/."""
    assert dataset_path.exists(), f"Raw dataset file not found at {dataset_path}"


def test_dataset_loading(dataset_path):
    """Test that dataset can be loaded by Pandas without errors."""
    df = pd.read_csv(dataset_path)
    assert df is not None
    assert isinstance(df, pd.DataFrame)


def test_dataset_not_empty(dataset_path):
    """Test that dataset is not completely empty."""
    df = pd.read_csv(dataset_path)
    assert len(df) > 0, "Dataset contains no records"
    assert df.shape[1] >= 2, "Dataset should have at least 2 columns"


def test_url_column_exists(dataset_path):
    """Test that 'url' column exists in dataset."""
    df = pd.read_csv(dataset_path)
    assert "url" in df.columns, f"Expected 'url' in columns, found: {list(df.columns)}"


def test_label_column_exists(dataset_path):
    """Test that 'label' column exists in dataset."""
    df = pd.read_csv(dataset_path)
    assert "label" in df.columns, f"Expected 'label' in columns, found: {list(df.columns)}"


def test_label_values(dataset_path):
    """Test that label contains valid class representations ('good' and 'bad')."""
    df = pd.read_csv(dataset_path)
    unique_labels = set(df["label"].unique())
    assert {"good", "bad"}.issubset(unique_labels), f"Unexpected labels: {unique_labels}"
