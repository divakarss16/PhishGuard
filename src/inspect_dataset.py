"""Dataset Inspection Script for PhishGuard.

Performs loading, structural validation, duplicate analysis, label checking,
and data quality reporting on raw URL dataset.
"""

from pathlib import Path
import pandas as pd


def inspect_dataset(raw_dataset_path: str = "dataset/raw/phishing_urls.csv") -> dict:
    """Load and perform quality inspection on raw URL dataset."""
    path = Path(raw_dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at expected path: {path}")

    # 1. Load raw dataset
    df = pd.read_csv(path)

    # 2. Shape, columns, data types
    rows, cols = df.shape
    columns = list(df.columns)
    dtypes = df.dtypes.to_dict()

    # 3. Sample rows
    head_5 = df.head(5)

    # 4. Missing values
    missing_counts = df.isnull().sum().to_dict()
    missing_url_count = int(missing_counts.get("url", 0))
    missing_label_count = int(missing_counts.get("label", 0))

    # 5. Duplicate analysis
    exact_duplicates = int(df.duplicated().sum())

    if "url" in df.columns:
        url_series = df["url"].astype(str)
        unique_url_count = int(url_series.nunique())
        duplicate_url_count = int(rows - unique_url_count)

        # Normalized duplicates (trimmed + lowercase)
        normalized_urls = url_series.str.strip().str.lower()
        normalized_unique_urls = int(normalized_urls.nunique())
        normalized_duplicate_count = int(rows - normalized_unique_urls)

        # Invalid URL checks (empty strings or non-string types)
        empty_url_count = int((url_series.str.strip() == "").sum())
        non_string_url_count = int((~df["url"].apply(lambda x: isinstance(x, str))).sum())
        invalid_url_count = empty_url_count + non_string_url_count
    else:
        unique_url_count = 0
        duplicate_url_count = 0
        normalized_duplicate_count = 0
        invalid_url_count = 0

    # 6. Label validation & Class distribution
    if "label" in df.columns:
        unique_labels = [str(lbl) for lbl in df["label"].unique()]
        class_counts = {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}
        class_percentages = {
            str(k): round(float(v) * 100, 2)
            for k, v in df["label"].value_counts(normalize=True).to_dict().items()
        }
    else:
        unique_labels = []
        class_counts = {}
        class_percentages = {}

    legitimate_samples = class_counts.get("good", class_counts.get("legitimate", class_counts.get("0", 0)))
    phishing_samples = class_counts.get("bad", class_counts.get("phishing", class_counts.get("1", 0)))

    # Output formatted report
    print("=" * 60)
    print("           PHISHGUARD DATASET INSPECTION REPORT")
    print("=" * 60)
    print(f"Dataset File Path:           {path}")
    print(f"Dataset Shape:               Rows: {rows} | Columns: {cols}")
    print(f"Column Names:                {columns}")
    print("\nData Types:")
    for col, dt in dtypes.items():
        print(f"  - {col}: {dt}")

    print("\nFirst 5 Rows:")
    print(head_5)

    print("\nMissing Values:")
    for col, count in missing_counts.items():
        print(f"  - {col}: {count}")

    print("\nDuplicate Analysis:")
    print(f"  - Exact Duplicate Rows:     {exact_duplicates}")
    print(f"  - Unique URL Strings:       {unique_url_count}")
    print(f"  - Duplicate URL Count:      {duplicate_url_count}")
    print(f"  - Normalized Duplicate URLs: {normalized_duplicate_count}")

    print("\nURL Validation:")
    print(f"  - Missing URLs:             {missing_url_count}")
    print(f"  - Invalid / Empty URLs:     {invalid_url_count}")

    print("\nLabel Representation & Distribution:")
    print(f"  - Raw Unique Labels Found:  {unique_labels}")
    print("  - Class Counts:")
    for label_val, count in class_counts.items():
        print(f"      * {label_val}: {count}")
    print("  - Class Percentages:")
    for label_val, pct in class_percentages.items():
        print(f"      * {label_val}: {pct}%")

    print("=" * 60)
    print("                     SUMMARY")
    print("=" * 60)
    print(f"Total Records:               {rows}")
    print(f"Total Columns:               {cols}")
    print(f"URL Column Name:             url")
    print(f"Label Column Name:           label")
    print(f"Missing URL Count:           {missing_url_count}")
    print(f"Duplicate URL Count:         {duplicate_url_count}")
    print(f"Invalid URL Count:           {invalid_url_count}")
    print(f"Legitimate ('good') Samples: {legitimate_samples} ({class_percentages.get('good', 0)}%)")
    print(f"Phishing ('bad') Samples:    {phishing_samples} ({class_percentages.get('bad', 0)}%)")
    print("=" * 60)

    return {
        "rows": rows,
        "cols": cols,
        "columns": columns,
        "missing_url_count": missing_url_count,
        "duplicate_url_count": duplicate_url_count,
        "invalid_url_count": invalid_url_count,
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "legitimate_samples": legitimate_samples,
        "phishing_samples": phishing_samples,
    }


if __name__ == "__main__":
    inspect_dataset()
