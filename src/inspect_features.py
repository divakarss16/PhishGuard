"""Feature Quality Report Script for PhishGuard.

Loads dataset/processed/features.csv and inspects feature shapes, data types,
missing values, infinite values, constant/zero features, minimum/maximum/mean statistics,
and final class distributions.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def inspect_features(feature_dataset_path: str = "dataset/processed/features.csv") -> dict:
    """Load features.csv and compute comprehensive quality report."""
    path = Path(feature_dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found at {path}")

    print(f"Loading feature dataset from {path}...")
    df = pd.read_csv(path)

    rows, cols = df.shape
    url_col = "url" if "url" in df.columns else None
    label_col = "label" if "label" in df.columns else None

    # Identify numerical feature columns (excluding 'url' and 'label')
    feature_cols = [c for c in df.columns if c not in ["url", "label"]]
    num_features = len(feature_cols)

    # Missing & Infinite value checks
    missing_counts = df[feature_cols].isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())

    inf_counts = {}
    for col in feature_cols:
        inf_counts[col] = int(np.isinf(df[col]).sum())
    total_inf = sum(inf_counts.values())

    # Constant & All-Zero feature checks
    constant_features = []
    all_zero_features = []
    stats = {}

    for col in feature_cols:
        unique_vals = df[col].nunique()
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())

        stats[col] = {
            "dtype": str(df[col].dtype),
            "unique_count": int(unique_vals),
            "min": min_val,
            "max": max_val,
            "mean": round(mean_val, 4),
        }

        if unique_vals <= 1:
            constant_features.append(col)
        if min_val == 0 and max_val == 0:
            all_zero_features.append(col)

    # Class distribution
    if label_col:
        class_counts = df[label_col].value_counts().to_dict()
        class_percentages = {
            k: round(v / rows * 100, 2) for k, v in class_counts.items()
        }
    else:
        class_counts = {}
        class_percentages = {}

    # Print Formatted Feature Report
    print("=" * 70)
    print("             PHISHGUARD FEATURE QUALITY REPORT")
    print("=" * 70)
    print(f"Feature Dataset Path:         {path}")
    print(f"Total Records (Rows):         {rows}")
    print(f"Total Dataset Columns:        {cols}")
    print(f"Numerical Feature Count:      {num_features}")
    print(f"Traceability Column:          {url_col}")
    print(f"Target Label Column:          {label_col}")
    print(f"Total Missing Feature Values: {total_missing}")
    print(f"Total Infinite Values:        {total_inf}")
    print(f"Constant Features (1 val):    {len(constant_features)} {constant_features}")
    print(f"All-Zero Features:            {len(all_zero_features)} {all_zero_features}")

    print("\nFeature Summary Statistics:")
    print(f"{'Feature Name':<32} | {'Type':<8} | {'Min':<8} | {'Max':<10} | {'Mean':<10}")
    print("-" * 75)
    for col in feature_cols:
        s = stats[col]
        print(f"{col:<32} | {s['dtype']:<8} | {s['min']:<8.2f} | {s['max']:<10.2f} | {s['mean']:<10.4f}")

    print("\nClass Distribution:")
    for lbl, count in class_counts.items():
        print(f"  - {lbl}: {count} ({class_percentages.get(lbl, 0)}%)")

    print("=" * 70)

    return {
        "rows": rows,
        "cols": cols,
        "num_features": num_features,
        "feature_names": feature_cols,
        "total_missing": total_missing,
        "total_inf": total_inf,
        "constant_features": constant_features,
        "all_zero_features": all_zero_features,
        "stats": stats,
        "class_counts": class_counts,
        "class_percentages": class_percentages,
    }


if __name__ == "__main__":
    inspect_features()
