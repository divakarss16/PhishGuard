"""Data Preprocessing and Cleaning Module for PhishGuard.

Performs conservative cleaning, label normalization, deduplication,
and integrity verification on raw URL datasets.
"""

from pathlib import Path
from urllib.parse import urlparse
import pandas as pd


def extract_netloc(u: str) -> str:
    """Extract clean hostname from URL string without network calls."""
    u_str = str(u).strip()
    u_lower = u_str.lower()
    if u_lower.startswith("https://") or u_lower.startswith("http://") or "://" in u_lower:
        target = u_str
    else:
        target = "//" + u_str
    try:
        parsed = urlparse(target)
        return parsed.netloc.split(":")[0].strip("[]")
    except Exception:
        return ""


def preprocess_data(
    raw_dataset_path: str = "dataset/raw/phishing_urls.csv",
    output_cleaned_path: str = "dataset/processed/cleaned_urls.csv",
) -> pd.DataFrame:
    """Read raw URL dataset, clean, normalize labels, augment domain-only & protocol variations, deduplicate, and save cleaned CSV."""
    raw_path = Path(raw_dataset_path)
    out_path = Path(output_cleaned_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    # 1. Read raw dataset
    print(f"Reading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    original_row_count = len(df)
    original_labels = df["label"].value_counts().to_dict()

    print(f"Original Row Count: {original_row_count}")
    print(f"Original Label Counts: {original_labels}")

    # 2. Conservative URL cleaning & strict format filtering
    df["url"] = df["url"].astype(str).str.strip()
    
    from src.feature_extraction import validate_url_format
    valid_mask = df["url"].apply(lambda u: validate_url_format(u)[0])
    invalid_dropped = (~valid_mask).sum()
    if invalid_dropped > 0:
        print(f"Filtered out {invalid_dropped} malformed/invalid URLs during raw preprocessing.")
    df = df[valid_mask].copy()

    # 3. Label normalization
    label_map = {"good": "legitimate", "bad": "phishing"}
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(label_map)

    if df["label"].isnull().any():
        unmapped_count = df["label"].isnull().sum()
        raise ValueError(f"Found {unmapped_count} unmapped or unexpected label values in raw dataset!")

    normalized_label_counts = df["label"].value_counts().to_dict()
    print(f"Normalized Label Counts: {normalized_label_counts}")

    # 4. Domain-Only & Scheme Protocol Representation Augmentation
    print("Performing systematic domain-only and protocol scheme augmentation...")
    df["hostname"] = df["url"].map(extract_netloc)

    legit_df = df[df["label"] == "legitimate"]
    phish_df = df[df["label"] == "phishing"]

    legit_hosts = sorted(list(set(legit_df["hostname"]) - {""}))
    phish_hosts = sorted(list(set(phish_df["hostname"]) - {""}))

    # Exclude hostnames appearing in both classes to maintain clean class boundaries
    overlap_hosts = set(legit_hosts).intersection(set(phish_hosts))
    legit_hosts = [h for h in legit_hosts if h not in overlap_hosts]
    phish_hosts = [h for h in phish_hosts if h not in overlap_hosts]

    aug_rows = []
    # Augment all unique legitimate hostnames with https://, http://, and scheme-less domain-only URLs
    for h in legit_hosts:
        aug_rows.append({"url": f"https://{h}", "label": "legitimate"})
        aug_rows.append({"url": f"http://{h}", "label": "legitimate"})
        aug_rows.append({"url": h, "label": "legitimate"})

    # Augment unique phishing hostnames with https://, http://, and scheme-less domain-only URLs
    for h in phish_hosts:
        aug_rows.append({"url": f"https://{h}", "label": "phishing"})
        aug_rows.append({"url": f"http://{h}", "label": "phishing"})
        aug_rows.append({"url": h, "label": "phishing"})

    # Representative path & query augmentation across sampled legitimate and phishing hostnames
    import random
    rng = random.Random(42)
    sampled_legit = rng.sample(legit_hosts, min(40000, len(legit_hosts)))
    for h in sampled_legit:
        aug_rows.append({"url": f"https://{h}/login", "label": "legitimate"})
        aug_rows.append({"url": f"https://{h}/signin", "label": "legitimate"})
        aug_rows.append({"url": f"https://{h}/search?q=python", "label": "legitimate"})
        aug_rows.append({"url": f"https://{h}/en-us/downloads", "label": "legitimate"})

    sampled_phish = rng.sample(phish_hosts, min(10000, len(phish_hosts)))
    for h in sampled_phish:
        aug_rows.append({"url": f"https://{h}/login", "label": "phishing"})
        aug_rows.append({"url": f"https://{h}/signin", "label": "phishing"})
        aug_rows.append({"url": f"https://{h}/verify", "label": "phishing"})

    # 4B. Reproducible IP Address Pattern Augmentation (Legitimate Portals vs Phishing Logins)
    print("Performing reproducible IP address pattern augmentation...")
    sample_ips = []
    # Private subnets
    for i in range(1, 51):
        sample_ips.append(f"192.168.1.{i}")
        sample_ips.append(f"10.0.0.{i}")
        sample_ips.append(f"172.16.0.{i}")
    # Public institutional subnets
    for i in range(1, 51):
        sample_ips.append(f"110.172.151.{i}")
        sample_ips.append(f"140.82.112.{i}")
        sample_ips.append(f"202.54.1.{i}")
        sample_ips.append(f"198.51.100.{i}")

    legit_ip_paths = [
        "",
        "/",
        "/index.html",
        "/index.php",
        "/main.html",
        "/home",
        "/portal",
        "/portal/index.html",
        "/dashboard",
        "/app/dashboard",
        "/LMS/index.html",
        "/lms/course/view.php",
        "/moodle/index.php",
        "/system/status",
        "/admin/main.php",
        "/docs/index.html",
        "/site/page.html",
        "/public/index.html",
        "/assets/img/logo.png",
        "/en/default.aspx",
    ]

    phish_ip_paths = [
        "/login",
        "/login.php",
        "/signin",
        "/signin/index.html",
        "/verify",
        "/verify-account",
        "/verify/identity.php",
        "/account/update",
        "/account-security/login",
        "/bank/verify.html",
        "/paypal/signin.php",
        "/secure/update-password",
        "/auth/login",
        "/credential/confirm",
        "/webmail/login.php",
        "/banking/verification.html",
        "/wallet/restore",
    ]

    for ip in sample_ips:
        for path in legit_ip_paths:
            aug_rows.append({"url": f"http://{ip}{path}", "label": "legitimate"})
            aug_rows.append({"url": f"https://{ip}{path}", "label": "legitimate"})
        for path in phish_ip_paths:
            aug_rows.append({"url": f"http://{ip}{path}", "label": "phishing"})
            aug_rows.append({"url": f"https://{ip}{path}", "label": "phishing"})

    df_augmented = pd.concat(
        [df[["url", "label"]], pd.DataFrame(aug_rows)], ignore_index=True
    )

    # 5. Deduplication strategy (trim + lowercase comparison)
    print("Performing URL deduplication...")
    df_augmented["normalized_url"] = df_augmented["url"].str.lower()

    # Drop duplicates keeping the first occurrence
    df_cleaned = df_augmented.drop_duplicates(subset=["normalized_url"], keep="first").copy()
    df_cleaned.drop(columns=["normalized_url"], inplace=True)

    cleaned_row_count = len(df_cleaned)
    rows_removed = len(df_augmented) - cleaned_row_count
    final_class_counts = df_cleaned["label"].value_counts().to_dict()

    # 6. Dataset Integrity Verification
    print("Verifying dataset integrity...")
    assert df_cleaned["url"].isnull().sum() == 0, "Null URLs found after cleaning!"
    assert df_cleaned["label"].isnull().sum() == 0, "Null labels found after cleaning!"
    assert set(df_cleaned["label"].unique()) == {"legitimate", "phishing"}, "Unexpected label values remaining!"
    assert (df_cleaned["url"] == "").sum() == 0, "Empty URL strings found after cleaning!"

    # Verify no duplicate normalized URLs remain
    dup_check = df_cleaned["url"].str.lower().duplicated().sum()
    assert dup_check == 0, f"Found {dup_check} duplicate normalized URLs remaining after deduplication!"

    # 7. Save cleaned dataset
    df_cleaned.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset to {out_path} ({cleaned_row_count} rows)")

    # 7. Print Final Cleaning Report
    print("=" * 60)
    print("           PHISHGUARD DATA CLEANING REPORT")
    print("=" * 60)
    print(f"Input Raw Dataset Path:       {raw_path}")
    print(f"Output Cleaned Dataset Path:   {out_path}")
    print(f"Rows Before Cleaning:         {original_row_count}")
    print(f"Rows After Cleaning:          {cleaned_row_count}")
    print(f"Rows Removed (Duplicates):    {rows_removed}")
    print("\nOriginal Label Counts:")
    for k, v in original_labels.items():
        print(f"  - {k}: {v}")
    print("\nFinal Normalized Class Counts:")
    for k, v in final_class_counts.items():
        pct = round((v / cleaned_row_count) * 100, 2)
        print(f"  - {k}: {v} ({pct}%)")
    print("=" * 60)

    return df_cleaned


if __name__ == "__main__":
    preprocess_data()
