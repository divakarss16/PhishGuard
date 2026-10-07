"""Machine Learning Training, Evaluation, and Serialization Module for PhishGuard.

Trains and evaluates 4 classification models (Logistic Regression, Decision Tree,
Random Forest, and XGBoost) on dataset/processed/features.csv using a reproducible
stratified 80/20 train/test split. Generates metrics, plots, and serialized model artifacts.
"""

import os
from pathlib import Path

# Configure matplotlib cache dir before importing matplotlib to prevent permission errors
cache_dir = Path.cwd() / "scratch" / ".matplotlib_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_dir)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


LABEL_MAPPING = {"legitimate": 0, "phishing": 1}
REVERSE_LABEL_MAPPING = {0: "legitimate", 1: "phishing"}


def train_and_evaluate_models(
    feature_dataset_path: str = "dataset/processed/features.csv",
    models_dir: str = "models",
    outputs_dir: str = "outputs",
) -> dict:
    """Train 4 models, evaluate performance, generate plots, and serialize the best model."""
    feature_path = Path(feature_dataset_path)
    models_path = Path(models_dir)
    outputs_path = Path(outputs_dir)

    models_path.mkdir(parents=True, exist_ok=True)
    outputs_path.mkdir(parents=True, exist_ok=True)

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature dataset not found at {feature_path}")

    # 1. Load dataset
    print(f"Loading feature dataset from {feature_path}...")
    df = pd.read_csv(feature_path)

    # Data leakage check: Verify 'url' and 'label' are present
    assert "url" in df.columns, "Expected 'url' column for traceability!"
    assert "label" in df.columns, "Expected 'label' target column!"

    # 2. Separate X (numerical features) and y (target label)
    feature_cols = [c for c in df.columns if c not in ["url", "label"]]
    X = df[feature_cols].copy()
    y = df["label"].map(LABEL_MAPPING)

    if y.isnull().any():
        raise ValueError(f"Unmapped labels found! Allowed: {list(LABEL_MAPPING.keys())}")

    print(f"Total Dataset Shape: X={X.shape}, y={y.shape}")
    print(f"Feature Count: {len(feature_cols)}")
    print(f"Label Mapping: {LABEL_MAPPING}")

    # 3. Reproducible Stratified Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTrain/Test Split Breakdown:")
    print(f"  - Train Set Rows: {len(X_train)} ({len(X_train)/len(X)*100:.2f}%)")
    print(f"  - Test Set Rows:  {len(X_test)} ({len(X_test)/len(X)*100:.2f}%)")
    print(f"  - Train Class Distribution: {dict(pd.Series(y_train).value_counts())}")
    print(f"  - Test Class Distribution:  {dict(pd.Series(y_test).value_counts())}")

    # Calculate scale_pos_weight for XGBoost (~3.44)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # 4. Define Models (n_jobs=1 for single-threaded sandbox stability)
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, class_weight="balanced", n_jobs=1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, scale_pos_weight=pos_weight, random_state=42, eval_metric="logloss", n_jobs=1),
    }

    results = {}
    roc_data = {}

    print("\n" + "=" * 70)
    print("                    MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # 5. Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Predictions & Probabilities on UNSEEN Test Set
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, y_prob)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr_val = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        legitimate_accuracy = specificity

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "specificity": specificity,
            "false_positive_rate": fpr_val,
            "legitimate_accuracy": legitimate_accuracy,
            "confusion_matrix": cm,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

        # Store ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)

        print(f"Results for {name}:")
        print(f"  - Accuracy:            {acc:.4f}")
        print(f"  - Precision:           {prec:.4f}")
        print(f"  - Recall:              {rec:.4f} (Phishing Detection Rate)")
        print(f"  - F1-Score:            {f1:.4f}")
        print(f"  - ROC-AUC:             {auc:.4f}")
        print(f"  - Specificity (Legit): {specificity:.4f}")
        print(f"  - False Positive Rate: {fpr_val:.4f}")
        print(f"  - Confusion Matrix:    TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # 6. Model Comparison Table
    print("\n" + "=" * 105)
    print("                                MODEL COMPARISON TABLE")
    print("=" * 105)
    print(f"{'Model':<22} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9} | {'Specificity':<11} | {'FPR':<7}")
    print("-" * 105)
    for name, r in results.items():
        print(f"{name:<22} | {r['accuracy']:<9.4f} | {r['precision']:<9.4f} | {r['recall']:<9.4f} | {r['f1']:<9.4f} | {r['roc_auc']:<9.4f} | {r['specificity']:<11.4f} | {r['false_positive_rate']:<7.4f}")
    print("=" * 105)

    # 7. Model Selection Criterion:
    # Prioritize: 1. Recall for phishing, 2. F1-Score, 3. ROC-AUC, 4. Precision, 5. Specificity
    best_name = max(
        results.keys(),
        key=lambda k: (
            results[k]["recall"],
            results[k]["f1"],
            results[k]["roc_auc"],
            results[k]["precision"],
            results[k]["specificity"],
        ),
    )

    best_result = results[best_name]
    best_model = best_result["model"]

    print("\n" + "=" * 70)
    print("                     FINAL MODEL SELECTION")
    print("=" * 70)
    print(f"Selected Production Model: {best_name}")
    print("Selection Rationale:")
    print(f"  - Highest Phishing Recall:  {best_result['recall']:.4f} (Minimizes dangerous False Negatives)")
    print(f"  - F1-Score:                 {best_result['f1']:.4f}")
    print(f"  - ROC-AUC Score:            {best_result['roc_auc']:.4f}")
    print(f"  - False Negatives (Misses): {best_result['fn']} out of {best_result['fn'] + best_result['tp']} phishing URLs")
    print("=" * 70)

    # 8. Plot & Save ROC Curve Comparison
    plt.figure(figsize=(9, 7))
    for name, (fpr, tpr, auc_val) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("ROC Curve Comparison — PhishGuard Models", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    roc_plot_path = outputs_path / "roc_curve.png"
    plt.savefig(roc_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC Curve comparison plot to {roc_plot_path}")

    # 9. Plot & Save Confusion Matrix for Selected Model
    plt.figure(figsize=(7, 6))
    cm = best_result["confusion_matrix"]
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Actual Legitimate\n(Class 0)", "Actual Phishing\n(Class 1)"], fontsize=10)
    plt.yticks(tick_marks, ["Predicted Legitimate\n(Class 0)", "Predicted Phishing\n(Class 1)"], fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]:,}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    cm_plot_path = outputs_path / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Confusion Matrix plot to {cm_plot_path}")

    # 10. Extract & Plot Feature Importance
    top_10_features = []
    actual_model = best_model
    if isinstance(best_model, Pipeline):
        actual_model = best_model.named_steps["classifier"]

    if hasattr(actual_model, "feature_importances_"):
        importances = actual_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False)

        top_10_features = feature_importance_df.head(10)

        print("\nTop 10 Feature Importances:")
        print(f"{'Feature Name':<32} | {'Importance':<10}")
        print("-" * 46)
        for _, row in top_10_features.iterrows():
            print(f"{row['Feature']:<32} | {row['Importance']:<10.4f}")

        plt.figure(figsize=(10, 6))
        plt.barh(top_10_features["Feature"][::-1], top_10_features["Importance"][::-1], color="#2b5c8f")
        plt.xlabel("Relative Feature Importance", fontsize=11)
        plt.ylabel("Feature Name", fontsize=11)
        plt.title(f"Top 10 Feature Importances — {best_name}", fontsize=13, fontweight="bold")
        plt.grid(True, alpha=0.3, axis="x")
        fi_plot_path = outputs_path / "feature_importance.png"
        plt.savefig(fi_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Feature Importance plot to {fi_plot_path}")

    # 11. Model & Feature Schema Serialization
    model_file_path = models_path / "phishing_model.pkl"
    feature_names_file_path = models_path / "feature_names.pkl"

    joblib.dump(best_model, model_file_path)
    joblib.dump(feature_cols, feature_names_file_path)

    print(f"\nSerialized final production model to {model_file_path}")
    print(f"Serialized feature schema to {feature_names_file_path}")

    return {
        "results": results,
        "best_model_name": best_name,
        "best_result": best_result,
        "top_10_features": top_10_features,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    train_and_evaluate_models()
