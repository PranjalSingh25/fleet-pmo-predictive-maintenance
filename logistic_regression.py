"""
FleetOps Anomaly Detection - Phase 5: Logistic Regression
==========================================================
Baseline classifier for predicting high maintenance cost
in the next month.

Input:  outputs/feature_matrix.csv
Output: outputs/lr_predictions.csv
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import config


def run_logistic_regression():
    """Train and evaluate Logistic Regression on the feature matrix."""

    print("=" * 60)
    print("Phase 5 - Logistic Regression Baseline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load feature matrix
    # ------------------------------------------------------------------
    print("\n[1/5] Loading feature matrix ...")
    feature_path = config.OUTPUT_DIR / "feature_matrix.csv"
    df = pd.read_csv(feature_path)
    df["month"] = pd.to_datetime(df["month"])
    print(f"  Shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Temporal split
    # ------------------------------------------------------------------
    print("[2/5] Temporal train/test split ...")

    train_mask = df["year"].isin(config.ML_TRAIN_YEARS)
    test_mask = df["year"] == config.ML_TEST_YEAR

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    print(f"  Train: {train_df.shape[0]} rows (years: {config.ML_TRAIN_YEARS})")
    print(f"  Test:  {test_df.shape[0]} rows (year: {config.ML_TEST_YEAR})")

    # ------------------------------------------------------------------
    # 3. Prepare features and target
    # ------------------------------------------------------------------
    exclude_cols = [
        "truck_id", "month", "year",
        "high_cost_next_month", "maintenance_cost_next_month",
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    target_col = "high_cost_next_month"

    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    # Fill NaN with 0 for features (rolling/lag features at start of series)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # ------------------------------------------------------------------
    # 4. Scale and train
    # ------------------------------------------------------------------
    print("[3/5] Training Logistic Regression ...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    print("[4/5] Evaluating model ...")

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n  Classification Report:")
    print("  " + "-" * 55)
    report = classification_report(y_test, y_pred, zero_division=0)
    for line in report.split("\n"):
        print(f"  {line}")

    print(f"\n  Positive-class metrics (high_cost_next_month = 1):")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {roc_auc:.4f}")

    # ------------------------------------------------------------------
    # 6. Coefficient analysis
    # ------------------------------------------------------------------
    print("\n[5/5] Coefficient analysis (top 5 by |coefficient|) ...")

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_[0],
        "abs_coefficient": np.abs(model.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False)

    top5 = coef_df.head(5)
    print(f"\n  {'Feature':<25} {'Coefficient':>12}")
    print(f"  {'-'*25} {'-'*12}")
    for _, row in top5.iterrows():
        print(f"  {row['feature']:<25} {row['coefficient']:>12.4f}")

    # ------------------------------------------------------------------
    # 7. Save predictions
    # ------------------------------------------------------------------
    predictions_df = test_df[["truck_id", "month", "high_cost_next_month"]].copy()
    predictions_df["lr_predicted"] = y_pred
    predictions_df["lr_probability"] = y_proba

    output_path = config.OUTPUT_DIR / "lr_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"\n  Predictions saved to: {output_path}")

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }

    print("\n" + "=" * 60)
    print("Logistic Regression complete.")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    run_logistic_regression()
