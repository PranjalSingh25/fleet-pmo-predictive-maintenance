"""
FleetOps Anomaly Detection - Phase 5: XGBoost Model
=====================================================
Advanced classifier for predicting high maintenance cost
in the next month. Compares against Logistic Regression baseline.

Input:  outputs/feature_matrix.csv
Output: outputs/xgb_predictions.csv
        outputs/feature_importance.png
        outputs/model_comparison_report.md
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import config


def run_xgboost():
    """Train XGBoost, evaluate, compare with LR, and generate reports."""

    print("=" * 60)
    print("Phase 5 - XGBoost Model")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load feature matrix
    # ------------------------------------------------------------------
    print("\n[1/7] Loading feature matrix ...")
    feature_path = config.OUTPUT_DIR / "feature_matrix.csv"
    df = pd.read_csv(feature_path)
    df["month"] = pd.to_datetime(df["month"])
    print(f"  Shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Temporal split
    # ------------------------------------------------------------------
    print("[2/7] Temporal train/test split ...")

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

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    # Fill NaN with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # ------------------------------------------------------------------
    # 4. Train XGBoost
    # ------------------------------------------------------------------
    print("[3/7] Training XGBoost ...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train, verbose=False)

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    print("[4/7] Evaluating XGBoost ...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    xgb_precision = precision_score(y_test, y_pred, zero_division=0)
    xgb_recall = recall_score(y_test, y_pred, zero_division=0)
    xgb_f1 = f1_score(y_test, y_pred, zero_division=0)
    xgb_roc_auc = roc_auc_score(y_test, y_proba)

    print("\n  Classification Report:")
    print("  " + "-" * 55)
    report = classification_report(y_test, y_pred, zero_division=0)
    for line in report.split("\n"):
        print(f"  {line}")

    print(f"\n  Positive-class metrics (high_cost_next_month = 1):")
    print(f"    Precision : {xgb_precision:.4f}")
    print(f"    Recall    : {xgb_recall:.4f}")
    print(f"    F1 Score  : {xgb_f1:.4f}")
    print(f"    ROC-AUC   : {xgb_roc_auc:.4f}")

    # ------------------------------------------------------------------
    # 6. Feature importance plot
    # ------------------------------------------------------------------
    print("[5/7] Generating feature importance plot ...")

    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    top10 = imp_df.head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        top10["feature"].values[::-1],
        top10["importance"].values[::-1],
        color="#2196F3",
        edgecolor="#1565C0",
    )
    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_title("XGBoost - Top 10 Feature Importances", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    plot_path = config.OUTPUT_DIR / "feature_importance.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved to: {plot_path}")

    # ------------------------------------------------------------------
    # 7. Compare with Logistic Regression
    # ------------------------------------------------------------------
    print("[6/7] Comparing with Logistic Regression ...")

    lr_pred_path = config.OUTPUT_DIR / "lr_predictions.csv"
    lr_df = pd.read_csv(lr_pred_path)

    lr_precision = precision_score(
        lr_df["high_cost_next_month"], lr_df["lr_predicted"], zero_division=0
    )
    lr_recall = recall_score(
        lr_df["high_cost_next_month"], lr_df["lr_predicted"], zero_division=0
    )
    lr_f1 = f1_score(
        lr_df["high_cost_next_month"], lr_df["lr_predicted"], zero_division=0
    )
    lr_roc_auc = roc_auc_score(
        lr_df["high_cost_next_month"], lr_df["lr_probability"]
    )

    print(f"\n  {'Metric':<12} {'Logistic Reg':>14} {'XGBoost':>14} {'Diff':>10}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*10}")
    for metric_name, lr_val, xgb_val in [
        ("Precision", lr_precision, xgb_precision),
        ("Recall", lr_recall, xgb_recall),
        ("F1 Score", lr_f1, xgb_f1),
        ("ROC-AUC", lr_roc_auc, xgb_roc_auc),
    ]:
        diff = xgb_val - lr_val
        sign = "+" if diff >= 0 else ""
        print(
            f"  {metric_name:<12} {lr_val:>14.4f} {xgb_val:>14.4f} {sign}{diff:>9.4f}"
        )

    # ------------------------------------------------------------------
    # 8. Generate comparison report
    # ------------------------------------------------------------------
    print("[7/7] Generating model comparison report ...")

    # Build top 10 feature importance text
    imp_lines = []
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        imp_lines.append(f"| {rank} | {row['feature']} | {row['importance']:.4f} |")

    report_md = f"""# FleetOps Anomaly Detection - Model Comparison Report

## Phase 5: ML Model Evaluation

---

## Model Comparison

| Metric | Logistic Regression | XGBoost | Difference |
|--------|--------------------:|--------:|-----------:|
| Precision | {lr_precision:.4f} | {xgb_precision:.4f} | {xgb_precision - lr_precision:+.4f} |
| Recall | {lr_recall:.4f} | {xgb_recall:.4f} | {xgb_recall - lr_recall:+.4f} |
| F1 Score | {lr_f1:.4f} | {xgb_f1:.4f} | {xgb_f1 - lr_f1:+.4f} |
| ROC-AUC | {lr_roc_auc:.4f} | {xgb_roc_auc:.4f} | {xgb_roc_auc - lr_roc_auc:+.4f} |

**Training Data**: Years {config.ML_TRAIN_YEARS}
**Test Data**: Year {config.ML_TEST_YEAR}

---

## Feature Importance Analysis (XGBoost)

| Rank | Feature | Importance |
|------|---------|----------:|
{chr(10).join(imp_lines)}

### Key Observations

The top features reveal what drives high maintenance costs in the fleet:

1. **Cost-history features** (rolling averages and lags) are typically the strongest predictors,
   confirming that trucks with recent high maintenance spending are likely to continue being expensive.
2. **Operational features** like utilization rate, trip counts, and fuel cost per mile capture
   how hard a truck is being worked, which contributes to wear and future maintenance needs.
3. **Truck age** reflects the general mechanical degradation curve - older trucks require more
   maintenance regardless of usage patterns.
4. **Safety incident flags** may correlate with poorly-maintained vehicles or rough operating conditions.

---

## When Would You Pick Logistic Regression Over XGBoost and Vice Versa?

### Choose Logistic Regression When:

1. **Interpretability is critical**: Fleet managers and maintenance supervisors need to understand
   *why* a truck is flagged. LR coefficients directly show the direction and magnitude of each
   feature's effect (e.g., "every $1,000 increase in 3-month rolling maintenance cost increases
   the odds of a high-cost month by X%"). This transparency is essential for gaining stakeholder
   trust in maintenance budget decisions.

2. **Regulatory or audit requirements**: If the maintenance prediction system feeds into compliance
   reporting or insurance assessments, a transparent, auditable model is often preferred or required.

3. **Small, stable datasets**: When the fleet is small (fewer than 50 trucks) or the data history
   is limited, LR's lower complexity reduces overfitting risk and produces more stable predictions.

4. **Baseline and sanity checking**: LR serves as an excellent baseline. If XGBoost cannot
   meaningfully beat LR, the problem may not benefit from complex modeling, or the features
   may need improvement.

### Choose XGBoost When:

1. **Predictive accuracy is the priority**: In fleet operations, missed predictions of high-cost
   months can mean unexpected breakdowns, roadside failures, and delivery delays. XGBoost's
   ability to capture non-linear interactions (e.g., the combined effect of truck age AND high
   utilization AND winter months) typically yields better recall and F1 scores.

2. **Complex feature interactions exist**: Fleet maintenance costs are driven by interacting
   factors - a 10-year-old truck running high miles in winter is not simply the sum of its
   individual risk factors. XGBoost captures these interactions automatically.

3. **Operational decision-making**: When the model's output directly drives preventive
   maintenance scheduling, parts inventory pre-ordering, or driver assignment, higher accuracy
   translates to direct cost savings. Even a small improvement in F1 score can prevent costly
   roadside breakdowns.

4. **Sufficient training data**: With {train_df.shape[0]} training samples across {len(train_df['truck_id'].unique())}
   trucks, there is enough data for XGBoost to learn meaningful patterns without severe overfitting.

### Recommendation for This Fleet

For **production deployment**, use XGBoost as the primary model for maintenance cost prediction,
but maintain the LR model as a secondary explainability layer. When XGBoost flags a truck as
high-risk, use the LR coefficients to generate a human-readable explanation of the top contributing
factors. This hybrid approach gives fleet managers both accuracy and interpretability.
"""

    report_path = config.OUTPUT_DIR / "model_comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  Report saved to: {report_path}")

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------
    predictions_df = test_df[["truck_id", "month", "high_cost_next_month"]].copy()
    predictions_df["xgb_predicted"] = y_pred
    predictions_df["xgb_probability"] = y_proba

    output_path = config.OUTPUT_DIR / "xgb_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"  Predictions saved to: {output_path}")

    metrics = {
        "precision": xgb_precision,
        "recall": xgb_recall,
        "f1": xgb_f1,
        "roc_auc": xgb_roc_auc,
    }

    print("\n" + "=" * 60)
    print("XGBoost model complete.")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    run_xgboost()
