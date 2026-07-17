"""
Phase 1 — Detection
====================
Identifies trucks with maintenance cost overruns vs. prior-year baseline
and flags fuel cost per mile trend anomalies as a leading indicator.

Severity Levels:
    CRITICAL — Maintenance cost >10% over YTD budget
    WARNING  — Maintenance cost >5% over YTD budget
    WATCH    — Fuel cost/mile trending >10% above baseline
               (engine degradation signal before repair bill arrives)

Usage:
    python detection.py
"""
import pandas as pd
import numpy as np

import config


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_utilization_data():
    """Load and prepare truck utilization metrics with date parsing."""
    df = pd.read_csv(config.TRUCK_UTILIZATION_FILE)
    df["month"] = pd.to_datetime(df["month"])
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month
    return df


def load_fuel_purchases():
    """Load fuel purchase records with date parsing."""
    df = pd.read_csv(config.FUEL_PURCHASES_FILE)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    df["month"] = df["purchase_date"].dt.to_period("M").dt.to_timestamp()
    return df


# ─────────────────────────────────────────────────────────────
# MAINTENANCE COST OVERRUN DETECTION
# ─────────────────────────────────────────────────────────────

def compute_baseline_budgets(util_df):
    """
    Compute annual maintenance budget per truck using the baseline year.
    The baseline year's total maintenance cost becomes next year's budget.

    Returns:
        DataFrame with columns [truck_id, annual_budget]
    """
    baseline = util_df[util_df["year"] == config.BASELINE_YEAR]
    budgets = (
        baseline
        .groupby("truck_id")["maintenance_cost"]
        .sum()
        .reset_index()
        .rename(columns={"maintenance_cost": "annual_budget"})
    )
    return budgets


def compute_ytd_overruns(util_df, budgets):
    """
    For each truck in analysis years (2023–2024), compute month-by-month
    YTD actual vs pro-rata YTD budget and flag overruns.

    The pro-rata budget for month M = (annual_budget / 12) × M.
    This catches overruns mid-year rather than waiting for year-end.

    Returns:
        DataFrame with one row per truck per month, including overrun
        calculations and severity flags.
    """
    results = []

    for year in config.ANALYSIS_YEARS:
        year_data = (
            util_df[util_df["year"] == year]
            .sort_values(["truck_id", "month"])
        )

        for truck_id, truck_months in year_data.groupby("truck_id"):
            # Look up this truck's baseline budget
            budget_row = budgets[budgets["truck_id"] == truck_id]
            if budget_row.empty:
                continue

            annual_budget = budget_row["annual_budget"].values[0]

            # Skip trucks with zero or negligible baseline
            # (no maintenance in 2022 → can't compute meaningful overrun %)
            if annual_budget <= 0:
                continue

            ytd_actual = 0.0
            for _, row in truck_months.iterrows():
                month_num = row["month_num"]
                ytd_actual += row["maintenance_cost"]
                ytd_budget = (annual_budget / 12) * month_num

                overrun_pct = ((ytd_actual - ytd_budget) / ytd_budget) * 100
                overrun_dollars = ytd_actual - ytd_budget

                # Assign severity
                if overrun_pct > config.CRITICAL_THRESHOLD_PCT:
                    severity = "CRITICAL"
                elif overrun_pct > config.WARNING_THRESHOLD_PCT:
                    severity = "WARNING"
                else:
                    severity = None

                results.append({
                    "truck_id": truck_id,
                    "year": year,
                    "current_month": month_num,
                    "month_date": row["month"],
                    "ytd_budget": round(ytd_budget, 2),
                    "ytd_actual": round(ytd_actual, 2),
                    "overrun_pct": round(overrun_pct, 2),
                    "overrun_dollars": round(overrun_dollars, 2),
                    "severity": severity,
                    "annual_budget": round(annual_budget, 2),
                })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# FUEL COST PER MILE TREND DETECTION (Leading Indicator)
# ─────────────────────────────────────────────────────────────

def compute_monthly_fuel_cost_per_mile(util_df, fuel_df):
    """
    Compute monthly fuel cost per mile per truck by joining:
    - fuel_purchases (grouped by truck_id + month → total fuel cost)
    - truck_utilization_metrics (total_miles per truck per month)

    Returns:
        DataFrame with columns [truck_id, month, year, month_num,
        total_miles, total_fuel_cost, fuel_cost_per_mile]
    """
    # Monthly fuel cost per truck
    monthly_fuel = (
        fuel_df
        .groupby(["truck_id", "month"])["total_cost"]
        .sum()
        .reset_index()
        .rename(columns={"total_cost": "total_fuel_cost"})
    )

    # Monthly miles from utilization metrics
    monthly_miles = (
        util_df[["truck_id", "month", "total_miles"]]
        .copy()
    )

    # Join: keep only months where we have both fuel and miles data
    merged = monthly_miles.merge(
        monthly_fuel, on=["truck_id", "month"], how="inner"
    )

    # Compute cost per mile (guard against zero-mile months)
    merged["fuel_cost_per_mile"] = np.where(
        merged["total_miles"] > 0,
        merged["total_fuel_cost"] / merged["total_miles"],
        np.nan,
    )

    merged["year"] = merged["month"].dt.year
    merged["month_num"] = merged["month"].dt.month

    return merged[
        ["truck_id", "month", "year", "month_num",
         "total_miles", "total_fuel_cost", "fuel_cost_per_mile"]
    ]


def compute_fuel_trend_flags(fuel_cpm_df):
    """
    Identify trucks where rolling 3-month fuel cost per mile is trending
    >10% above their 2022 baseline. This signals potential engine
    degradation before the maintenance bill arrives.

    Returns:
        DataFrame with fuel trend metrics and alert flags for analysis years.
    """
    # ── Baseline: 2022 average fuel cost/mile per truck ──
    baseline = (
        fuel_cpm_df[fuel_cpm_df["year"] == config.BASELINE_YEAR]
        .groupby("truck_id")["fuel_cost_per_mile"]
        .mean()
        .reset_index()
        .rename(columns={"fuel_cost_per_mile": "baseline_fuel_cpm"})
    )

    # ── Analysis years: rolling 3-month average ──
    analysis = (
        fuel_cpm_df[fuel_cpm_df["year"].isin(config.ANALYSIS_YEARS)]
        .sort_values(["truck_id", "month"])
        .copy()
    )

    analysis["fuel_cpm_rolling_3mo"] = (
        analysis
        .groupby("truck_id")["fuel_cost_per_mile"]
        .transform(
            lambda x: x.rolling(
                window=config.FUEL_ROLLING_WINDOW, min_periods=1
            ).mean()
        )
    )

    # ── Compare to baseline ──
    analysis = analysis.merge(baseline, on="truck_id", how="left")

    analysis["fuel_trend_pct"] = np.where(
        analysis["baseline_fuel_cpm"] > 0,
        (
            (analysis["fuel_cpm_rolling_3mo"] - analysis["baseline_fuel_cpm"])
            / analysis["baseline_fuel_cpm"]
        )
        * 100,
        0.0,
    )

    analysis["fuel_trend_alert"] = (
        analysis["fuel_trend_pct"] > config.FUEL_TREND_THRESHOLD_PCT
    )

    return analysis[
        ["truck_id", "month", "year", "month_num",
         "fuel_cost_per_mile", "fuel_cpm_rolling_3mo",
         "baseline_fuel_cpm", "fuel_trend_pct", "fuel_trend_alert"]
    ]


# ─────────────────────────────────────────────────────────────
# MERGE AND OUTPUT
# ─────────────────────────────────────────────────────────────

def merge_flags(overruns_df, fuel_flags_df):
    """
    Merge maintenance cost overrun flags with fuel trend flags.

    A truck is flagged if it has:
    - A maintenance overrun (WARNING / CRITICAL), OR
    - A fuel trend alert (WATCH — leading indicator)

    CRITICAL and WARNING take priority over WATCH.
    """
    # Rename fuel column to match overruns naming convention
    fuel_flags_renamed = fuel_flags_df.rename(
        columns={"month_num": "current_month"}
    )

    merged = overruns_df.merge(
        fuel_flags_renamed,
        on=["truck_id", "year", "current_month"],
        how="left",
        suffixes=("", "_fuel"),
    )

    # Drop duplicate month column from fuel side
    if "month_fuel" in merged.columns:
        merged = merged.drop(columns=["month_fuel"])

    # Assign final severity: CRITICAL > WARNING > WATCH > None
    def _resolve_severity(row):
        if row["severity"] in ("CRITICAL", "WARNING"):
            return row["severity"]
        if row.get("fuel_trend_alert") is True:
            return "WATCH"
        return row["severity"]  # None — not flagged

    merged["severity"] = merged.apply(_resolve_severity, axis=1)

    return merged


def write_outputs(flagged_df):
    """
    Write flagged trucks to:
    1. Per-month CSV files (for drill-down analysis)
    2. A single combined CSV (for downstream phases)
    """
    output_cols = [
        "truck_id", "year", "current_month", "ytd_budget", "ytd_actual",
        "overrun_pct", "overrun_dollars", "severity", "annual_budget",
        "fuel_cost_per_mile", "fuel_cpm_rolling_3mo", "baseline_fuel_cpm",
        "fuel_trend_pct", "fuel_trend_alert",
    ]
    available_cols = [c for c in output_cols if c in flagged_df.columns]

    # Per-month files
    months_written = 0
    for (year, month_num), month_df in flagged_df.groupby(["year", "current_month"]):
        month_str = f"{year}-{month_num:02d}"
        output_path = config.OUTPUT_DIR / f"flagged_trucks_{month_str}.csv"
        month_df[available_cols].to_csv(output_path, index=False)
        months_written += 1

    # Combined file for downstream phases
    combined_path = config.OUTPUT_DIR / "flagged_trucks_all.csv"
    flagged_df[available_cols].to_csv(combined_path, index=False)

    return months_written, combined_path


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_detection():
    """
    Execute the full Phase 1 detection pipeline.

    Returns:
        DataFrame of all flagged truck-month records.
    """
    print("=" * 60)
    print("PHASE 1 — DETECTION")
    print("=" * 60)

    # ── Step 1: Load data ──
    print("\n[1/5] Loading truck utilization metrics...")
    util_df = load_utilization_data()
    truck_count = util_df["truck_id"].nunique()
    print(f"      {len(util_df):,} records | {truck_count} trucks")

    # ── Step 2: Compute baseline budgets ──
    print(f"\n[2/5] Computing baseline budgets from {config.BASELINE_YEAR}...")
    budgets = compute_baseline_budgets(util_df)
    print(f"      Budgets for {len(budgets)} trucks")
    print(f"      Range: ${budgets['annual_budget'].min():,.0f}"
          f" – ${budgets['annual_budget'].max():,.0f}")
    print(f"      Median: ${budgets['annual_budget'].median():,.0f}")

    # ── Step 3: Compute YTD overruns ──
    print(f"\n[3/5] Computing YTD overruns for {config.ANALYSIS_YEARS}...")
    overruns = compute_ytd_overruns(util_df, budgets)
    maint_flagged = overruns[overruns["severity"].notna()]
    print(f"      Total snapshots: {len(overruns):,}")
    print(f"      Flagged (maintenance):")
    print(f"        WARNING:  {len(maint_flagged[maint_flagged['severity'] == 'WARNING']):,}")
    print(f"        CRITICAL: {len(maint_flagged[maint_flagged['severity'] == 'CRITICAL']):,}")

    # ── Step 4: Fuel cost per mile trends ──
    print("\n[4/5] Computing fuel cost per mile trends...")
    fuel_df = load_fuel_purchases()
    print(f"      Loaded {len(fuel_df):,} fuel purchases")

    fuel_cpm = compute_monthly_fuel_cost_per_mile(util_df, fuel_df)
    fuel_flags = compute_fuel_trend_flags(fuel_cpm)
    fuel_alerts = fuel_flags[fuel_flags["fuel_trend_alert"]]
    print(f"      Fuel trend alerts: {len(fuel_alerts):,} truck-month snapshots")
    print(f"      Trucks with >=1 alert: {fuel_alerts['truck_id'].nunique()}")

    # ── Step 5: Merge and output ──
    print("\n[5/5] Merging flags and writing output...")
    merged = merge_flags(overruns, fuel_flags)
    all_flagged = merged[merged["severity"].notna()].copy()

    months_written, combined_path = write_outputs(all_flagged)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("DETECTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Unique trucks flagged: {all_flagged['truck_id'].nunique()} / {truck_count}")
    print(f"  Monthly files written: {months_written}")
    print(f"  Combined output: {combined_path}")
    print()

    severity_counts = all_flagged["severity"].value_counts()
    for severity, count in severity_counts.items():
        print(f"  {severity:>8}: {count:,} truck-month snapshots")

    # Show sample of highest overruns
    print(f"\n  Top 5 highest overruns:")
    top5 = (
        all_flagged
        .sort_values("overrun_pct", ascending=False)
        .head(5)
    )
    for _, row in top5.iterrows():
        print(f"    {row['truck_id']} | {row['year']}-{row['current_month']:02d}"
              f" | {row['overrun_pct']:+.1f}% | ${row['overrun_dollars']:+,.0f}"
              f" | {row['severity']}")

    return all_flagged


if __name__ == "__main__":
    run_detection()
