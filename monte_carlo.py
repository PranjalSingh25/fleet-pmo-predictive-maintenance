"""
FleetOps Anomaly Detection - Phase 3: Monte Carlo Simulation
=============================================================
Forecasts year-end maintenance cost overruns for flagged trucks
using Monte Carlo simulation with vectorized NumPy operations.
"""
import time
import numpy as np
import pandas as pd

import config


def run_monte_carlo():
    """Run Monte Carlo simulations to forecast year-end maintenance costs."""
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load input data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 3: MONTE CARLO SIMULATION")
    print("=" * 60)
    print()

    print("[Step 1/6] Loading input data ...")
    flagged = pd.read_csv(config.OUTPUT_DIR / "flagged_trucks_all.csv")
    utilization = pd.read_csv(config.TRUCK_UTILIZATION_FILE)
    print(f"  Flagged trucks file : {len(flagged):,} rows")
    print(f"  Utilization file    : {len(utilization):,} rows")

    # ------------------------------------------------------------------
    # Step 2: Filter to forecastable rows
    # ------------------------------------------------------------------
    print()
    print("[Step 2/6] Filtering to forecastable truck-month snapshots ...")

    # Only rows with a non-null severity
    flagged = flagged.dropna(subset=["severity"])

    # Only months 1-11 (month 12 has 0 remaining months)
    flagged = flagged[flagged["current_month"] <= 11]

    # Keep unique (truck_id, year, current_month) combinations
    flagged = flagged.drop_duplicates(subset=["truck_id", "year", "current_month"])

    print(f"  Forecastable snapshots: {len(flagged):,}")
    print(f"  Unique trucks         : {flagged['truck_id'].nunique()}")

    # ------------------------------------------------------------------
    # Step 3: Compute monthly maintenance cost statistics per truck
    # ------------------------------------------------------------------
    print()
    print("[Step 3/6] Computing maintenance cost statistics from full history ...")

    # Parse the month column and extract month number for reference
    utilization["month"] = pd.to_datetime(utilization["month"])

    # Compute mean and std of monthly maintenance_cost per truck (all history)
    cost_stats = (
        utilization
        .groupby("truck_id")["maintenance_cost"]
        .agg(mean_monthly_cost="mean", std_monthly_cost="std")
        .reset_index()
    )
    # Fill NaN std (trucks with only 1 data point) with 0
    cost_stats["std_monthly_cost"] = cost_stats["std_monthly_cost"].fillna(0)

    print(f"  Trucks with statistics: {len(cost_stats):,}")
    print(f"  Mean monthly cost (avg across trucks): ${cost_stats['mean_monthly_cost'].mean():,.2f}")
    print(f"  Std  monthly cost (avg across trucks): ${cost_stats['std_monthly_cost'].mean():,.2f}")

    # Merge statistics onto flagged data
    flagged = flagged.merge(cost_stats, on="truck_id", how="left")

    # Drop rows where we have no cost history
    before = len(flagged)
    flagged = flagged.dropna(subset=["mean_monthly_cost"])
    after = len(flagged)
    if before != after:
        print(f"  Dropped {before - after} rows with no maintenance cost history")

    # ------------------------------------------------------------------
    # Step 4: Run Monte Carlo simulations (vectorized)
    # ------------------------------------------------------------------
    print()
    print("[Step 4/6] Running Monte Carlo simulations ...")
    print(f"  Simulations per snapshot: {config.NUM_SIMULATIONS:,}")
    print(f"  Random seed            : {config.RANDOM_SEED}")

    rng = np.random.default_rng(config.RANDOM_SEED)

    # Compute remaining months
    flagged["remaining_months"] = 12 - flagged["current_month"]

    results = []
    total = len(flagged)

    for idx, row in flagged.iterrows():
        remaining = int(row["remaining_months"])
        mu = row["mean_monthly_cost"]
        sigma = row["std_monthly_cost"]
        ytd = row["ytd_actual"]
        budget = row["annual_budget"]

        if remaining <= 0:
            continue

        # Vectorized: generate all simulations at once
        # Shape: (NUM_SIMULATIONS, remaining_months)
        if sigma > 0:
            simulated_months = rng.normal(mu, sigma, size=(config.NUM_SIMULATIONS, remaining))
        else:
            simulated_months = np.full((config.NUM_SIMULATIONS, remaining), mu)

        # Clip to >= 0 (no negative costs)
        simulated_months = simulated_months.clip(min=0)

        # Sum across remaining months and add YTD actual
        projected_year_end = simulated_months.sum(axis=1) + ytd

        # Compute metrics
        p_over_budget = (projected_year_end > budget).mean()
        p10, p50, p90 = np.percentile(projected_year_end, [10, 50, 90])
        projected_overrun_p50 = p50 - budget

        results.append({
            "truck_id": row["truck_id"],
            "year": int(row["year"]),
            "forecast_month": int(row["current_month"]),
            "ytd_actual": round(ytd, 2),
            "annual_budget": round(budget, 2),
            "remaining_months": remaining,
            "mean_monthly_cost": round(mu, 2),
            "std_monthly_cost": round(sigma, 2),
            "p_over_budget": round(p_over_budget, 4),
            "projected_p10": round(p10, 2),
            "projected_p50": round(p50, 2),
            "projected_p90": round(p90, 2),
            "projected_overrun_p50": round(projected_overrun_p50, 2),
            "severity": row["severity"],
        })

    forecast_df = pd.DataFrame(results)
    print(f"  Completed {total:,} simulations")

    # ------------------------------------------------------------------
    # Step 5: Save outputs
    # ------------------------------------------------------------------
    print()
    print("[Step 5/6] Saving outputs ...")

    # Combined output
    combined_path = config.OUTPUT_DIR / "forecast_all.csv"
    forecast_df.to_csv(combined_path, index=False)
    print(f"  Combined : {combined_path} ({len(forecast_df):,} rows)")

    # Per-month files
    month_files_written = 0
    for (year, month), group in forecast_df.groupby(["year", "forecast_month"]):
        month_str = f"{int(year)}-{int(month):02d}"
        month_path = config.OUTPUT_DIR / f"forecast_{month_str}.csv"
        group.to_csv(month_path, index=False)
        month_files_written += 1

    print(f"  Monthly  : {month_files_written} files written")

    # ------------------------------------------------------------------
    # Step 6: Summary statistics
    # ------------------------------------------------------------------
    print()
    print("[Step 6/6] Summary statistics")
    print("-" * 60)

    print(f"  Total forecasts generated : {len(forecast_df):,}")
    print(f"  Unique trucks forecasted  : {forecast_df['truck_id'].nunique()}")
    print(f"  Year range                : {forecast_df['year'].min()}-{forecast_df['year'].max()}")
    print()

    # Over-budget probability distribution
    print("  P(over budget) distribution:")
    high_risk = (forecast_df["p_over_budget"] >= 0.75).sum()
    med_risk = ((forecast_df["p_over_budget"] >= 0.50) & (forecast_df["p_over_budget"] < 0.75)).sum()
    low_risk = ((forecast_df["p_over_budget"] >= 0.25) & (forecast_df["p_over_budget"] < 0.50)).sum()
    minimal = (forecast_df["p_over_budget"] < 0.25).sum()
    print(f"    High risk   (>=75%) : {high_risk:,}")
    print(f"    Medium risk (50-74%): {med_risk:,}")
    print(f"    Low risk    (25-49%): {low_risk:,}")
    print(f"    Minimal     (<25%)  : {minimal:,}")
    print()

    # Projected overrun statistics
    print("  Projected overrun at P50:")
    print(f"    Mean   : ${forecast_df['projected_overrun_p50'].mean():>12,.2f}")
    print(f"    Median : ${forecast_df['projected_overrun_p50'].median():>12,.2f}")
    print(f"    Max    : ${forecast_df['projected_overrun_p50'].max():>12,.2f}")
    print()

    # Severity breakdown
    print("  By severity:")
    for sev in ["CRITICAL", "WARNING", "WATCH"]:
        sub = forecast_df[forecast_df["severity"] == sev]
        if len(sub) > 0:
            avg_p = sub["p_over_budget"].mean()
            avg_overrun = sub["projected_overrun_p50"].mean()
            print(f"    {sev:10s}: {len(sub):>4,} forecasts | "
                  f"Avg P(over)={avg_p:.1%} | "
                  f"Avg overrun P50=${avg_overrun:>10,.2f}")

    elapsed = time.time() - t_start
    print()
    print(f"Phase 3 completed in {elapsed:.1f}s")
    print("=" * 60)

    return forecast_df


if __name__ == "__main__":
    run_monte_carlo()
