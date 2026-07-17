"""
Tests for Phase 1 -- Detection
"""
import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from detection import (
    compute_baseline_budgets,
    compute_ytd_overruns,
    merge_flags,
)


@pytest.fixture
def sample_utilization():
    """Create sample truck utilization data for testing."""
    data = []
    # Truck A: 2022 baseline = $12,000 ($1,000/month)
    for m in range(1, 13):
        data.append({
            "truck_id": "TRK_A",
            "month": f"2022-{m:02d}-01",
            "maintenance_cost": 1000.0,
            "total_miles": 5000,
            "trips_completed": 10,
            "total_revenue": 20000,
            "average_mpg": 6.5,
            "maintenance_events": 1,
            "downtime_hours": 10,
            "utilization_rate": 0.8,
        })
    # 2023: overrun months
    for m in range(1, 13):
        cost = 1200.0 if m <= 6 else 1500.0  # 20% -> 50% over
        data.append({
            "truck_id": "TRK_A",
            "month": f"2023-{m:02d}-01",
            "maintenance_cost": cost,
            "total_miles": 5000,
            "trips_completed": 10,
            "total_revenue": 20000,
            "average_mpg": 6.5,
            "maintenance_events": 1,
            "downtime_hours": 10,
            "utilization_rate": 0.8,
        })
    df = pd.DataFrame(data)
    df["month"] = pd.to_datetime(df["month"])
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month
    return df


class TestBaselineBudgets:
    def test_budget_calculation(self, sample_utilization):
        budgets = compute_baseline_budgets(sample_utilization)
        assert len(budgets) == 1
        assert budgets.iloc[0]["annual_budget"] == 12000.0

    def test_budget_columns(self, sample_utilization):
        budgets = compute_baseline_budgets(sample_utilization)
        assert "truck_id" in budgets.columns
        assert "annual_budget" in budgets.columns


class TestYTDOverruns:
    def test_overrun_detection(self, sample_utilization):
        budgets = compute_baseline_budgets(sample_utilization)
        overruns = compute_ytd_overruns(sample_utilization, budgets)
        assert len(overruns) > 0

    def test_critical_threshold(self, sample_utilization):
        budgets = compute_baseline_budgets(sample_utilization)
        overruns = compute_ytd_overruns(sample_utilization, budgets)
        critical = overruns[overruns["severity"] == "CRITICAL"]
        for _, row in critical.iterrows():
            assert row["overrun_pct"] > config.CRITICAL_THRESHOLD_PCT

    def test_warning_threshold(self, sample_utilization):
        budgets = compute_baseline_budgets(sample_utilization)
        overruns = compute_ytd_overruns(sample_utilization, budgets)
        warnings = overruns[overruns["severity"] == "WARNING"]
        for _, row in warnings.iterrows():
            assert row["overrun_pct"] > config.WARNING_THRESHOLD_PCT
            assert row["overrun_pct"] <= config.CRITICAL_THRESHOLD_PCT

    def test_no_false_negatives_below_threshold(self, sample_utilization):
        """Unflagged records should be at or below the warning threshold."""
        budgets = compute_baseline_budgets(sample_utilization)
        overruns = compute_ytd_overruns(sample_utilization, budgets)
        unflagged = overruns[overruns["severity"].isna()]
        for _, row in unflagged.iterrows():
            assert row["overrun_pct"] <= config.WARNING_THRESHOLD_PCT

    def test_ytd_accumulates(self, sample_utilization):
        """YTD actual should always be >= previous month's YTD."""
        budgets = compute_baseline_budgets(sample_utilization)
        overruns = compute_ytd_overruns(sample_utilization, budgets)
        for truck_id in overruns["truck_id"].unique():
            truck_data = overruns[overruns["truck_id"] == truck_id].sort_values("current_month")
            for year in truck_data["year"].unique():
                year_data = truck_data[truck_data["year"] == year]
                ytd_values = year_data["ytd_actual"].tolist()
                assert ytd_values == sorted(ytd_values), "YTD should monotonically increase"


class TestMergeFlags:
    def test_critical_overrides_watch(self):
        """CRITICAL and WARNING should take priority over WATCH."""
        overruns = pd.DataFrame({
            "truck_id": ["T1", "T2"],
            "year": [2023, 2023],
            "current_month": [6, 6],
            "severity": ["CRITICAL", None],
            "overrun_pct": [15.0, 2.0],
            "overrun_dollars": [1500, 200],
            "ytd_budget": [10000, 10000],
            "ytd_actual": [11500, 10200],
            "month_date": pd.to_datetime(["2023-06-01", "2023-06-01"]),
            "annual_budget": [20000, 20000],
        })
        fuel_flags = pd.DataFrame({
            "truck_id": ["T1", "T2"],
            "year": [2023, 2023],
            "month_num": [6, 6],
            "fuel_trend_alert": [True, True],
            "fuel_trend_pct": [15.0, 15.0],
            "fuel_cost_per_mile": [0.5, 0.5],
            "fuel_cpm_rolling_3mo": [0.55, 0.55],
            "baseline_fuel_cpm": [0.45, 0.45],
            "month": pd.to_datetime(["2023-06-01", "2023-06-01"]),
        })
        merged = merge_flags(overruns, fuel_flags)
        assert merged.loc[merged["truck_id"] == "T1", "severity"].iloc[0] == "CRITICAL"
        assert merged.loc[merged["truck_id"] == "T2", "severity"].iloc[0] == "WATCH"
