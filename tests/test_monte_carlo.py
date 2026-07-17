"""
Tests for Phase 3 -- Monte Carlo Forecasting
"""
import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class TestMonteCarloOutputs:
    """Test that Monte Carlo forecast outputs are correct."""

    def test_forecast_file_exists(self):
        path = config.OUTPUT_DIR / "forecast_all.csv"
        if not path.exists():
            pytest.skip("Forecast not generated yet")
        assert path.exists(), "Forecast file not found. Run monte_carlo.py first."

    def test_forecast_columns(self):
        path = config.OUTPUT_DIR / "forecast_all.csv"
        if not path.exists():
            pytest.skip("Forecast not generated yet")
        df = pd.read_csv(path)
        required = [
            "truck_id", "year", "forecast_month", "ytd_actual",
            "annual_budget", "remaining_months", "p_over_budget",
            "projected_p10", "projected_p50", "projected_p90",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_probability_range(self):
        """P(over_budget) must be between 0 and 1."""
        path = config.OUTPUT_DIR / "forecast_all.csv"
        if not path.exists():
            pytest.skip("Forecast not generated yet")
        df = pd.read_csv(path)
        assert (df["p_over_budget"] >= 0).all(), "P(over_budget) below 0"
        assert (df["p_over_budget"] <= 1).all(), "P(over_budget) above 1"

    def test_percentile_ordering(self):
        """P10 <= P50 <= P90 must always hold."""
        path = config.OUTPUT_DIR / "forecast_all.csv"
        if not path.exists():
            pytest.skip("Forecast not generated yet")
        df = pd.read_csv(path)
        assert (df["projected_p10"] <= df["projected_p50"]).all(), "P10 > P50"
        assert (df["projected_p50"] <= df["projected_p90"]).all(), "P50 > P90"

    def test_remaining_months_valid(self):
        """Remaining months should be 1-11 (month 12 has no forecast)."""
        path = config.OUTPUT_DIR / "forecast_all.csv"
        if not path.exists():
            pytest.skip("Forecast not generated yet")
        df = pd.read_csv(path)
        assert (df["remaining_months"] >= 1).all(), "Remaining months < 1"
        assert (df["remaining_months"] <= 11).all(), "Remaining months > 11"

    def test_confidence_interval_widens(self):
        """
        For a given truck-year, the confidence interval (P90-P10)
        should generally be wider earlier in the year (more remaining months).
        We test this by checking correlation is positive.
        """
        path = config.OUTPUT_DIR / "forecast_all.csv"
        if not path.exists():
            pytest.skip("Forecast not generated yet")
        df = pd.read_csv(path)
        df["ci_width"] = df["projected_p90"] - df["projected_p10"]
        # More remaining months -> wider CI (positive correlation)
        corr = df["remaining_months"].corr(df["ci_width"])
        assert corr > 0, (
            f"Expected positive correlation between remaining_months and "
            f"CI width, got {corr:.3f}"
        )

    def test_projections_above_ytd(self):
        """All projections should be >= YTD actual (can't un-spend money)."""
        path = config.OUTPUT_DIR / "forecast_all.csv"
        if not path.exists():
            pytest.skip("Forecast not generated yet")
        df = pd.read_csv(path)
        assert (df["projected_p10"] >= df["ytd_actual"] * 0.99).all(), (
            "P10 projection below YTD actual"
        )
