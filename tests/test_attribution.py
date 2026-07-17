"""
Tests for Phase 2 -- Attribution
"""
import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Import attribution functions - handle different module structures
try:
    from attribution import run_attribution
except ImportError:
    run_attribution = None


class TestAttributionOutputs:
    """Test that attribution outputs exist and have correct structure."""

    def test_attribution_report_exists(self):
        path = config.OUTPUT_DIR / "attribution_report_all.csv"
        assert path.exists(), "Attribution report not found. Run attribution.py first."

    def test_attribution_report_columns(self):
        path = config.OUTPUT_DIR / "attribution_report_all.csv"
        if not path.exists():
            pytest.skip("Attribution report not generated yet")
        df = pd.read_csv(path)
        required_cols = ["truck_id", "year", "month", "root_cause", "evidence_summary"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_root_cause_labels_valid(self):
        path = config.OUTPUT_DIR / "attribution_report_all.csv"
        if not path.exists():
            pytest.skip("Attribution report not generated yet")
        df = pd.read_csv(path)
        valid_labels = {
            "equipment_age_failure",
            "extended_downtime",
            "cluster_overspend",
            "routine_overspend",
        }
        actual_labels = set(df["root_cause"].unique())
        assert actual_labels.issubset(valid_labels), (
            f"Invalid root cause labels: {actual_labels - valid_labels}"
        )

    def test_no_null_root_causes(self):
        path = config.OUTPUT_DIR / "attribution_report_all.csv"
        if not path.exists():
            pytest.skip("Attribution report not generated yet")
        df = pd.read_csv(path)
        assert df["root_cause"].notna().all(), "Found null root_cause values"

    def test_evidence_not_empty(self):
        path = config.OUTPUT_DIR / "attribution_report_all.csv"
        if not path.exists():
            pytest.skip("Attribution report not generated yet")
        df = pd.read_csv(path)
        assert (df["evidence_summary"].str.len() > 0).all(), "Found empty evidence summaries"

    def test_equipment_age_only_old_trucks(self):
        """Equipment age failures should only apply to trucks <= cutoff year."""
        path = config.OUTPUT_DIR / "attribution_report_all.csv"
        if not path.exists():
            pytest.skip("Attribution report not generated yet")
        df = pd.read_csv(path)
        age_failures = df[df["root_cause"] == "equipment_age_failure"]
        if len(age_failures) > 0 and "truck_model_year" in df.columns:
            over_cutoff = age_failures[
                age_failures["truck_model_year"] > config.EQUIPMENT_AGE_CUTOFF_YEAR
            ]
            assert len(over_cutoff) == 0, (
                f"Found equipment_age_failure for trucks newer than {config.EQUIPMENT_AGE_CUTOFF_YEAR}"
            )
