"""
Tests for Production -- Data Quality Checks
Tests the quality check framework itself using a small SQLite DB.
"""
import sys
import os

import pandas as pd
import pytest
from sqlalchemy import create_engine

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "production"
))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_quality import DataQualityChecker


@pytest.fixture
def test_engine(tmp_path):
    """Create a test SQLite database with sample star schema tables."""
    db_path = tmp_path / "test_fleetops.db"
    engine = create_engine(f"sqlite:///{db_path}")

    # dim_truck
    pd.DataFrame({
        "truck_id": ["T1", "T2", "T3"],
        "make": ["Peterbilt", "Kenworth", "Freightliner"],
        "model_year": [2016, 2018, 2020],
        "status": ["Active", "Active", "Maintenance"],
        "is_current": [True, True, True],
    }).to_sql("dim_truck", engine, index=False)

    # dim_driver
    pd.DataFrame({
        "driver_id": ["D1", "D2"],
        "first_name": ["John", "Jane"],
    }).to_sql("dim_driver", engine, index=False)

    # dim_date
    pd.DataFrame({
        "date_key": [20220101, 20220201],
        "full_date": ["2022-01-01", "2022-02-01"],
    }).to_sql("dim_date", engine, index=False)

    # dim_facility
    pd.DataFrame({
        "facility_id": ["F1"],
        "facility_name": ["Test Hub"],
    }).to_sql("dim_facility", engine, index=False)

    # dim_route
    pd.DataFrame({
        "route_id": ["R1"],
        "origin_city": ["Atlanta"],
    }).to_sql("dim_route", engine, index=False)

    # fact_truck_monthly
    pd.DataFrame({
        "truck_id": ["T1", "T2"],
        "date_key": [20220101, 20220201],
        "maintenance_cost": [1500.0, 2000.0],
        "utilization_rate": [0.8, 0.9],
        "average_mpg": [6.5, 7.0],
    }).to_sql("fact_truck_monthly", engine, index=False)

    # fact_maintenance
    pd.DataFrame({
        "maintenance_id": ["M1", "M2"],
        "truck_id": ["T1", "T2"],
        "date_key": [20220101, 20220201],
        "total_cost": [1500.0, 2000.0],
        "downtime_hours": [20.0, 15.0],
    }).to_sql("fact_maintenance", engine, index=False)

    # fact_trip
    pd.DataFrame({
        "trip_id": ["TR1"],
        "truck_id": ["T1"],
        "driver_id": ["D1"],
        "date_key": [20220101],
    }).to_sql("fact_trip", engine, index=False)

    # fact_fuel
    pd.DataFrame({
        "fuel_purchase_id": ["F1"],
        "truck_id": ["T1"],
        "date_key": [20220101],
        "price_per_gallon": [3.50],
        "gallons": [100.0],
    }).to_sql("fact_fuel", engine, index=False)

    return engine


class TestNullChecks:
    def test_no_null_pks(self, test_engine):
        checker = DataQualityChecker(test_engine)
        checker.check_null_primary_keys()
        for result in checker.results:
            assert result["status"] == "PASS", f"Null PK in {result['table']}"


class TestReferentialIntegrity:
    def test_valid_references(self, test_engine):
        checker = DataQualityChecker(test_engine)
        checker.check_referential_integrity()
        for result in checker.results:
            assert result["status"] == "PASS", (
                f"RI violation: {result['table']} - {result['details']}"
            )

    def test_detects_orphans(self, test_engine):
        """Insert an orphaned record and verify it's caught."""
        pd.DataFrame({
            "trip_id": ["TR_ORPHAN"],
            "truck_id": ["T_NONEXISTENT"],
            "driver_id": ["D_NONEXISTENT"],
            "date_key": [20220101],
        }).to_sql("fact_trip", test_engine, if_exists="append", index=False)

        checker = DataQualityChecker(test_engine)
        checker.check_referential_integrity()
        trip_checks = [r for r in checker.results if "fact_trip" in r["table"]]
        failed = [r for r in trip_checks if r["status"] == "FAIL"]
        assert len(failed) > 0, "Should detect orphaned truck_id or driver_id"


class TestRangeValidation:
    def test_valid_ranges(self, test_engine):
        checker = DataQualityChecker(test_engine)
        checker.check_numeric_ranges()
        cost_checks = [
            r for r in checker.results
            if "maintenance_cost" in r.get("table", "")
        ]
        for result in cost_checks:
            assert result["status"] == "PASS", f"Range violation: {result['details']}"
