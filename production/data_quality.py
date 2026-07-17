"""
Production -- Data Quality Checks
====================================
Validates data integrity across the star schema:
    - Null checks on primary keys
    - Referential integrity between facts and dimensions
    - Range validation on numeric fields
    - Freshness monitoring

Usage:
    python production/data_quality.py
"""
import sys
import os

import pandas as pd
from sqlalchemy import create_engine, inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DataQualityChecker:
    """Runs data quality checks against the star schema."""

    def __init__(self, engine):
        self.engine = engine
        self.results = []

    def _record(self, check_name, table, passed, details=""):
        self.results.append({
            "check": check_name,
            "table": table,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        })

    def _table_exists(self, table_name):
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()

    # ── Null Checks ──

    def check_null_primary_keys(self):
        """Primary keys must never be null."""
        pk_checks = {
            "dim_truck": "truck_id",
            "dim_driver": "driver_id",
            "dim_facility": "facility_id",
            "dim_route": "route_id",
            "fact_truck_monthly": "truck_id",
            "fact_maintenance": "maintenance_id",
            "fact_trip": "trip_id",
            "fact_fuel": "fuel_purchase_id",
        }

        for table, pk_col in pk_checks.items():
            if not self._table_exists(table):
                self._record("null_pk", table, False, f"Table {table} does not exist")
                continue

            query = f"SELECT COUNT(*) as cnt FROM {table} WHERE {pk_col} IS NULL"
            result = pd.read_sql(query, self.engine)
            null_count = result["cnt"].iloc[0]
            self._record(
                "null_pk", table, null_count == 0,
                f"{null_count} null values in {pk_col}"
            )

    # ── Referential Integrity ──

    def check_referential_integrity(self):
        """Fact table foreign keys must reference existing dimension records."""
        fk_checks = [
            ("fact_truck_monthly", "truck_id", "dim_truck", "truck_id"),
            ("fact_maintenance", "truck_id", "dim_truck", "truck_id"),
            ("fact_trip", "truck_id", "dim_truck", "truck_id"),
            ("fact_trip", "driver_id", "dim_driver", "driver_id"),
            ("fact_fuel", "truck_id", "dim_truck", "truck_id"),
        ]

        for fact_table, fk_col, dim_table, dim_col in fk_checks:
            if not self._table_exists(fact_table) or not self._table_exists(dim_table):
                self._record(
                    "referential_integrity", f"{fact_table}.{fk_col}",
                    False, "Table(s) missing"
                )
                continue

            query = f"""
                SELECT COUNT(DISTINCT f.{fk_col}) as orphan_count
                FROM {fact_table} f
                LEFT JOIN {dim_table} d ON f.{fk_col} = d.{dim_col}
                WHERE d.{dim_col} IS NULL
                AND f.{fk_col} IS NOT NULL
            """
            result = pd.read_sql(query, self.engine)
            orphans = result["orphan_count"].iloc[0]
            self._record(
                "referential_integrity",
                f"{fact_table}.{fk_col} -> {dim_table}",
                orphans == 0,
                f"{orphans} orphaned records"
            )

    # ── Range Validation ──

    def check_numeric_ranges(self):
        """Numeric fields must be within expected business ranges."""
        range_checks = [
            ("fact_truck_monthly", "maintenance_cost", 0, 15000,
             "Maintenance cost should be $0-$15K"),
            ("fact_truck_monthly", "utilization_rate", 0, 2.0,
             "Utilization rate should be 0-200%"),
            ("fact_truck_monthly", "average_mpg", 0, 15,
             "MPG should be 0-15 for heavy trucks"),
            ("fact_maintenance", "total_cost", 0, 15000,
             "Single maintenance event should be $0-$15K"),
            ("fact_maintenance", "downtime_hours", 0, 720,
             "Downtime should be 0-720 hrs (30 days)"),
            ("fact_fuel", "price_per_gallon", 1.0, 8.0,
             "Fuel price should be $1-$8/gal"),
            ("fact_fuel", "gallons", 0, 500,
             "Fuel fill should be 0-500 gal"),
        ]

        for table, col, min_val, max_val, description in range_checks:
            if not self._table_exists(table):
                self._record("range", f"{table}.{col}", False, "Table missing")
                continue

            query = f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN {col} < {min_val} OR {col} > {max_val}
                         THEN 1 ELSE 0 END) as out_of_range,
                    MIN({col}) as min_val,
                    MAX({col}) as max_val
                FROM {table}
                WHERE {col} IS NOT NULL
            """
            result = pd.read_sql(query, self.engine)
            oor = result["out_of_range"].iloc[0]
            actual_min = result["min_val"].iloc[0]
            actual_max = result["max_val"].iloc[0]
            self._record(
                "range", f"{table}.{col}",
                oor == 0,
                f"{oor} out of range [{min_val}, {max_val}]. "
                f"Actual: [{actual_min:.2f}, {actual_max:.2f}]"
            )

    # ── Freshness ──

    def check_data_freshness(self):
        """Check that data is reasonably recent."""
        if not self._table_exists("dim_date") or not self._table_exists("fact_truck_monthly"):
            self._record("freshness", "fact_truck_monthly", False, "Table missing")
            return

        query = """
            SELECT MAX(d.full_date) as latest_date
            FROM fact_truck_monthly f
            JOIN dim_date d ON f.date_key = d.date_key
        """
        result = pd.read_sql(query, self.engine)
        latest = result["latest_date"].iloc[0]
        self._record(
            "freshness", "fact_truck_monthly",
            latest is not None,
            f"Latest data: {latest}"
        )

    # ── Row Count ──

    def check_row_counts(self):
        """Validate tables have reasonable row counts."""
        expected_counts = {
            "dim_truck": (90, 200),
            "dim_driver": (100, 300),
            "dim_date": (1000, 1200),
            "fact_truck_monthly": (3000, 4000),
            "fact_maintenance": (2500, 3500),
            "fact_trip": (80000, 100000),
            "fact_fuel": (150000, 250000),
        }

        for table, (min_rows, max_rows) in expected_counts.items():
            if not self._table_exists(table):
                self._record("row_count", table, False, "Table missing")
                continue

            query = f"SELECT COUNT(*) as cnt FROM {table}"
            result = pd.read_sql(query, self.engine)
            count = result["cnt"].iloc[0]
            in_range = min_rows <= count <= max_rows
            self._record(
                "row_count", table, in_range,
                f"{count:,} rows (expected {min_rows:,}-{max_rows:,})"
            )

    def run_all_checks(self):
        """Run the full data quality suite."""
        self.check_null_primary_keys()
        self.check_referential_integrity()
        self.check_numeric_ranges()
        self.check_data_freshness()
        self.check_row_counts()
        return self.results


def run_data_quality():
    """Execute all data quality checks and print results."""
    print("=" * 60)
    print("PRODUCTION -- DATA QUALITY CHECKS")
    print("=" * 60)

    engine = create_engine(config.get_db_url())
    checker = DataQualityChecker(engine)
    results = checker.run_all_checks()

    passes = sum(1 for r in results if r["status"] == "PASS")
    fails = sum(1 for r in results if r["status"] == "FAIL")

    for r in results:
        status_marker = "OK  " if r["status"] == "PASS" else "FAIL"
        print(f"  [{status_marker}] {r['check']:.<25} {r['table']:.<40} {r['details']}")

    print(f"\n{'=' * 60}")
    print("DATA QUALITY COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Passed: {passes}/{len(results)}")
    print(f"  Failed: {fails}/{len(results)}")

    return results


if __name__ == "__main__":
    run_data_quality()
