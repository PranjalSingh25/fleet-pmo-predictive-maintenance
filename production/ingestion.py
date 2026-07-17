"""
Production -- CSV to Database Ingestion
=========================================
Loads all 15 source CSVs into the database as raw tables.
Supports both PostgreSQL and SQLite via config toggle.

Usage:
    python production/ingestion.py
"""
import sys
import os
import time

import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Mapping of CSV files to raw table names
CSV_TABLE_MAP = {
    config.TRUCKS_FILE: "raw_trucks",
    config.DRIVERS_FILE: "raw_drivers",
    config.TRAILERS_FILE: "raw_trailers",
    config.CUSTOMERS_FILE: "raw_customers",
    config.FACILITIES_FILE: "raw_facilities",
    config.ROUTES_FILE: "raw_routes",
    config.LOADS_FILE: "raw_loads",
    config.TRIPS_FILE: "raw_trips",
    config.FUEL_PURCHASES_FILE: "raw_fuel_purchases",
    config.MAINTENANCE_RECORDS_FILE: "raw_maintenance_records",
    config.DELIVERY_EVENTS_FILE: "raw_delivery_events",
    config.SAFETY_INCIDENTS_FILE: "raw_safety_incidents",
    config.DRIVER_MONTHLY_METRICS_FILE: "raw_driver_monthly_metrics",
    config.TRUCK_UTILIZATION_FILE: "raw_truck_utilization_metrics",
}


def ingest_csv(engine, csv_path, table_name, audit_log):
    """
    Load a single CSV into the database, replacing existing data.

    Returns:
        Number of rows loaded.
    """
    start = time.time()

    df = pd.read_csv(csv_path)
    row_count = len(df)

    df.to_sql(table_name, engine, if_exists="replace", index=False)

    duration = time.time() - start

    audit_log.append({
        "stage_name": "ingestion",
        "status": "SUCCESS",
        "rows_processed": row_count,
        "source_file": str(csv_path.name),
        "duration_seconds": round(duration, 2),
        "notes": f"Loaded into {table_name}",
    })

    return row_count


def log_audit(engine, audit_log):
    """Write audit records to pipeline_runs table if it exists."""
    try:
        audit_df = pd.DataFrame(audit_log)
        audit_df["run_timestamp"] = pd.Timestamp.now()
        audit_df.to_sql("pipeline_runs", engine, if_exists="append", index=False)
    except Exception:
        pass  # Audit table may not exist yet on first run


def run_ingestion():
    """
    Execute the full ingestion pipeline.

    Returns:
        dict with table_name -> row_count
    """
    print("=" * 60)
    print("PRODUCTION -- INGESTION")
    print("=" * 60)

    engine = create_engine(config.get_db_url())
    audit_log = []
    results = {}
    total_rows = 0

    print(f"\n  Database: {config.get_db_url().split('///')[0]}...")
    print(f"  Source: {config.DATA_DIR}")
    print(f"  Tables to load: {len(CSV_TABLE_MAP)}\n")

    for csv_path, table_name in CSV_TABLE_MAP.items():
        if not csv_path.exists():
            print(f"  [SKIP] {csv_path.name} -- file not found")
            continue

        try:
            rows = ingest_csv(engine, csv_path, table_name, audit_log)
            results[table_name] = rows
            total_rows += rows
            print(f"  [OK] {table_name:.<40} {rows:>8,} rows")
        except Exception as e:
            print(f"  [FAIL] {table_name:.<40} {str(e)[:50]}")
            audit_log.append({
                "stage_name": "ingestion",
                "status": "FAILED",
                "source_file": str(csv_path.name),
                "notes": str(e)[:200],
            })

    log_audit(engine, audit_log)

    print(f"\n{'=' * 60}")
    print("INGESTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Tables loaded: {len(results)}")
    print(f"  Total rows: {total_rows:,}")

    return results


if __name__ == "__main__":
    run_ingestion()
