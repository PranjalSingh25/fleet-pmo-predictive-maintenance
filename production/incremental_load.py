"""
Production -- Incremental Data Load
======================================
Handles loading new months of data without full reprocessing.
Appends new records to raw tables, then re-runs transforms
for affected tables only.

Includes SCD Type 2 merge logic for dim_truck status changes.

Usage:
    python production/incremental_load.py --file new_month_data.csv --table raw_truck_utilization_metrics
"""
import sys
import os
import argparse

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def append_to_raw(engine, csv_path, table_name):
    """
    Append new records from a CSV to an existing raw table.
    Deduplicates based on the first column (assumed to be the ID).
    """
    new_data = pd.read_csv(csv_path)

    try:
        existing = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 1", engine)
        id_col = existing.columns[0]

        existing_ids = pd.read_sql(
            f"SELECT {id_col} FROM {table_name}", engine
        )[id_col].tolist()

        new_only = new_data[~new_data[id_col].isin(existing_ids)]
    except Exception:
        new_only = new_data

    if len(new_only) > 0:
        new_only.to_sql(table_name, engine, if_exists="append", index=False)

    return len(new_only)


def scd2_merge_truck(engine, new_trucks_df):
    """
    SCD Type 2 merge for dim_truck.

    When a truck's status changes:
    1. Close the current record (set expiry_date = today, is_current = FALSE)
    2. Insert a new record with the updated status (effective_date = today,
       expiry_date = '9999-12-31', is_current = TRUE)
    """
    today = pd.Timestamp.now().strftime("%Y-%m-%d")

    current_dim = pd.read_sql(
        "SELECT * FROM dim_truck WHERE is_current = 1", engine
    )

    if current_dim.empty:
        return 0

    merged = current_dim.merge(
        new_trucks_df[["truck_id", "status"]],
        on="truck_id",
        suffixes=("_old", "_new"),
    )

    changed = merged[merged["status_old"] != merged["status_new"]]

    if changed.empty:
        return 0

    changes_applied = 0
    for _, row in changed.iterrows():
        truck_id = row["truck_id"]

        # Close current record
        update_sql = text(
            "UPDATE dim_truck SET expiry_date = :today, is_current = 0 "
            "WHERE truck_id = :truck_id AND is_current = 1"
        )
        with engine.begin() as conn:
            conn.execute(update_sql, {"today": today, "truck_id": truck_id})

        # Insert new version
        new_row = current_dim[current_dim["truck_id"] == truck_id].iloc[0].copy()
        new_row["status"] = row["status_new"]
        new_row["effective_date"] = today
        new_row["expiry_date"] = "9999-12-31"
        new_row["is_current"] = True

        if "truck_key" in new_row.index:
            new_row = new_row.drop("truck_key")
        if "status_old" in new_row.index:
            new_row = new_row.drop("status_old")

        pd.DataFrame([new_row]).to_sql(
            "dim_truck", engine, if_exists="append", index=False
        )
        changes_applied += 1

    return changes_applied


def run_incremental_load(csv_path=None, table_name=None):
    """Execute an incremental load for a specific table."""
    print("=" * 60)
    print("PRODUCTION -- INCREMENTAL LOAD")
    print("=" * 60)

    engine = create_engine(config.get_db_url())

    if csv_path and table_name:
        print(f"\n  Loading: {csv_path}")
        print(f"  Target:  {table_name}")

        rows = append_to_raw(engine, csv_path, table_name)
        print(f"  New rows appended: {rows}")

        # If trucks data changed, run SCD2 merge
        if table_name == "raw_trucks":
            new_trucks = pd.read_csv(csv_path)
            scd_changes = scd2_merge_truck(engine, new_trucks)
            print(f"  SCD2 status changes applied: {scd_changes}")
    else:
        print("\n  No file specified. Usage:")
        print("  python production/incremental_load.py "
              "--file path/to/data.csv --table raw_table_name")

    print(f"\n{'=' * 60}")
    print("INCREMENTAL LOAD COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental data loader")
    parser.add_argument("--file", help="Path to CSV file to load")
    parser.add_argument("--table", help="Target raw table name")
    args = parser.parse_args()
    run_incremental_load(args.file, args.table)
