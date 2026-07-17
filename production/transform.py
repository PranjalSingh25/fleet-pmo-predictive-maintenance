"""
Production -- Raw to Star Schema Transformation
==================================================
Transforms raw tables into the dimensional model:
    5 dimensions (dim_truck with SCD2, dim_date, dim_driver,
                  dim_facility, dim_route)
    4 fact tables (fact_truck_monthly, fact_maintenance,
                   fact_trip, fact_fuel)

Usage:
    python production/transform.py
"""
import sys
import os
import time

import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_dim_date(engine, start_year=2022, end_year=2024):
    """Generate and load the date dimension table."""
    dates = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31",
        freq="D",
    )

    dim_date = pd.DataFrame({
        "date_key": dates.strftime("%Y%m%d").astype(int),
        "full_date": dates,
        "year": dates.year,
        "quarter": dates.quarter,
        "month": dates.month,
        "month_name": dates.strftime("%B"),
        "day_of_month": dates.day,
        "day_of_week": dates.dayofweek,
        "day_name": dates.strftime("%A"),
        "is_weekend": dates.dayofweek >= 5,
        "week_of_year": dates.isocalendar().week.astype(int),
        "year_month": dates.strftime("%Y-%m"),
    })

    dim_date.to_sql("dim_date", engine, if_exists="replace", index=False)
    return len(dim_date)


def transform_dim_truck(engine):
    """
    Transform raw_trucks into dim_truck with SCD Type 2 columns.

    Initial load: all records get effective_date = acquisition_date,
    expiry_date = '9999-12-31', is_current = True.

    On subsequent loads, the SCD2 merge in incremental_load.py will
    handle status changes by closing old records and inserting new ones.
    """
    df = pd.read_sql("SELECT * FROM raw_trucks", engine)

    dim = pd.DataFrame({
        "truck_id": df["truck_id"],
        "unit_number": df["unit_number"],
        "make": df["make"],
        "model_year": df["model_year"],
        "vin": df["vin"],
        "acquisition_date": pd.to_datetime(df["acquisition_date"]),
        "acquisition_mileage": df["acquisition_mileage"],
        "fuel_type": df["fuel_type"],
        "tank_capacity_gal": df["tank_capacity_gallons"],
        "status": df["status"],
        "home_terminal": df["home_terminal"],
        # SCD Type 2 columns
        "effective_date": pd.to_datetime(df["acquisition_date"]),
        "expiry_date": pd.Timestamp("9999-12-31"),
        "is_current": True,
    })

    dim.to_sql("dim_truck", engine, if_exists="replace", index=False)
    return len(dim)


def transform_dim_driver(engine):
    """Transform raw_drivers into dim_driver."""
    df = pd.read_sql("SELECT * FROM raw_drivers", engine)

    dim = pd.DataFrame({
        "driver_id": df["driver_id"],
        "first_name": df["first_name"],
        "last_name": df["last_name"],
        "full_name": df["first_name"] + " " + df["last_name"],
        "hire_date": pd.to_datetime(df["hire_date"]),
        "termination_date": pd.to_datetime(df["termination_date"]),
        "license_number": df["license_number"],
        "license_state": df["license_state"],
        "date_of_birth": pd.to_datetime(df["date_of_birth"]),
        "home_terminal": df["home_terminal"],
        "employment_status": df["employment_status"],
        "cdl_class": df["cdl_class"],
        "years_experience": df["years_experience"],
    })

    dim.to_sql("dim_driver", engine, if_exists="replace", index=False)
    return len(dim)


def transform_dim_facility(engine):
    """Transform raw_facilities into dim_facility."""
    df = pd.read_sql("SELECT * FROM raw_facilities", engine)
    df.to_sql("dim_facility", engine, if_exists="replace", index=False)
    return len(df)


def transform_dim_route(engine):
    """Transform raw_routes into dim_route."""
    df = pd.read_sql("SELECT * FROM raw_routes", engine)
    df.to_sql("dim_route", engine, if_exists="replace", index=False)
    return len(df)


def transform_fact_truck_monthly(engine):
    """Transform raw_truck_utilization_metrics into fact_truck_monthly."""
    df = pd.read_sql("SELECT * FROM raw_truck_utilization_metrics", engine)
    df["month_dt"] = pd.to_datetime(df["month"])
    df["date_key"] = df["month_dt"].dt.strftime("%Y%m%d").astype(int)

    fact = df[[
        "truck_id", "date_key", "trips_completed", "total_miles",
        "total_revenue", "average_mpg", "maintenance_events",
        "maintenance_cost", "downtime_hours", "utilization_rate",
    ]].copy()

    fact.to_sql("fact_truck_monthly", engine, if_exists="replace", index=False)
    return len(fact)


def transform_fact_maintenance(engine):
    """Transform raw_maintenance_records into fact_maintenance."""
    df = pd.read_sql("SELECT * FROM raw_maintenance_records", engine)
    df["maint_dt"] = pd.to_datetime(df["maintenance_date"])
    df["date_key"] = df["maint_dt"].dt.strftime("%Y%m%d").astype(int)

    fact = pd.DataFrame({
        "maintenance_id": df["maintenance_id"],
        "truck_id": df["truck_id"],
        "date_key": df["date_key"],
        "maintenance_type": df["maintenance_type"],
        "odometer_reading": df["odometer_reading"],
        "labor_hours": df["labor_hours"],
        "labor_cost": df["labor_cost"],
        "parts_cost": df["parts_cost"],
        "total_cost": df["total_cost"],
        "facility_location": df["facility_location"],
        "downtime_hours": df["downtime_hours"],
        "service_description": df["service_description"],
    })

    fact.to_sql("fact_maintenance", engine, if_exists="replace", index=False)
    return len(fact)


def transform_fact_trip(engine):
    """Transform raw_trips into fact_trip."""
    df = pd.read_sql("SELECT * FROM raw_trips", engine)
    df["dispatch_dt"] = pd.to_datetime(df["dispatch_date"])
    df["date_key"] = df["dispatch_dt"].dt.strftime("%Y%m%d").astype(int)

    fact = pd.DataFrame({
        "trip_id": df["trip_id"],
        "load_id": df["load_id"],
        "driver_id": df["driver_id"],
        "truck_id": df["truck_id"],
        "trailer_id": df["trailer_id"],
        "date_key": df["date_key"],
        "actual_distance_miles": df["actual_distance_miles"],
        "actual_duration_hours": df["actual_duration_hours"],
        "fuel_gallons_used": df["fuel_gallons_used"],
        "average_mpg": df["average_mpg"],
        "idle_time_hours": df["idle_time_hours"],
        "trip_status": df["trip_status"],
    })

    fact.to_sql("fact_trip", engine, if_exists="replace", index=False)
    return len(fact)


def transform_fact_fuel(engine):
    """Transform raw_fuel_purchases into fact_fuel."""
    df = pd.read_sql("SELECT * FROM raw_fuel_purchases", engine)
    df["purchase_dt"] = pd.to_datetime(df["purchase_date"])
    df["date_key"] = df["purchase_dt"].dt.strftime("%Y%m%d").astype(int)

    fact = pd.DataFrame({
        "fuel_purchase_id": df["fuel_purchase_id"],
        "trip_id": df["trip_id"],
        "truck_id": df["truck_id"],
        "driver_id": df["driver_id"],
        "date_key": df["date_key"],
        "location_city": df["location_city"],
        "location_state": df["location_state"],
        "gallons": df["gallons"],
        "price_per_gallon": df["price_per_gallon"],
        "total_cost": df["total_cost"],
        "fuel_card_number": df["fuel_card_number"],
    })

    fact.to_sql("fact_fuel", engine, if_exists="replace", index=False)
    return len(fact)


def run_transform():
    """Execute the full transformation pipeline."""
    print("=" * 60)
    print("PRODUCTION -- TRANSFORM (Raw -> Star Schema)")
    print("=" * 60)

    engine = create_engine(config.get_db_url())
    results = {}

    transforms = [
        ("dim_date", lambda: generate_dim_date(engine)),
        ("dim_truck (SCD2)", lambda: transform_dim_truck(engine)),
        ("dim_driver", lambda: transform_dim_driver(engine)),
        ("dim_facility", lambda: transform_dim_facility(engine)),
        ("dim_route", lambda: transform_dim_route(engine)),
        ("fact_truck_monthly", lambda: transform_fact_truck_monthly(engine)),
        ("fact_maintenance", lambda: transform_fact_maintenance(engine)),
        ("fact_trip", lambda: transform_fact_trip(engine)),
        ("fact_fuel", lambda: transform_fact_fuel(engine)),
    ]

    for name, transform_fn in transforms:
        try:
            start = time.time()
            rows = transform_fn()
            duration = time.time() - start
            results[name] = rows
            print(f"  [OK] {name:.<40} {rows:>8,} rows ({duration:.1f}s)")
        except Exception as e:
            print(f"  [FAIL] {name:.<40} {str(e)[:60]}")

    print(f"\n{'=' * 60}")
    print("TRANSFORM COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Tables created: {len(results)}")
    print(f"  Total rows: {sum(results.values()):,}")

    return results


if __name__ == "__main__":
    run_transform()
