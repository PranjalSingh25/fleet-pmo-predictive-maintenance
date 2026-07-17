"""
FleetOps Anomaly Detection - Phase 5: Feature Engineering
==========================================================
Builds a feature matrix at the (truck_id, month) grain using
cross-source joins from the Logistics Operations Database.

Output: outputs/feature_matrix.csv
"""
import pandas as pd
import numpy as np
import config


def build_features():
    """Build the ML feature matrix and save to CSV."""

    print("=" * 60)
    print("Phase 5 - Feature Engineering")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load source data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading source data ...")

    util = pd.read_csv(config.TRUCK_UTILIZATION_FILE)
    trucks = pd.read_csv(config.TRUCKS_FILE)
    trips = pd.read_csv(config.TRIPS_FILE)
    fuel = pd.read_csv(config.FUEL_PURCHASES_FILE)
    safety = pd.read_csv(config.SAFETY_INCIDENTS_FILE)

    print(f"  truck_utilization_metrics : {util.shape}")
    print(f"  trucks                    : {trucks.shape}")
    print(f"  trips                     : {trips.shape}")
    print(f"  fuel_purchases            : {fuel.shape}")
    print(f"  safety_incidents          : {safety.shape}")

    # Ensure month is datetime
    util["month"] = pd.to_datetime(util["month"])

    # Sort for rolling calculations
    util = util.sort_values(["truck_id", "month"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Rolling / lag features from truck_utilization_metrics
    # ------------------------------------------------------------------
    print("[2/6] Computing rolling and lag features ...")

    # Group by truck for rolling window calculations
    grouped = util.groupby("truck_id")

    # Rolling averages (use shift(1) so we don't include current month)
    util["cost_rolling_3mo"] = grouped["maintenance_cost"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    util["cost_rolling_6mo"] = grouped["maintenance_cost"].transform(
        lambda x: x.shift(1).rolling(window=6, min_periods=1).mean()
    )
    util["downtime_rolling_3mo"] = grouped["downtime_hours"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    util["maint_event_count_3mo"] = grouped["maintenance_events"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).sum()
    )

    # Lag features
    util["cost_lag_1mo"] = grouped["maintenance_cost"].transform(
        lambda x: x.shift(1)
    )
    util["cost_lag_2mo"] = grouped["maintenance_cost"].transform(
        lambda x: x.shift(2)
    )

    # Utilization rate (current month - already present)
    # Total revenue monthly
    util["total_revenue_monthly"] = util["total_revenue"]

    # ------------------------------------------------------------------
    # 3. Truck age from trucks table
    # ------------------------------------------------------------------
    print("[3/6] Computing truck age ...")

    trucks_lookup = trucks[["truck_id", "model_year"]].drop_duplicates()
    util = util.merge(trucks_lookup, on="truck_id", how="left")

    # Extract year from month column to compute age
    util["year"] = util["month"].dt.year
    util["truck_age_years"] = util["year"] - util["model_year"]

    # Month of year for seasonality
    util["month_of_year"] = util["month"].dt.month

    # ------------------------------------------------------------------
    # 4. Fuel cost per mile from trips + fuel_purchases
    # ------------------------------------------------------------------
    print("[4/6] Computing fuel cost per mile (this may take a moment) ...")

    # Parse dates
    trips["dispatch_date"] = pd.to_datetime(trips["dispatch_date"])
    fuel["purchase_date"] = pd.to_datetime(fuel["purchase_date"])

    # Create month column for aggregation
    trips["trip_month"] = trips["dispatch_date"].dt.to_period("M")
    fuel["fuel_month"] = fuel["purchase_date"].dt.to_period("M")

    # Aggregate trips by truck and month
    trip_agg = trips.groupby(["truck_id", "trip_month"]).agg(
        trip_distance_sum=("actual_distance_miles", "sum"),
        trip_count_monthly=("trip_id", "count"),
        avg_trip_distance=("actual_distance_miles", "mean"),
    ).reset_index()
    trip_agg["trip_month"] = trip_agg["trip_month"].dt.to_timestamp()

    # Aggregate fuel by truck and month
    fuel_agg = fuel.groupby(["truck_id", "fuel_month"]).agg(
        fuel_cost_sum=("total_cost", "sum"),
    ).reset_index()
    fuel_agg["fuel_month"] = fuel_agg["fuel_month"].dt.to_timestamp()

    # Merge trip and fuel aggregates
    trip_fuel = trip_agg.merge(
        fuel_agg,
        left_on=["truck_id", "trip_month"],
        right_on=["truck_id", "fuel_month"],
        how="left",
    )
    trip_fuel["fuel_cost_per_mile"] = (
        trip_fuel["fuel_cost_sum"] / trip_fuel["trip_distance_sum"]
    )
    # Handle division by zero / missing
    trip_fuel["fuel_cost_per_mile"] = trip_fuel["fuel_cost_per_mile"].replace(
        [np.inf, -np.inf], np.nan
    )

    # Merge into main dataframe
    util = util.merge(
        trip_fuel[["truck_id", "trip_month", "fuel_cost_per_mile",
                   "trip_count_monthly", "avg_trip_distance"]],
        left_on=["truck_id", "month"],
        right_on=["truck_id", "trip_month"],
        how="left",
    )

    # ------------------------------------------------------------------
    # 5. Safety incident flag (any incident in last 3 months)
    # ------------------------------------------------------------------
    print("[5/6] Computing safety incident flags ...")

    safety["incident_date"] = pd.to_datetime(safety["incident_date"])
    safety["incident_month"] = safety["incident_date"].dt.to_period("M").dt.to_timestamp()

    # Get unique truck-month combinations with incidents
    incident_months = safety.groupby(["truck_id", "incident_month"]).size().reset_index(
        name="incident_count"
    )

    # For each (truck_id, month) in util, check if there was any incident
    # in the current or previous 2 months (3-month window)
    def compute_safety_flag(df):
        """Compute 3-month rolling safety incident flag per truck."""
        all_months = df[["truck_id", "month"]].drop_duplicates()
        results = []
        for _, row in all_months.iterrows():
            tid = row["truck_id"]
            m = row["month"]
            # 3-month lookback window: current month and 2 prior months
            start = m - pd.DateOffset(months=2)
            truck_incidents = incident_months[
                (incident_months["truck_id"] == tid)
                & (incident_months["incident_month"] >= start)
                & (incident_months["incident_month"] <= m)
            ]
            results.append({
                "truck_id": tid,
                "month": m,
                "safety_incident_flag": 1 if len(truck_incidents) > 0 else 0,
            })
        return pd.DataFrame(results)

    # Vectorized approach for better performance
    # Create a cross-join of truck-month with incident months
    util_keys = util[["truck_id", "month"]].drop_duplicates().copy()
    util_keys["_key"] = 1

    # For each util row, check incidents in [month-2, month]
    safety_flags = []
    for truck_id in util_keys["truck_id"].unique():
        truck_util = util_keys[util_keys["truck_id"] == truck_id].copy()
        truck_incidents = incident_months[
            incident_months["truck_id"] == truck_id
        ]["incident_month"].values

        if len(truck_incidents) == 0:
            truck_util["safety_incident_flag"] = 0
            safety_flags.append(truck_util[["truck_id", "month", "safety_incident_flag"]])
            continue

        flags = []
        for _, row in truck_util.iterrows():
            m = row["month"]
            start = m - pd.DateOffset(months=2)
            has_incident = any(
                (inc >= start) and (inc <= m)
                for inc in pd.to_datetime(truck_incidents)
            )
            flags.append(1 if has_incident else 0)
        truck_util["safety_incident_flag"] = flags
        safety_flags.append(truck_util[["truck_id", "month", "safety_incident_flag"]])

    safety_flag_df = pd.concat(safety_flags, ignore_index=True)
    util = util.merge(safety_flag_df, on=["truck_id", "month"], how="left")

    # ------------------------------------------------------------------
    # 6. Target variable: high_cost_next_month
    # ------------------------------------------------------------------
    print("[6/6] Creating target variable ...")

    # Compute the threshold (75th percentile of all monthly maintenance costs)
    cost_threshold = util["maintenance_cost"].quantile(
        config.HIGH_COST_PERCENTILE / 100.0
    )
    print(f"  Maintenance cost {config.HIGH_COST_PERCENTILE}th percentile: ${cost_threshold:,.2f}")

    # Shift maintenance_cost forward by 1 month per truck to get "next month's cost"
    util["maintenance_cost_next_month"] = util.groupby("truck_id")[
        "maintenance_cost"
    ].shift(-1)

    # Binary target
    util["high_cost_next_month"] = (
        util["maintenance_cost_next_month"] > cost_threshold
    ).astype(float)

    # Drop rows where target is NaN (last month per truck)
    rows_before = len(util)
    util = util.dropna(subset=["high_cost_next_month"]).reset_index(drop=True)
    util["high_cost_next_month"] = util["high_cost_next_month"].astype(int)
    rows_dropped = rows_before - len(util)
    print(f"  Dropped {rows_dropped} rows with NaN target (last month per truck)")

    # ------------------------------------------------------------------
    # Select final feature columns
    # ------------------------------------------------------------------
    feature_cols = [
        "truck_id", "month", "year",
        "cost_rolling_3mo", "cost_rolling_6mo",
        "downtime_rolling_3mo", "utilization_rate",
        "maint_event_count_3mo",
        "cost_lag_1mo", "cost_lag_2mo",
        "truck_age_years", "month_of_year",
        "fuel_cost_per_mile", "trip_count_monthly", "avg_trip_distance",
        "safety_incident_flag",
        "total_revenue_monthly",
        "maintenance_cost_next_month",
        "high_cost_next_month",
    ]

    feature_matrix = util[feature_cols].copy()

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    output_path = config.OUTPUT_DIR / "feature_matrix.csv"
    feature_matrix.to_csv(output_path, index=False)
    print(f"\nFeature matrix saved to: {output_path}")
    print(f"  Shape: {feature_matrix.shape}")
    print(f"\nTarget distribution (high_cost_next_month):")
    dist = feature_matrix["high_cost_next_month"].value_counts()
    total = len(feature_matrix)
    for val in sorted(dist.index):
        count = dist[val]
        pct = count / total * 100
        print(f"  {val}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("Feature engineering complete.")
    print("=" * 60)

    return feature_matrix


if __name__ == "__main__":
    build_features()
