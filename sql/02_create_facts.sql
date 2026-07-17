-- ============================================================
-- FleetOps Anomaly Detection -- Fact Tables
-- ============================================================
-- 4 fact tables at different grains supporting the fleet
-- maintenance analytics star schema.
-- ============================================================


-- ────────────────────────────────────────────────────────────
-- fact_truck_monthly: Monthly aggregated truck performance
-- Grain: one row per truck per month
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_truck_monthly (
    truck_id            VARCHAR(10) NOT NULL,
    date_key            INTEGER NOT NULL,               -- FK to dim_date (first of month)
    trips_completed     INTEGER,
    total_miles         DECIMAL(12, 2),
    total_revenue       DECIMAL(12, 2),
    average_mpg         DECIMAL(6, 2),
    maintenance_events  INTEGER,
    maintenance_cost    DECIMAL(12, 2),
    downtime_hours      DECIMAL(8, 2),
    utilization_rate    DECIMAL(5, 3),

    PRIMARY KEY (truck_id, date_key),
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);

CREATE INDEX IF NOT EXISTS idx_fact_truck_monthly_truck
    ON fact_truck_monthly(truck_id);
CREATE INDEX IF NOT EXISTS idx_fact_truck_monthly_date
    ON fact_truck_monthly(date_key);


-- ────────────────────────────────────────────────────────────
-- fact_maintenance: Individual maintenance events
-- Grain: one row per maintenance event
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_maintenance (
    maintenance_id      VARCHAR(20) PRIMARY KEY,
    truck_id            VARCHAR(10) NOT NULL,
    date_key            INTEGER NOT NULL,               -- FK to dim_date
    maintenance_type    VARCHAR(30),
    odometer_reading    INTEGER,
    labor_hours         DECIMAL(6, 2),
    labor_cost          DECIMAL(10, 2),
    parts_cost          DECIMAL(10, 2),
    total_cost          DECIMAL(10, 2),
    facility_location   VARCHAR(50),
    downtime_hours      DECIMAL(8, 2),
    service_description VARCHAR(200),

    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);

CREATE INDEX IF NOT EXISTS idx_fact_maintenance_truck
    ON fact_maintenance(truck_id);
CREATE INDEX IF NOT EXISTS idx_fact_maintenance_date
    ON fact_maintenance(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_maintenance_type
    ON fact_maintenance(maintenance_type);


-- ────────────────────────────────────────────────────────────
-- fact_trip: Individual trip performance
-- Grain: one row per trip
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_trip (
    trip_id             VARCHAR(20) PRIMARY KEY,
    load_id             VARCHAR(20),
    driver_id           VARCHAR(10),
    truck_id            VARCHAR(10) NOT NULL,
    trailer_id          VARCHAR(10),
    date_key            INTEGER NOT NULL,               -- FK to dim_date (dispatch date)
    actual_distance_miles   DECIMAL(10, 2),
    actual_duration_hours   DECIMAL(8, 2),
    fuel_gallons_used       DECIMAL(10, 2),
    average_mpg             DECIMAL(6, 2),
    idle_time_hours         DECIMAL(8, 2),
    trip_status             VARCHAR(20),

    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);

CREATE INDEX IF NOT EXISTS idx_fact_trip_truck ON fact_trip(truck_id);
CREATE INDEX IF NOT EXISTS idx_fact_trip_driver ON fact_trip(driver_id);
CREATE INDEX IF NOT EXISTS idx_fact_trip_date ON fact_trip(date_key);


-- ────────────────────────────────────────────────────────────
-- fact_fuel: Individual fuel purchase transactions
-- Grain: one row per fuel purchase
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_fuel (
    fuel_purchase_id    VARCHAR(20) PRIMARY KEY,
    trip_id             VARCHAR(20),
    truck_id            VARCHAR(10) NOT NULL,
    driver_id           VARCHAR(10),
    date_key            INTEGER NOT NULL,               -- FK to dim_date
    location_city       VARCHAR(50),
    location_state      VARCHAR(5),
    gallons             DECIMAL(8, 2),
    price_per_gallon    DECIMAL(6, 3),
    total_cost          DECIMAL(10, 2),
    fuel_card_number    VARCHAR(20),

    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);

CREATE INDEX IF NOT EXISTS idx_fact_fuel_truck ON fact_fuel(truck_id);
CREATE INDEX IF NOT EXISTS idx_fact_fuel_date ON fact_fuel(date_key);
