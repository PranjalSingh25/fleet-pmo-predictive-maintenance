-- ============================================================
-- FleetOps Anomaly Detection -- Dimension Tables
-- ============================================================
-- Star schema design for fleet maintenance analytics.
-- 5 dimension tables supporting 4 fact tables.
--
-- NOTE: dim_truck implements SCD Type 2 to track status changes
-- over time (Active -> Maintenance -> Retired).
-- ============================================================


-- ────────────────────────────────────────────────────────────
-- dim_date: Standard calendar dimension
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_date (
    date_key        INTEGER PRIMARY KEY,    -- YYYYMMDD format
    full_date       DATE NOT NULL UNIQUE,
    year            INTEGER NOT NULL,
    quarter         INTEGER NOT NULL,
    month           INTEGER NOT NULL,
    month_name      VARCHAR(20) NOT NULL,
    day_of_month    INTEGER NOT NULL,
    day_of_week     INTEGER NOT NULL,       -- 0=Monday, 6=Sunday
    day_name        VARCHAR(20) NOT NULL,
    is_weekend      BOOLEAN NOT NULL,
    week_of_year    INTEGER NOT NULL,
    year_month      VARCHAR(7) NOT NULL     -- '2022-01' format
);


-- ────────────────────────────────────────────────────────────
-- dim_truck: Fleet equipment dimension (SCD Type 2)
-- ────────────────────────────────────────────────────────────
-- Tracks status changes over time.
-- Each row represents a version of the truck's attributes.
-- is_current = TRUE for the latest version.
--
-- Example: TRK00003 changes from Active -> Maintenance
--   Row 1: status='Active',  effective_date='2020-09-17', expiry_date='2024-03-15', is_current=FALSE
--   Row 2: status='Maintenance', effective_date='2024-03-15', expiry_date='9999-12-31', is_current=TRUE

CREATE TABLE IF NOT EXISTS dim_truck (
    truck_key           SERIAL PRIMARY KEY,         -- Surrogate key (SCD2)
    truck_id            VARCHAR(10) NOT NULL,        -- Natural/business key
    unit_number         VARCHAR(10),
    make                VARCHAR(50),
    model_year          INTEGER,
    vin                 VARCHAR(25),
    acquisition_date    DATE,
    acquisition_mileage INTEGER,
    fuel_type           VARCHAR(20),
    tank_capacity_gal   INTEGER,
    status              VARCHAR(20),                 -- Tracked attribute (SCD2)
    home_terminal       VARCHAR(50),
    -- SCD Type 2 columns
    effective_date      DATE NOT NULL,
    expiry_date         DATE NOT NULL DEFAULT '9999-12-31',
    is_current          BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_dim_truck_id ON dim_truck(truck_id);
CREATE INDEX IF NOT EXISTS idx_dim_truck_current ON dim_truck(truck_id, is_current);


-- ────────────────────────────────────────────────────────────
-- dim_driver: Driver dimension
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_driver (
    driver_id           VARCHAR(10) PRIMARY KEY,
    first_name          VARCHAR(50),
    last_name           VARCHAR(50),
    full_name           VARCHAR(100),
    hire_date           DATE,
    termination_date    DATE,
    license_number      VARCHAR(20),
    license_state       VARCHAR(5),
    date_of_birth       DATE,
    home_terminal       VARCHAR(50),
    employment_status   VARCHAR(20),
    cdl_class           VARCHAR(5),
    years_experience    INTEGER
);


-- ────────────────────────────────────────────────────────────
-- dim_facility: Terminal and warehouse locations
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_facility (
    facility_id         VARCHAR(10) PRIMARY KEY,
    facility_name       VARCHAR(100),
    facility_type       VARCHAR(30),
    city                VARCHAR(50),
    state               VARCHAR(5),
    latitude            DECIMAL(10, 6),
    longitude           DECIMAL(10, 6),
    dock_doors          INTEGER,
    operating_hours     VARCHAR(20)
);


-- ────────────────────────────────────────────────────────────
-- dim_route: Origin-destination pairs
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_route (
    route_id                VARCHAR(10) PRIMARY KEY,
    origin_city             VARCHAR(50),
    origin_state            VARCHAR(5),
    destination_city        VARCHAR(50),
    destination_state       VARCHAR(5),
    typical_distance_miles  INTEGER,
    base_rate_per_mile      DECIMAL(6, 2),
    fuel_surcharge_rate     DECIMAL(6, 2),
    typical_transit_days    INTEGER
);
