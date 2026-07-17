"""
FleetOps Anomaly Detection — Central Configuration
===================================================
All paths, thresholds, and settings for the pipeline.
"""
import os
from pathlib import Path


# ──────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT.parent / "Logistics Operations Database"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────
# DATA FILES
# ──────────────────────────────────────────────────────────
TRUCK_UTILIZATION_FILE = DATA_DIR / "truck_utilization_metrics.csv"
MAINTENANCE_RECORDS_FILE = DATA_DIR / "maintenance_records.csv"
TRUCKS_FILE = DATA_DIR / "trucks.csv"
TRIPS_FILE = DATA_DIR / "trips.csv"
FUEL_PURCHASES_FILE = DATA_DIR / "fuel_purchases.csv"
SAFETY_INCIDENTS_FILE = DATA_DIR / "safety_incidents.csv"
DRIVERS_FILE = DATA_DIR / "drivers.csv"
FACILITIES_FILE = DATA_DIR / "facilities.csv"
ROUTES_FILE = DATA_DIR / "routes.csv"
LOADS_FILE = DATA_DIR / "loads.csv"
CUSTOMERS_FILE = DATA_DIR / "customers.csv"
DELIVERY_EVENTS_FILE = DATA_DIR / "delivery_events.csv"
TRAILERS_FILE = DATA_DIR / "trailers.csv"
DRIVER_MONTHLY_METRICS_FILE = DATA_DIR / "driver_monthly_metrics.csv"


# ──────────────────────────────────────────────────────────
# DETECTION THRESHOLDS (Phase 1)
# ──────────────────────────────────────────────────────────
BASELINE_YEAR = 2022
ANALYSIS_YEARS = [2023, 2024]

WARNING_THRESHOLD_PCT = 5.0      # >5% overrun → WARNING
CRITICAL_THRESHOLD_PCT = 10.0    # >10% overrun → CRITICAL
FUEL_TREND_THRESHOLD_PCT = 10.0  # >10% fuel cost/mile increase → WATCH
FUEL_ROLLING_WINDOW = 3          # months for rolling average


# ──────────────────────────────────────────────────────────
# ATTRIBUTION THRESHOLDS (Phase 2)
# ──────────────────────────────────────────────────────────
EQUIPMENT_AGE_CUTOFF_YEAR = 2017     # model_year <= this → "old" truck
EXTENDED_DOWNTIME_HOURS = 30         # >30 hrs in a month → extended_downtime
CLUSTER_EVENT_COUNT = 4              # 4+ events in rolling window → cluster
CLUSTER_ROLLING_MONTHS = 3           # window for cluster detection
EXPENSIVE_REPAIR_TYPES = ["Engine", "Transmission"]


# ──────────────────────────────────────────────────────────
# MONTE CARLO (Phase 3)
# ──────────────────────────────────────────────────────────
NUM_SIMULATIONS = 10_000
RANDOM_SEED = 42


# ──────────────────────────────────────────────────────────
# ML (Phase 5)
# ──────────────────────────────────────────────────────────
ML_TRAIN_YEARS = [2022, 2023]
ML_TEST_YEAR = 2024
HIGH_COST_PERCENTILE = 75  # target: cost > 75th percentile


# ──────────────────────────────────────────────────────────
# DATABASE (Phase 6)
# ──────────────────────────────────────────────────────────
USE_POSTGRES = False  # Set True when PostgreSQL is available

POSTGRES_CONFIG = {
    "host": os.getenv("FLEETOPS_DB_HOST", "localhost"),
    "port": int(os.getenv("FLEETOPS_DB_PORT", "5432")),
    "database": os.getenv("FLEETOPS_DB_NAME", "fleetops"),
    "user": os.getenv("FLEETOPS_DB_USER", "fleetops_user"),
    "password": os.getenv("FLEETOPS_DB_PASSWORD", ""),
}

SQLITE_PATH = PROJECT_ROOT / "fleetops.db"


def get_db_url():
    """Return SQLAlchemy database URL based on configuration."""
    if USE_POSTGRES:
        cfg = POSTGRES_CONFIG
        return (
            f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
            f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
        )
    return f"sqlite:///{SQLITE_PATH}"


# ──────────────────────────────────────────────────────────
# LLM (Phase 4)
# ──────────────────────────────────────────────────────────
LLM_API_KEY = os.getenv("FLEETOPS_LLM_API_KEY", "")
LLM_MODEL = os.getenv("FLEETOPS_LLM_MODEL", "gpt-4o-mini")
