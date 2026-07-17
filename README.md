# FleetOps Anomaly Detection

A production-grade system that catches truck maintenance cost overruns **mid-year** (not at year-end), explains why they're happening, forecasts year-end outcomes, and triggers actionable escalations — all on real fleet data.

## The Problem

Fleet maintenance budgets are typically reviewed at year-end. By then, a truck that's been bleeding money since March has already cost the company thousands in avoidable overruns. This system catches those overruns as they develop and attributes root causes before the repair bill compounds.

## Architecture

```
Source Data (15 CSVs, 300K+ records)
        |
        v
  [Phase 1] Detection -----> Flags trucks: CRITICAL / WARNING / WATCH
        |
        v
  [Phase 2] Attribution ----> Classifies root cause per flagged truck
        |
        v
  [Phase 3] Monte Carlo ----> Forecasts P(over_budget_at_year_end)
        |
        v
  [Phase 4] Escalation -----> Generates structured escalation emails
        |
  [Phase 5] ML Extension ---> LR vs XGBoost for cost spike prediction
        |
        v
  [Phase 6] Productionization
        |- PostgreSQL/SQLite star schema (5 dims + 4 facts)
        |- SCD Type 2 on dim_truck
        |- Data quality suite
        |- Pipeline orchestrator with audit logging
        |
  [Phase 7] Power Platform
        |- DAX measures (fleet KPI dashboard)
        |- Power Automate (weekly escalation email flow)
```

## How It Works

### Phase 1 — Detection (`detection.py`)
Uses 2022 as the baseline budget. For each truck in 2023–2024, computes YTD actual vs. pro-rata expected at each month-end. Flags **WARNING** at >5% overrun, **CRITICAL** at >10%.

**Leading indicator**: Also computes rolling 3-month fuel cost per mile from `fuel_purchases`. Flags **WATCH** when fuel cost/mile trends >10% above baseline — this signals engine degradation before the maintenance bill arrives.

### Phase 2 — Attribution (`attribution.py`)
Joins flagged trucks to `maintenance_records` and `trucks` to classify root cause:
- **equipment_age_failure** — Engine/Transmission repair on a pre-2017 truck
- **extended_downtime** — >30 hours downtime in a month
- **cluster_overspend** — 4+ maintenance events in a 3-month window
- **routine_overspend** — Above-baseline spend, no single root cause

### Phase 3 — Forecasting (`monte_carlo.py`)
Runs 10,000 Monte Carlo simulations per flagged truck to project year-end total cost. Reports P(over_budget), 10th/50th/90th percentile outcomes, and projected overrun at median.

### Phase 4 — Action (`escalation_engine.py`)
Generates structured escalation emails for every CRITICAL truck combining:
- **What** — overrun % and dollar amount
- **Why** — root cause + evidence
- **Where it's headed** — P(over_budget) + projected overrun
- **Who should act** — routing based on root cause
- **Suggested next step** — actionable recommendation

### Phase 5 — ML Extension
- **Feature engineering** across 5 source tables: rolling windows, cross-source fuel cost per mile, safety incident flags, truck age
- **Logistic Regression** baseline vs. **XGBoost** gradient boosting
- Temporal train/test split (2022–2023 train, 2024 test)
- Comparison: precision, recall, F1, ROC-AUC, feature importance

### Phase 6 — Productionization
- **Star schema**: 5 dimensions + 4 fact tables using 9 of 15 source tables
- **SCD Type 2** on `dim_truck` for status change tracking
- **Pipeline orchestrator** (`run_pipeline.py`) with audit logging
- **Data quality suite**: null checks, referential integrity, range validation, freshness
- **Incremental load** support with SCD2 merge logic
- **CI/CD**: GitHub Actions with ruff + pytest + quality gates

## Dataset & Setup

This project uses the real-world **[Logistics Operations Database](https://www.kaggle.com/datasets/yogape/logistics-operations-database)** from Kaggle (created by Yoga PE).

### Dataset Scale & Scope
- **15 Relational CSV Tables**: `trucks.csv`, `maintenance_records.csv`, `truck_utilization_metrics.csv`, `fuel_purchases.csv`, `safety_incidents.csv`, `trips.csv`, `drivers.csv`, `loads.csv`, and more.
- **Volume**: 300,000+ total records covering **36 continuous months (2022–2024)** across a fleet of **92 heavy trucks**.
- **Key Metrics**: 196,000+ fuel transactions, 2,920 detailed maintenance events across 7 repair categories, and full trip/utilization logs.

### Getting the Data
To keep the Git repository lightweight (`<100 MB`), raw data files are not tracked directly in version control. To run the pipeline locally:

1. **Download via Kaggle CLI** (recommended):
   ```bash
   pip install kaggle
   kaggle datasets download -d yogape/logistics-operations-database --unzip -p data/
   ```
2. **Or Manual Download**:
   - Download the archive directly from [Kaggle](https://www.kaggle.com/datasets/yogape/logistics-operations-database).
   - Extract the 15 CSV files into a folder named `data/` inside the project root (or set the `FLEETOPS_DATA_DIR` environment variable pointing to your custom data folder).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full analysis pipeline (CSV-based, no database required)
python detection.py
python attribution.py
python monte_carlo.py
python escalation_engine.py

# Run ML pipeline
python feature_engineering.py
python logistic_regression.py
python xgboost_model.py

# Run the orchestrated pipeline
python production/run_pipeline.py --skip-db

# With database (PostgreSQL or SQLite)
python production/ingestion.py
python production/transform.py
python production/data_quality.py
python production/run_pipeline.py

# Run tests
pytest tests/ -v
```

## Project Structure

```
antigravity-FleetOps-Anomaly-Detection/
|-- config.py                    # Central configuration
|-- requirements.txt             # Dependencies
|
|-- detection.py                 # Phase 1: Overrun detection + fuel trends
|-- attribution.py               # Phase 2: Root cause classification
|-- monte_carlo.py               # Phase 3: Year-end forecasting
|-- escalation_engine.py         # Phase 4: Escalation email generation
|-- feature_engineering.py       # Phase 5: Cross-source feature matrix
|-- logistic_regression.py       # Phase 5: Baseline ML model
|-- xgboost_model.py             # Phase 5: Gradient boosting + comparison
|
|-- sql/
|   |-- 01_create_dimensions.sql # 5 dimension tables (SCD2 on dim_truck)
|   |-- 02_create_facts.sql      # 4 fact tables
|   |-- 03_create_audit.sql      # Pipeline audit table
|   |-- 04_analytical_views.sql  # Window functions, CTEs, rankings
|
|-- production/
|   |-- ingestion.py             # CSV -> database
|   |-- transform.py             # Raw -> star schema
|   |-- incremental_load.py      # New month ingestion + SCD2
|   |-- data_quality.py          # Validation suite
|   |-- run_pipeline.py          # Full pipeline orchestrator
|
|-- power_platform/
|   |-- dax_measures.txt          # Power BI DAX expressions
|   |-- power_automate_flow.md    # Weekly escalation automation
|
|-- tests/                       # pytest suite (25 tests)
|-- outputs/                     # Generated CSVs, reports, emails
|-- .github/workflows/ci.yml    # CI/CD pipeline
```

## Key Results

| Metric | Value |
|--------|-------|
| Trucks flagged | 87 / 92 (across 2023-2024) |
| CRITICAL snapshots | 911 |
| WARNING snapshots | 46 |
| WATCH (fuel trend) snapshots | 66 |
| Fuel trend alerts | 43 unique trucks |
| High-risk forecasts (>=75% P(over)) | 798 |
| Mean projected overrun (P50) | $7,159 |
| Root cause: equipment age failure | 84.5% |
| Tests passing | 25/25 |

## Skills Demonstrated

| Area | Implementation |
|------|---------------|
| **Python / pandas** | 10 production scripts, cross-source joins, rolling windows |
| **SQL** | Star schema DDL, SCD Type 2, window functions, CTEs, analytical views |
| **Data Modeling** | 5 dimensions + 4 facts, multi-grain star schema |
| **ETL/ELT** | CSV ingestion, raw-to-star transformation, incremental loads |
| **Pipeline Engineering** | Orchestration, audit logging, error handling, idempotency |
| **Power BI (DAX)** | 10 DAX measures: YTD maintenance cost, overrun %, fleet budget, route profitability, driver scorecard, safety dashboard |
| **Power Automate** | Scheduled weekly flow: SQL query → HTML email → audit logging |
| **Applied Statistics** | Monte Carlo simulation, confidence intervals, P(over_budget) |
| **ML** | Feature engineering, LR vs XGBoost, temporal train/test split |
| **Data Quality** | Null checks, referential integrity, range validation, freshness |
| **CI/CD** | GitHub Actions: ruff + pytest + quality gates |
| **Production Thinking** | SCD2, audit tables, incremental loads, SQLite fallback |

## Data Source

Fleet logistics database with 15 tables:
- 92 trucks, 120+ drivers, 85K+ trips, 196K+ fuel purchases
- 2,920 maintenance records across 7 categories
- 36 months of data (2022–2024)
