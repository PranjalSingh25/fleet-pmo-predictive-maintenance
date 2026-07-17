-- ============================================================
-- FleetOps Anomaly Detection -- Analytical Views
-- ============================================================
-- Pre-built analytical queries using window functions, CTEs,
-- and aggregations for fleet maintenance analytics.
-- These demonstrate SQL proficiency beyond basic SELECT.
-- ============================================================


-- ────────────────────────────────────────────────────────────
-- VIEW 1: Rolling Maintenance Cost Trends
-- Shows each truck's monthly cost alongside 3-month and 6-month
-- rolling averages, plus fleet-wide percentile rank.
-- ────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW vw_rolling_cost_trends AS
SELECT
    ftm.truck_id,
    dd.full_date                                        AS month_date,
    dd.year,
    dd.month,
    ftm.maintenance_cost,

    -- Rolling averages (window functions)
    AVG(ftm.maintenance_cost) OVER (
        PARTITION BY ftm.truck_id
        ORDER BY dd.full_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    )                                                   AS cost_rolling_3mo,

    AVG(ftm.maintenance_cost) OVER (
        PARTITION BY ftm.truck_id
        ORDER BY dd.full_date
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    )                                                   AS cost_rolling_6mo,

    -- Month-over-month change
    ftm.maintenance_cost - LAG(ftm.maintenance_cost) OVER (
        PARTITION BY ftm.truck_id
        ORDER BY dd.full_date
    )                                                   AS cost_mom_change,

    -- Fleet-wide percentile rank for this month
    PERCENT_RANK() OVER (
        PARTITION BY dd.full_date
        ORDER BY ftm.maintenance_cost
    )                                                   AS fleet_percentile_rank

FROM fact_truck_monthly ftm
JOIN dim_date dd ON ftm.date_key = dd.date_key
WHERE dd.day_of_month = 1;


-- ────────────────────────────────────────────────────────────
-- VIEW 2: Year-over-Year Maintenance Cost Comparison
-- Compares each truck's monthly cost to the same month last year.
-- ────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW vw_yoy_cost_comparison AS
WITH monthly_costs AS (
    SELECT
        ftm.truck_id,
        dd.year,
        dd.month,
        ftm.maintenance_cost,
        ftm.downtime_hours,
        ftm.maintenance_events
    FROM fact_truck_monthly ftm
    JOIN dim_date dd ON ftm.date_key = dd.date_key
    WHERE dd.day_of_month = 1
)
SELECT
    curr.truck_id,
    curr.year                                       AS current_year,
    curr.month,
    curr.maintenance_cost                           AS current_cost,
    prev.maintenance_cost                           AS prior_year_cost,
    curr.maintenance_cost - COALESCE(prev.maintenance_cost, 0)
                                                    AS yoy_change_dollars,
    CASE
        WHEN prev.maintenance_cost > 0
        THEN ((curr.maintenance_cost - prev.maintenance_cost)
              / prev.maintenance_cost) * 100
        ELSE NULL
    END                                             AS yoy_change_pct,
    curr.downtime_hours                             AS current_downtime,
    prev.downtime_hours                             AS prior_year_downtime
FROM monthly_costs curr
LEFT JOIN monthly_costs prev
    ON curr.truck_id = prev.truck_id
    AND curr.month = prev.month
    AND curr.year = prev.year + 1;


-- ────────────────────────────────────────────────────────────
-- VIEW 3: Truck Cost-to-Revenue Ratio
-- Per-truck-month ratio of maintenance cost to revenue generated.
-- Identifies trucks that are costing more to maintain relative
-- to the revenue they produce.
-- ────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW vw_cost_to_revenue AS
SELECT
    ftm.truck_id,
    dd.year,
    dd.month,
    dd.full_date                                    AS month_date,
    ftm.maintenance_cost,
    ftm.total_revenue,
    CASE
        WHEN ftm.total_revenue > 0
        THEN (ftm.maintenance_cost / ftm.total_revenue) * 100
        ELSE NULL
    END                                             AS cost_to_revenue_pct,
    dt.make,
    dt.model_year,
    dt.home_terminal
FROM fact_truck_monthly ftm
JOIN dim_date dd ON ftm.date_key = dd.date_key
JOIN dim_truck dt ON ftm.truck_id = dt.truck_id AND dt.is_current = TRUE
WHERE dd.day_of_month = 1;


-- ────────────────────────────────────────────────────────────
-- VIEW 4: Fleet Summary Dashboard
-- Monthly fleet-wide KPIs aggregated across all trucks.
-- ────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW vw_fleet_summary AS
SELECT
    dd.year,
    dd.month,
    dd.year_month,
    COUNT(DISTINCT ftm.truck_id)                    AS active_trucks,
    SUM(ftm.total_miles)                            AS fleet_total_miles,
    SUM(ftm.total_revenue)                          AS fleet_total_revenue,
    SUM(ftm.maintenance_cost)                       AS fleet_total_maint_cost,
    AVG(ftm.maintenance_cost)                       AS avg_maint_cost_per_truck,
    SUM(ftm.downtime_hours)                         AS fleet_total_downtime,
    AVG(ftm.utilization_rate)                       AS avg_utilization_rate,
    SUM(ftm.maintenance_events)                     AS fleet_total_maint_events,

    -- Cost per mile
    CASE
        WHEN SUM(ftm.total_miles) > 0
        THEN SUM(ftm.maintenance_cost) / SUM(ftm.total_miles)
        ELSE NULL
    END                                             AS maint_cost_per_mile

FROM fact_truck_monthly ftm
JOIN dim_date dd ON ftm.date_key = dd.date_key
WHERE dd.day_of_month = 1
GROUP BY dd.year, dd.month, dd.year_month
ORDER BY dd.year, dd.month;


-- ────────────────────────────────────────────────────────────
-- VIEW 5: Maintenance Type Analysis
-- Breakdown of maintenance costs by type, with running totals
-- and proportion of total spend.
-- ────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW vw_maintenance_type_analysis AS
WITH type_summary AS (
    SELECT
        fm.maintenance_type,
        dd.year,
        COUNT(*)                                    AS event_count,
        SUM(fm.total_cost)                          AS total_cost,
        AVG(fm.total_cost)                          AS avg_cost_per_event,
        AVG(fm.downtime_hours)                      AS avg_downtime_hours,
        MAX(fm.total_cost)                          AS max_single_event_cost
    FROM fact_maintenance fm
    JOIN dim_date dd ON fm.date_key = dd.date_key
    GROUP BY fm.maintenance_type, dd.year
),
yearly_totals AS (
    SELECT year, SUM(total_cost) AS year_total
    FROM type_summary
    GROUP BY year
)
SELECT
    ts.maintenance_type,
    ts.year,
    ts.event_count,
    ts.total_cost,
    ts.avg_cost_per_event,
    ts.avg_downtime_hours,
    ts.max_single_event_cost,
    (ts.total_cost / yt.year_total) * 100           AS pct_of_yearly_spend
FROM type_summary ts
JOIN yearly_totals yt ON ts.year = yt.year
ORDER BY ts.year, ts.total_cost DESC;


-- ────────────────────────────────────────────────────────────
-- VIEW 6: Fuel Efficiency Trends per Truck
-- Monthly fuel metrics with fleet comparison.
-- ────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW vw_fuel_efficiency AS
SELECT
    ff.truck_id,
    dd.year,
    dd.month,
    SUM(ff.gallons)                                 AS monthly_gallons,
    SUM(ff.total_cost)                              AS monthly_fuel_cost,
    AVG(ff.price_per_gallon)                        AS avg_price_per_gallon,

    -- Compare to fleet average for same month
    SUM(ff.total_cost) - AVG(SUM(ff.total_cost)) OVER (
        PARTITION BY dd.year, dd.month
    )                                               AS vs_fleet_avg_cost

FROM fact_fuel ff
JOIN dim_date dd ON ff.date_key = dd.date_key
GROUP BY ff.truck_id, dd.year, dd.month
ORDER BY ff.truck_id, dd.year, dd.month;
