# Power Automate Flow: Weekly Escalation Email

## Trigger
- **Schedule**: Every Monday at 8:00 AM

## Actions

### Step 1: Run SQL Query
Connect to PostgreSQL and run:

```sql
SELECT
    t.truck_id,
    t.make || ' ' || t.model_year AS truck_name,
    t.home_terminal,
    f.maintenance_cost AS ytd_actual,
    (b.budget / 12.0 * d.month) AS ytd_expected,
    ((f.maintenance_cost - (b.budget / 12.0 * d.month)) / NULLIF(b.budget / 12.0 * d.month, 0)) * 100 AS overrun_pct,
    a.root_cause,
    fc.p_over_budget
FROM fact_truck_monthly f
JOIN dim_truck t ON f.truck_key = t.truck_key AND t.is_current = 1
JOIN dim_date d ON f.date_key = d.date_key
LEFT JOIN (
    SELECT truck_id, SUM(maintenance_cost) AS budget
    FROM fact_truck_monthly
    JOIN dim_date ON fact_truck_monthly.date_key = dim_date.date_key
    WHERE dim_date.year = 2022
    GROUP BY truck_id
) b ON t.truck_id = b.truck_id
LEFT JOIN attribution_report a ON t.truck_id = a.truck_id AND a.month = d.month AND a.year = d.year
LEFT JOIN forecast_report fc ON t.truck_id = fc.truck_id AND fc.forecast_month = d.month
WHERE d.year = YEAR(CURRENT_DATE)
    AND d.month = MONTH(CURRENT_DATE)
    AND ((f.maintenance_cost - (b.budget / 12.0 * d.month)) / NULLIF(b.budget / 12.0 * d.month, 0)) * 100 > 10
ORDER BY overrun_pct DESC
```

### Step 2: Format HTML Table
Convert query results to a styled HTML table.

### Step 3: Send Email
- **To**: Fleet Manager + Maintenance Team
- **Subject**: `[FleetOps Alert] Weekly Overrun Report — {run_date}`
- **Body**: HTML table with flagged trucks

### Step 4: Log to pipeline_runs
INSERT a record to the pipeline_runs audit table with stage = "power_automate_flow", rows_flagged = count.
