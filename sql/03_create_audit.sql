-- ============================================================
-- FleetOps Anomaly Detection -- Pipeline Audit Table
-- ============================================================
-- Tracks every pipeline run for operational monitoring.
-- Answers: "When was this data last refreshed?" and
-- "Which stage caused the last failure?"
-- ============================================================

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id              SERIAL PRIMARY KEY,
    run_timestamp       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    stage_name          VARCHAR(50) NOT NULL,       -- e.g. 'ingestion', 'transform', 'detection'
    status              VARCHAR(20) NOT NULL,       -- 'STARTED', 'SUCCESS', 'FAILED'
    rows_processed      INTEGER,
    rows_flagged        INTEGER,                    -- detection/attribution specific
    error_message       TEXT,
    duration_seconds    DECIMAL(10, 2),
    source_file         VARCHAR(200),               -- which CSV or table was processed
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_stage
    ON pipeline_runs(stage_name);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_timestamp
    ON pipeline_runs(run_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status
    ON pipeline_runs(status);


-- ────────────────────────────────────────────────────────────
-- Useful queries for pipeline monitoring
-- ────────────────────────────────────────────────────────────

-- Last successful run per stage
-- SELECT stage_name, MAX(run_timestamp) as last_success
-- FROM pipeline_runs
-- WHERE status = 'SUCCESS'
-- GROUP BY stage_name;

-- Recent failures
-- SELECT run_id, stage_name, run_timestamp, error_message
-- FROM pipeline_runs
-- WHERE status = 'FAILED'
-- ORDER BY run_timestamp DESC
-- LIMIT 10;

-- Average stage duration
-- SELECT stage_name, AVG(duration_seconds) as avg_seconds
-- FROM pipeline_runs
-- WHERE status = 'SUCCESS'
-- GROUP BY stage_name
-- ORDER BY avg_seconds DESC;
