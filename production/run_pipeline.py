"""
Production -- Pipeline Orchestrator
=====================================
Runs the full pipeline end-to-end with audit logging:
    1. Ingestion   (CSV -> raw tables)
    2. Transform   (raw -> star schema)
    3. Quality     (validation checks)
    4. Detection   (Phase 1)
    5. Attribution (Phase 2)
    6. Forecast    (Phase 3)
    7. Escalation  (Phase 4)

Each stage is logged to the pipeline_runs audit table.

Usage:
    python production/run_pipeline.py
    python production/run_pipeline.py --skip-db   # Skip DB stages, run CSV-based phases only
"""
import sys
import os
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def run_stage(stage_name, stage_fn, audit_log):
    """Run a pipeline stage with timing and error handling."""
    print(f"\n{'-' * 60}")
    print(f"  STAGE: {stage_name}")
    print(f"{'-' * 60}")

    start = time.time()
    try:
        result = stage_fn()
        duration = time.time() - start

        # Safely convert result to notes string
        if result is None:
            notes = ""
        elif isinstance(result, (list, dict)):
            notes = f"{type(result).__name__} with {len(result)} items"
        else:
            try:
                notes = f"{type(result).__name__} ({len(result)} rows)"
            except (TypeError, AttributeError):
                notes = str(result)[:200]

        audit_log.append({
            "stage_name": stage_name,
            "status": "SUCCESS",
            "duration_seconds": round(duration, 2),
            "notes": notes,
        })
        print(f"\n  >> {stage_name} completed in {duration:.1f}s")
        return True
    except Exception as e:
        duration = time.time() - start
        audit_log.append({
            "stage_name": stage_name,
            "status": "FAILED",
            "duration_seconds": round(duration, 2),
            "error_message": str(e)[:200],
        })
        print(f"\n  >> {stage_name} FAILED: {str(e)[:100]}")
        return False


def save_audit_log(audit_log):
    """Save audit log to CSV (always available, even without DB)."""
    import pandas as pd
    audit_df = pd.DataFrame(audit_log)
    audit_df["run_timestamp"] = datetime.now().isoformat()
    audit_path = config.OUTPUT_DIR / "pipeline_audit_log.csv"

    if audit_path.exists():
        existing = pd.read_csv(audit_path)
        audit_df = pd.concat([existing, audit_df], ignore_index=True)

    audit_df.to_csv(audit_path, index=False)
    return audit_path


def run_pipeline(skip_db=False):
    """Execute the full pipeline."""
    pipeline_start = time.time()
    audit_log = []

    print("=" * 60)
    print("FLEETOPS ANOMALY DETECTION -- FULL PIPELINE")
    print("=" * 60)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Skip DB: {skip_db}")

    # ── Database stages ──
    if not skip_db:
        from production.ingestion import run_ingestion
        from production.transform import run_transform
        from production.data_quality import run_data_quality

        run_stage("ingestion", run_ingestion, audit_log)
        run_stage("transform", run_transform, audit_log)
        run_stage("data_quality", run_data_quality, audit_log)

    # ── Analysis stages (always run, CSV-based) ──
    from detection import run_detection
    from attribution import run_attribution
    from monte_carlo import run_monte_carlo
    from escalation_engine import run_escalation

    run_stage("detection", run_detection, audit_log)
    run_stage("attribution", run_attribution, audit_log)
    run_stage("forecast", run_monte_carlo, audit_log)
    run_stage("escalation", run_escalation, audit_log)

    # ── Save audit log ──
    audit_path = save_audit_log(audit_log)

    # ── Summary ──
    total_duration = time.time() - pipeline_start
    successes = sum(1 for a in audit_log if a["status"] == "SUCCESS")
    failures = sum(1 for a in audit_log if a["status"] == "FAILED")

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total duration: {total_duration:.1f}s")
    print(f"  Stages passed:  {successes}/{len(audit_log)}")
    print(f"  Stages failed:  {failures}/{len(audit_log)}")
    print(f"  Audit log: {audit_path}")

    for entry in audit_log:
        marker = "OK  " if entry["status"] == "SUCCESS" else "FAIL"
        print(f"    [{marker}] {entry['stage_name']:.<30} {entry['duration_seconds']:.1f}s")

    return audit_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FleetOps Pipeline Orchestrator")
    parser.add_argument(
        "--skip-db", action="store_true",
        help="Skip database stages (ingestion, transform, quality). "
             "Run CSV-based analysis phases only."
    )
    args = parser.parse_args()
    run_pipeline(skip_db=args.skip_db)
