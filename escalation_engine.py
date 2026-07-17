"""
Phase 4 -- Escalation Engine
==============================
Generates structured escalation emails for CRITICAL-flagged trucks by
combining detection, attribution, and forecast data.

Each email contains:
    - What's happening (overrun % + dollar amount)
    - Why (attribution label + evidence)
    - Where it's headed (P(over_budget) + projected overrun)
    - Who should act (routing based on root cause + severity)
    - Suggested next step

Usage:
    python escalation_engine.py
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

import config


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_phase_outputs():
    """
    Load outputs from Phases 1-3.
    Returns tuple of (flagged_df, attribution_df, forecast_df).
    """
    flagged_path = config.OUTPUT_DIR / "flagged_trucks_all.csv"
    attribution_path = config.OUTPUT_DIR / "attribution_report_all.csv"
    forecast_path = config.OUTPUT_DIR / "forecast_all.csv"

    flagged = pd.read_csv(flagged_path)
    attribution = pd.read_csv(attribution_path)
    forecast = pd.read_csv(forecast_path)

    return flagged, attribution, forecast


def build_escalation_dataset(flagged, attribution, forecast):
    """
    Merge detection + attribution + forecast into a single dataset
    for escalation email generation.

    Only includes CRITICAL-flagged truck-month snapshots.
    """
    # Filter to CRITICAL only for escalation
    critical = flagged[flagged["severity"] == "CRITICAL"].copy()

    # Merge attribution
    # Attribution uses 'month' column; flagged uses 'current_month'
    attr_cols = [
        "truck_id", "year", "month", "root_cause", "evidence_summary",
        "maintenance_events_count", "total_downtime_hrs",
        "top_cost_event_type", "top_cost_event_amount", "truck_model_year",
    ]
    available_attr_cols = [c for c in attr_cols if c in attribution.columns]

    merged = critical.merge(
        attribution[available_attr_cols],
        left_on=["truck_id", "year", "current_month"],
        right_on=["truck_id", "year", "month"],
        how="left",
    )

    # Merge forecast
    forecast_cols = [
        "truck_id", "year", "forecast_month", "p_over_budget",
        "projected_p10", "projected_p50", "projected_p90",
        "projected_overrun_p50", "remaining_months",
    ]
    available_fc_cols = [c for c in forecast_cols if c in forecast.columns]

    merged = merged.merge(
        forecast[available_fc_cols],
        left_on=["truck_id", "year", "current_month"],
        right_on=["truck_id", "year", "forecast_month"],
        how="left",
    )

    return merged


# ─────────────────────────────────────────────────────────────
# ROUTING LOGIC
# ─────────────────────────────────────────────────────────────

def determine_escalation_target(root_cause, overrun_pct, p_over_budget):
    """
    Route the escalation to the appropriate team based on root cause
    and severity of the projected outcome.

    Returns:
        tuple of (primary_owner, escalate_to)
    """
    # Financial escalation for extreme overruns
    if overrun_pct > 50 or (p_over_budget and p_over_budget > 0.9):
        escalate_to = "Finance Director + VP of Fleet Operations"
    else:
        escalate_to = None

    # Primary routing by root cause
    routing = {
        "equipment_age_failure": "Fleet Manager -- Capital Planning",
        "extended_downtime": "Maintenance Supervisor -- Scheduling",
        "cluster_overspend": "Maintenance Team Lead -- Vendor Review",
        "routine_overspend": "Fleet Analyst -- Budget Review",
    }

    primary_owner = routing.get(root_cause, "Fleet Manager")

    return primary_owner, escalate_to


def get_recommended_action(root_cause, truck_model_year=None, overrun_pct=None):
    """
    Generate a concrete, actionable recommendation based on root cause.
    """
    actions = {
        "equipment_age_failure": (
            f"1. Schedule comprehensive inspection of engine and drivetrain\n"
            f"   2. Run cost-of-ownership analysis: repair vs. replacement\n"
            f"   3. If model year is {truck_model_year or 'N/A'}, evaluate "
            f"early retirement and replacement cycle timing"
        ),
        "extended_downtime": (
            "1. Audit maintenance scheduling -- are repairs being batched efficiently?\n"
            "   2. Check if parts availability is causing extended wait times\n"
            "   3. Evaluate whether this truck should be temporarily pulled from rotation"
        ),
        "cluster_overspend": (
            "1. Review the last 3 months of maintenance orders for this unit\n"
            "   2. Check if multiple vendors are doing overlapping work\n"
            "   3. Consider assigning a single maintenance coordinator for this truck"
        ),
        "routine_overspend": (
            "1. Compare this truck's per-mile maintenance cost to fleet average\n"
            "   2. Review preventive maintenance schedule adherence\n"
            "   3. Flag for next monthly fleet review meeting"
        ),
    }

    return actions.get(root_cause, "Review maintenance history and schedule fleet review.")


# ─────────────────────────────────────────────────────────────
# EMAIL GENERATION
# ─────────────────────────────────────────────────────────────

def generate_email(row):
    """
    Generate a structured escalation email for a single truck-month record.
    Uses f-string templates (no LLM API required).
    """
    truck_id = row.get("truck_id", "UNKNOWN")
    year = row.get("year", "")
    month = row.get("current_month", "")
    overrun_pct = row.get("overrun_pct", 0)
    overrun_dollars = row.get("overrun_dollars", 0)
    ytd_actual = row.get("ytd_actual", 0)
    ytd_budget = row.get("ytd_budget", 0)
    severity = row.get("severity", "CRITICAL")
    root_cause = row.get("root_cause", "routine_overspend")
    evidence = row.get("evidence_summary", "No attribution data available")
    p_over_budget = row.get("p_over_budget", None)
    projected_p50 = row.get("projected_p50", None)
    projected_overrun_p50 = row.get("projected_overrun_p50", None)
    truck_model_year = row.get("truck_model_year", None)
    annual_budget = row.get("annual_budget", 0)

    # Routing
    primary_owner, escalate_to = determine_escalation_target(
        root_cause, overrun_pct, p_over_budget
    )

    # Recommended action
    action = get_recommended_action(root_cause, truck_model_year, overrun_pct)

    # Build forecast section
    if p_over_budget is not None and not pd.isna(p_over_budget):
        forecast_section = (
            f"  Probability of exceeding annual budget: {p_over_budget:.0%}\n"
            f"  Projected year-end cost (median):       ${projected_p50:,.0f}\n"
            f"  Projected overrun (median):             ${projected_overrun_p50:+,.0f}\n"
            f"  Annual budget:                          ${annual_budget:,.0f}"
        )
    else:
        forecast_section = (
            "  Forecast not available for this month (year-end or insufficient data)"
        )

    # Build email
    email = f"""
{'=' * 70}
FLEET MAINTENANCE COST ALERT -- {severity}
{'=' * 70}
Truck:    {truck_id}
Period:   {year}-{month:02d} (Month {month} of {year})
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

--- WHAT IS HAPPENING ---

  {truck_id} is {overrun_pct:+.1f}% over its YTD maintenance budget.
  YTD Actual:  ${ytd_actual:,.0f}
  YTD Budget:  ${ytd_budget:,.0f}
  Overrun:     ${overrun_dollars:+,.0f}

--- WHY ---

  Root Cause: {root_cause.replace('_', ' ').title()}
  Evidence:   {evidence}

--- WHERE IT IS HEADED ---

{forecast_section}

--- WHO SHOULD ACT ---

  Primary Owner:  {primary_owner}
{f'  Escalate To:    {escalate_to}' if escalate_to else '  (No additional escalation required at this time)'}

--- RECOMMENDED NEXT STEPS ---

   {action}

{'=' * 70}
"""
    return email.strip()


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_escalation():
    """
    Execute the full Phase 4 escalation pipeline.

    Returns:
        list of generated email strings
    """
    print("=" * 60)
    print("PHASE 4 -- ESCALATION ENGINE")
    print("=" * 60)

    # ── Step 1: Load phase outputs ──
    print("\n[1/4] Loading detection, attribution, and forecast data...")
    flagged, attribution, forecast = load_phase_outputs()
    critical_count = len(flagged[flagged["severity"] == "CRITICAL"])
    print(f"      Flagged trucks: {len(flagged):,}")
    print(f"      CRITICAL snapshots: {critical_count:,}")
    print(f"      Attribution records: {len(attribution):,}")
    print(f"      Forecast records: {len(forecast):,}")

    # ── Step 2: Build escalation dataset ──
    print("\n[2/4] Building escalation dataset (CRITICAL only)...")
    escalation_data = build_escalation_dataset(flagged, attribution, forecast)
    print(f"      Records for escalation: {len(escalation_data):,}")

    if len(escalation_data) == 0:
        print("\n  No CRITICAL trucks to escalate. Done.")
        return []

    # Root cause distribution
    if "root_cause" in escalation_data.columns:
        print("\n      Root cause distribution:")
        rc_counts = escalation_data["root_cause"].value_counts()
        for cause, count in rc_counts.items():
            print(f"        {cause}: {count}")

    # ── Step 3: Generate emails ──
    print("\n[3/4] Generating escalation emails...")
    emails = []
    for _, row in escalation_data.iterrows():
        email = generate_email(row)
        emails.append(email)

    print(f"      Generated {len(emails)} emails")

    # ── Step 4: Write outputs ──
    print("\n[4/4] Writing output files...")

    # Combined file
    combined_path = config.OUTPUT_DIR / "escalation_emails_all.txt"
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(f"FleetOps Maintenance Cost Escalation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Total Alerts: {len(emails)}\n")
        f.write(f"{'=' * 70}\n\n")
        for email in emails:
            f.write(email)
            f.write("\n\n")

    # Per-month files
    months_written = 0
    for (year, month), month_df in escalation_data.groupby(["year", "current_month"]):
        month_str = f"{year}-{month:02d}"
        month_path = config.OUTPUT_DIR / f"escalation_emails_{month_str}.txt"
        with open(month_path, "w", encoding="utf-8") as f:
            f.write(f"Escalation Report: {month_str}\n")
            f.write(f"{'=' * 70}\n\n")
            for _, row in month_df.iterrows():
                f.write(generate_email(row))
                f.write("\n\n")
        months_written += 1

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("ESCALATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total emails generated: {len(emails)}")
    print(f"  Monthly files written:  {months_written}")
    print(f"  Combined output: {combined_path}")

    # Show sample email
    if emails:
        print(f"\n  --- SAMPLE EMAIL (first) ---")
        # Print first 25 lines of first email
        sample_lines = emails[0].split("\n")[:25]
        for line in sample_lines:
            print(f"  {line}")
        if len(emails[0].split("\n")) > 25:
            print("  ... (truncated)")

    return emails


if __name__ == "__main__":
    run_escalation()
