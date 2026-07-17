"""
Phase 2 — Root-Cause Attribution
=================================
For each truck flagged in Phase 1 (severity is not null), pull maintenance
events from the overrun period and classify the dominant root cause.

Root Cause Labels (descending priority):
    equipment_age_failure  — Engine/Transmission repair on a truck with
                             model_year <= EQUIPMENT_AGE_CUTOFF_YEAR
    extended_downtime      — Any month with > EXTENDED_DOWNTIME_HOURS downtime
    cluster_overspend      — CLUSTER_EVENT_COUNT+ maintenance events in any
                             rolling CLUSTER_ROLLING_MONTHS-month window
    routine_overspend      — None of the above; cost is consistently above
                             baseline with no single root cause

Usage:
    python attribution.py
"""
import calendar

import pandas as pd

import config


# ─────────────────────────────────────────────────────────────
# PRIORITY MAP  (lower number = higher priority)
# ─────────────────────────────────────────────────────────────
ROOT_CAUSE_PRIORITY = {
    "equipment_age_failure": 1,
    "extended_downtime": 2,
    "cluster_overspend": 3,
    "routine_overspend": 4,
}


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_flagged_trucks() -> pd.DataFrame:
    """Load Phase 1 flagged trucks (only rows with a severity value)."""
    path = config.OUTPUT_DIR / "flagged_trucks_all.csv"
    df = pd.read_csv(path)
    # Keep only flagged rows (severity is not null / not empty)
    df = df.dropna(subset=["severity"])
    df = df[df["severity"].astype(str).str.strip() != ""]
    return df


def load_maintenance_records() -> pd.DataFrame:
    """Load maintenance records with parsed dates."""
    df = pd.read_csv(config.MAINTENANCE_RECORDS_FILE)
    df["maintenance_date"] = pd.to_datetime(df["maintenance_date"])
    df["maint_year"] = df["maintenance_date"].dt.year
    df["maint_month"] = df["maintenance_date"].dt.month
    return df


def load_trucks() -> pd.DataFrame:
    """Load truck master data (need model_year and make)."""
    df = pd.read_csv(config.TRUCKS_FILE)
    return df[["truck_id", "make", "model_year"]]


# ─────────────────────────────────────────────────────────────
# ROOT-CAUSE CLASSIFICATION HELPERS
# ─────────────────────────────────────────────────────────────

def _check_equipment_age_failure(
    events: pd.DataFrame, model_year: int
) -> tuple[bool, str | None]:
    """
    True when maintenance_type ∈ EXPENSIVE_REPAIR_TYPES
    AND model_year <= EQUIPMENT_AGE_CUTOFF_YEAR.
    Returns (matched, evidence_summary).
    """
    if model_year > config.EQUIPMENT_AGE_CUTOFF_YEAR:
        return False, None

    expensive = events[
        events["maintenance_type"].isin(config.EXPENSIVE_REPAIR_TYPES)
    ]
    if expensive.empty:
        return False, None

    # Pick the single costliest event for the evidence line
    top = expensive.loc[expensive["total_cost"].idxmax()]
    summary = (
        f"{top['maintenance_type']} repair (${top['total_cost']:,.2f}) "
        f"on {model_year} {top.get('make', 'truck')} "
        f"-- equipment age failure"
    )
    return True, summary


def _check_extended_downtime(
    events: pd.DataFrame, year: int, month: int
) -> tuple[bool, str | None]:
    """
    True when total downtime_hours in the given (year, month) exceeds the
    threshold.
    """
    month_events = events[
        (events["maint_year"] == year) & (events["maint_month"] == month)
    ]
    total_dt = month_events["downtime_hours"].sum()
    if total_dt > config.EXTENDED_DOWNTIME_HOURS:
        month_name = calendar.month_name[month]
        summary = (
            f"{total_dt:.1f} hrs downtime in {month_name} "
            f"-- extended downtime"
        )
        return True, summary
    return False, None


def _check_cluster_overspend(
    events: pd.DataFrame, year: int, month: int
) -> tuple[bool, str | None]:
    """
    True when CLUSTER_EVENT_COUNT or more maintenance events fall within
    any rolling CLUSTER_ROLLING_MONTHS-month window ending at or before
    the current (year, month).
    """
    window = config.CLUSTER_ROLLING_MONTHS
    threshold = config.CLUSTER_EVENT_COUNT

    # Build the end of the target month
    end_date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    # Start is (window - 1) months before the first day of the target month
    start_date = pd.Timestamp(year=year, month=month, day=1) - pd.DateOffset(months=window - 1)

    windowed = events[
        (events["maintenance_date"] >= start_date)
        & (events["maintenance_date"] <= end_date)
    ]
    count = len(windowed)
    if count >= threshold:
        start_label = start_date.strftime("%b")
        end_label = end_date.strftime("%b")
        summary = (
            f"{count} maintenance events in {start_label}-{end_label} window "
            f"-- cluster overspend"
        )
        return True, summary
    return False, None


def _routine_evidence() -> str:
    """Fallback evidence string."""
    return "Steady above-baseline spend, no single root cause"


# ─────────────────────────────────────────────────────────────
# MAIN ATTRIBUTION LOGIC
# ─────────────────────────────────────────────────────────────

def classify_root_cause(
    events: pd.DataFrame,
    model_year: int,
    make: str,
    year: int,
    month: int,
) -> tuple[str, str]:
    """
    Classify the root cause for one (truck, year, month) combination.

    Returns (root_cause_label, evidence_summary).
    """
    # Inject make into events for the evidence string builder
    events = events.copy()
    events["make"] = make

    labels: list[tuple[int, str, str]] = []  # (priority, label, evidence)

    # 1. equipment_age_failure
    matched, evidence = _check_equipment_age_failure(events, model_year)
    if matched:
        labels.append((ROOT_CAUSE_PRIORITY["equipment_age_failure"],
                        "equipment_age_failure", evidence))

    # 2. extended_downtime
    matched, evidence = _check_extended_downtime(events, year, month)
    if matched:
        labels.append((ROOT_CAUSE_PRIORITY["extended_downtime"],
                        "extended_downtime", evidence))

    # 3. cluster_overspend
    matched, evidence = _check_cluster_overspend(events, year, month)
    if matched:
        labels.append((ROOT_CAUSE_PRIORITY["cluster_overspend"],
                        "cluster_overspend", evidence))

    if not labels:
        return "routine_overspend", _routine_evidence()

    # Pick highest priority (lowest number)
    labels.sort(key=lambda x: x[0])
    return labels[0][1], labels[0][2]


def build_attribution_report(
    flagged: pd.DataFrame,
    maintenance: pd.DataFrame,
    trucks: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full attribution report for every flagged (truck, year, month).
    """
    # Merge truck metadata into flagged
    flagged = flagged.merge(trucks, on="truck_id", how="left")

    rows: list[dict] = []

    # Group flagged trucks by (truck_id, year, current_month)
    for _, row in flagged.iterrows():
        truck_id = row["truck_id"]
        year = int(row["year"])
        month = int(row["current_month"])
        model_year = int(row["model_year"]) if pd.notna(row.get("model_year")) else 0
        make = row.get("make", "Unknown")
        if pd.isna(make):
            make = "Unknown"

        # Pull ALL maintenance events for this truck in the analysis year
        # (so rolling windows can look back before the flagged month)
        truck_maint = maintenance[
            (maintenance["truck_id"] == truck_id)
            & (maintenance["maint_year"] == year)
        ].copy()

        # Also pull events from the prior year's last months for cluster
        # windows that span year boundaries
        prior_start = pd.Timestamp(year=year, month=1, day=1) - pd.DateOffset(
            months=config.CLUSTER_ROLLING_MONTHS - 1
        )
        truck_maint_extended = maintenance[
            (maintenance["truck_id"] == truck_id)
            & (maintenance["maintenance_date"] >= prior_start)
            & (maintenance["maint_year"] <= year)
        ].copy()

        # Events in the specific month (for stats)
        month_events = truck_maint[truck_maint["maint_month"] == month]

        # Classify root cause (pass extended window for cluster detection)
        root_cause, evidence = classify_root_cause(
            events=truck_maint_extended,
            model_year=model_year,
            make=make,
            year=year,
            month=month,
        )

        # Compute summary stats
        maint_count = len(month_events)
        total_downtime = month_events["downtime_hours"].sum() if maint_count > 0 else 0.0

        if maint_count > 0:
            top_idx = month_events["total_cost"].idxmax()
            top_type = month_events.loc[top_idx, "maintenance_type"]
            top_amount = month_events.loc[top_idx, "total_cost"]
        else:
            top_type = None
            top_amount = 0.0

        rows.append({
            "truck_id": truck_id,
            "year": year,
            "month": month,
            "severity": row["severity"],
            "overrun_pct": row["overrun_pct"],
            "overrun_dollars": row["overrun_dollars"],
            "root_cause": root_cause,
            "evidence_summary": evidence,
            "maintenance_events_count": maint_count,
            "total_downtime_hrs": round(total_downtime, 1),
            "top_cost_event_type": top_type,
            "top_cost_event_amount": round(top_amount, 2),
            "truck_model_year": model_year,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────

def save_reports(report: pd.DataFrame) -> None:
    """
    Save the combined attribution report and per-month CSVs.
    """
    combined_path = config.OUTPUT_DIR / "attribution_report_all.csv"
    report.to_csv(combined_path, index=False)
    print(f"  [OK] Combined report: {combined_path}  ({len(report)} rows)")

    # Per-month files
    for (year, month), group in report.groupby(["year", "month"]):
        label = f"{int(year)}-{int(month):02d}"
        month_path = config.OUTPUT_DIR / f"attribution_report_{label}.csv"
        group.to_csv(month_path, index=False)
        print(f"  [OK] {month_path}  ({len(group)} rows)")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run_attribution() -> pd.DataFrame:
    """
    Execute Phase 2: Root-Cause Attribution.

    Returns the full attribution DataFrame (also saved to disk).
    """
    print("=" * 60)
    print("Phase 2 — Root-Cause Attribution")
    print("=" * 60)

    # 1. Load data
    print("\n[1/3] Loading data ...")
    flagged = load_flagged_trucks()
    print(f"      Flagged trucks rows: {len(flagged)}")

    maintenance = load_maintenance_records()
    print(f"      Maintenance records: {len(maintenance)}")

    trucks = load_trucks()
    print(f"      Truck roster:        {len(trucks)}")

    # 2. Build report
    print("\n[2/3] Classifying root causes ...")
    report = build_attribution_report(flagged, maintenance, trucks)

    # Summary stats
    cause_counts = report["root_cause"].value_counts()
    print("\n      Root-cause distribution:")
    for cause, count in cause_counts.items():
        print(f"        {cause:30s} {count:>5d}")

    # 3. Save
    print("\n[3/3] Saving reports ...")
    save_reports(report)

    print("\n" + "=" * 60)
    print("Phase 2 complete.")
    print("=" * 60)

    return report


if __name__ == "__main__":
    run_attribution()
