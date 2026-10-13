"""
ml/audit_pmdata_timing.py

Research Data Quality & Timing Audit for PMData Wellness & sRPE Sessions.
Student IT23166660 - FullStrength.ai Research Project.

Tasks performed:
1. Inspects raw session end times (end_date_time) and durations (duration_min)
   to infer actual workout start times.
2. Compares wellness timestamps (effective_time_frame) against workout start times.
3. Classifies all 772 usable PMData sessions into:
   - matched_before_session: wellness strictly before workout start (prospective candidate)
   - during_session: wellness recorded while workout was ongoing (ambiguous/in-progress)
   - after_session: wellness recorded strictly after workout ended (retrospective leakage)
   - unmatched: no wellness record found on the same calendar date
4. Examines timezone representations (UTC vs Europe/Oslo) and effective_time_frame semantics.
5. Produces an audit-approved prospective subset (pmdata_audited_prospective_sessions.csv)
   strictly containing matched_before_session records.
6. Generates comprehensive audit reports and diagnostic figures under:
   C:\\Users\\rivinma\\Desktop\\ResearchData\\pmdata_experiments\\
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Force UTF-8 on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Configure Paths
INPUT_MERGED_CSV = Path(r"C:\Users\rivinma\Desktop\ResearchData\pmdata_merged_sessions.csv")
RAW_DATA_ROOT = Path(r"C:\Users\rivinma\Desktop\ResearchData\pmdata_selected")
EXP_DIR = Path(r"C:\Users\rivinma\Desktop\ResearchData\pmdata_experiments")
FIGURES_DIR = EXP_DIR / "figures"

EXP_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_and_audit_timing():
    print("=" * 80)
    print("  RESEARCH DATA QUALITY AUDIT : PMDATA TIMING & PROSPECTIVE ORDERING")
    print("=" * 80)

    if not INPUT_MERGED_CSV.is_file():
        raise FileNotFoundError(f"Merged CSV not found at: {INPUT_MERGED_CSV}")

    df = pd.read_csv(INPUT_MERGED_CSV)
    print(f"Loaded merged session dataset: {len(df)} rows, {df['participant_id'].nunique()} participants.")

    # 1. Parse Datetimes
    # Session end time is end_date_time from sRPE
    s_end = pd.to_datetime(df["session_datetime"], format="ISO8601", utc=True)
    # Session start time = session end time - duration_min
    s_dur = pd.to_numeric(df["session_duration_min"], errors="coerce")
    s_start = s_end - pd.to_timedelta(s_dur, unit="m")

    # Wellness timestamp from PMSys effective_time_frame
    w_time = pd.to_datetime(df["wellness_datetime"], format="ISO8601", utc=True)

    df["session_start_datetime"] = s_start
    df["session_end_datetime"] = s_end
    df["wellness_parsed_datetime"] = w_time

    # 2. Timing classification
    # Calculate time gap in hours: workout_start - wellness_time
    # Positive gap means wellness was answered BEFORE workout started.
    df["time_gap_hours"] = (s_start - w_time).dt.total_seconds() / 3600.0

    conditions = [
        df["wellness_parsed_datetime"].isna(),
        df["wellness_parsed_datetime"] < df["session_start_datetime"],
        (df["wellness_parsed_datetime"] >= df["session_start_datetime"]) & (df["wellness_parsed_datetime"] <= df["session_end_datetime"]),
        df["wellness_parsed_datetime"] > df["session_end_datetime"],
    ]
    choices = [
        "unmatched",
        "matched_before_session",
        "during_session",
        "after_session"
    ]
    df["timing_status"] = np.select(conditions, choices, default="ambiguous")

    # 3. Report Breakdown
    counts = df["timing_status"].value_counts()
    pcts = (df["timing_status"].value_counts(normalize=True) * 100).round(2)
    
    summary_table = pd.DataFrame({
        "Count": counts,
        "Percentage (%)": pcts
    })
    print("\n--- TIMING CLASSIFICATION BREAKDOWN ---")
    print(summary_table.to_string())

    # Participant-level breakdown
    p_breakdown = pd.crosstab(df["participant_id"], df["timing_status"], margins=True)
    print("\n--- PARTICIPANT-LEVEL BREAKDOWN ---")
    print(p_breakdown.to_string())

    # Time gap analysis for matched_before_session
    before_mask = df["timing_status"] == "matched_before_session"
    before_gaps = df.loc[before_mask, "time_gap_hours"]

    print("\n--- TIME GAP STATISTICS FOR MATCHED BEFORE SESSION (Hours) ---")
    print(f"Count : {before_gaps.count()}")
    print(f"Mean  : {before_gaps.mean():.2f} hours (wellness completed ~{before_gaps.mean():.1f}h before session)")
    print(f"Std   : {before_gaps.std():.2f} hours")
    print(f"Median: {before_gaps.median():.2f} hours")
    print(f"Min   : {before_gaps.min():.2f} hours ({before_gaps.min() * 60:.1f} mins)")
    print(f"Max   : {before_gaps.max():.2f} hours")
    print(f"IQR   : [{before_gaps.quantile(0.25):.2f}h - {before_gaps.quantile(0.75):.2f}h]")

    # Time gap analysis for after_session
    after_mask = df["timing_status"] == "after_session"
    after_gaps = df.loc[after_mask, "time_gap_hours"]
    print("\n--- TIME GAP STATISTICS FOR AFTER SESSION (Negative Hours = recorded post-workout) ---")
    print(f"Count : {after_gaps.count()}")
    print(f"Mean  : {after_gaps.mean():.2f} hours (wellness completed ~{abs(after_gaps.mean()):.1f}h AFTER session)")
    print(f"Std   : {after_gaps.std():.2f} hours")
    print(f"Median: {after_gaps.median():.2f} hours")
    print(f"Min   : {after_gaps.min():.2f} hours (most delayed: ~{abs(after_gaps.min()):.1f}h after)")
    print(f"Max   : {after_gaps.max():.2f} hours")

    # Timezone analysis: UTC vs Local Norway (Europe/Oslo)
    s_start_oslo = s_start.dt.tz_convert("Europe/Oslo")
    w_time_oslo = w_time.dt.tz_convert("Europe/Oslo")
    matched_nonnull = df["wellness_parsed_datetime"].notna()

    utc_same_day = (s_start.dt.date == w_time.dt.date)[matched_nonnull]
    oslo_same_day = (s_start_oslo.dt.date == w_time_oslo.dt.date)[matched_nonnull]
    tz_mismatch = (utc_same_day != oslo_same_day).sum()

    print("\n--- TIMEZONE & DATE BOUNDARY AUDIT ---")
    print(f"Matched non-null surveys: {matched_nonnull.sum()}")
    print(f"Same calendar date in UTC        : {utc_same_day.sum()} / {matched_nonnull.sum()}")
    print(f"Same calendar date in Europe/Oslo: {oslo_same_day.sum()} / {matched_nonnull.sum()}")
    print(f"Timezone date boundary mismatches : {tz_mismatch}")

    # 4. Save Audited Datasets
    # A. All sessions with audit flag
    all_audit_path = EXP_DIR / "pmdata_timing_audit_all_sessions.csv"
    df.to_csv(all_audit_path, index=False)
    print(f"\n[SAVED] All-sessions audit file -> {all_audit_path}")

    # B. Audit-approved prospective subset
    # Only records where wellness was recorded strictly before workout start
    audited_df = df[before_mask].copy().reset_index(drop=True)
    
    # Feature columns for modeling
    prospective_columns = [
        "participant_id",
        "session_datetime",
        "session_start_datetime",
        "wellness_datetime",
        "time_gap_hours",
        "primary_activity",
        "session_rpe",
        "fatigue",
        "mood",
        "readiness",
        "sleep_duration_h",
        "sleep_quality",
        "soreness",
        "stress"
    ]
    audited_df = audited_df[prospective_columns]
    
    # Verify no future records leaked
    assert (audited_df["time_gap_hours"] > 0).all(), "CRITICAL LEAKAGE: Found non-positive time gap in audited subset!"
    assert audited_df["participant_id"].nunique() == 15, "Expected all 15 active participants in audited subset."
    
    audited_csv_path = EXP_DIR / "pmdata_audited_prospective_sessions.csv"
    audited_df.to_csv(audited_csv_path, index=False)
    print(f"[SAVED] Audit-approved prospective subset -> {audited_csv_path}")
    print(f"        Rows: {len(audited_df)} from {audited_df['participant_id'].nunique()} participants.")

    # 5. Write Comprehensive Audit Report
    report_path = EXP_DIR / "pmdata_timing_audit_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PMDATA RESEARCH DATA QUALITY & TIMING AUDIT REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("Student ID: IT23166660 | Research: FullStrength.ai TEQS Component\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. RESEARCH CONTEXT & TIMING SEMANTICS:\n")
        f.write("In sports science and ML prediction pipelines, valid pre-session predictive modeling\n")
        f.write("requires STRICT temporal ordering: inputs must be collected before the athletic\n")
        f.write("session commences. Merging simply by calendar date (same-day matching) introduces\n")
        f.write("severe retrospective bias whenever an athlete fills their wellness survey after\n")
        f.write("having already exercised.\n\n")

        f.write("In PMData (PMSys app):\n")
        f.write("- effective_time_frame denotes the exact moment of survey submission in the mobile app.\n")
        f.write("- sRPE (perceived_exertion, duration_min, activity_names) is logged post-session.\n")
        f.write("- Inferred session start time = end_date_time - duration_min.\n")
        f.write("- Time gap = session_start_time - wellness_time.\n\n")

        f.write("2. AUDIT FINDINGS (N = 772 USABLE SESSIONS FROM 15 PARTICIPANTS):\n")
        f.write(summary_table.to_string() + "\n\n")

        f.write("DETAILED CATEGORY DEFINITIONS:\n")
        f.write(f"- matched_before_session: {counts.get('matched_before_session', 0)} sessions ({pcts.get('matched_before_session', 0):.1f}%).\n")
        f.write("  Wellness survey was completed strictly prior to workout start.\n")
        f.write(f"  Mean lead time: {before_gaps.mean():.2f} hours (median: {before_gaps.median():.2f}h, range: {before_gaps.min():.2f}h to {before_gaps.max():.2f}h).\n")
        f.write("  STATUS: Scientifically valid for pre-session modeling.\n\n")

        f.write(f"- during_session: {counts.get('during_session', 0)} sessions ({pcts.get('during_session', 0):.1f}%).\n")
        f.write("  Wellness survey was logged between session start and session end.\n")
        f.write("  STATUS: Ambiguous / in-progress; excluded from pre-session models.\n\n")

        f.write(f"- after_session: {counts.get('after_session', 0)} sessions ({pcts.get('after_session', 0):.1f}%).\n")
        f.write("  Wellness survey was logged after the session had completed.\n")
        f.write(f"  Mean delay: {abs(after_gaps.mean()):.2f} hours post-workout.\n")
        f.write("  STATUS: RETROSPECTIVE LEAKAGE. Post-exercise soreness and fatigue contaminate\n")
        f.write("  the survey. Excluded from prospective predictive validation.\n\n")

        f.write(f"- unmatched: {counts.get('unmatched', 0)} sessions ({pcts.get('unmatched', 0):.1f}%).\n")
        f.write("  No wellness survey was logged by the participant on that calendar date.\n")
        f.write("  STATUS: Completely missing wellness; excluded from modeling.\n\n")

        f.write("3. PARTICIPANT-LEVEL AUDIT MATRIX:\n")
        f.write(p_breakdown.to_string() + "\n\n")

        f.write("4. TIMEZONE & CALENDAR INTEGRITY:\n")
        f.write(f"- Raw timestamps are in UTC ('Z' suffix).\n")
        f.write(f"- Converting to local Europe/Oslo timezone reveals {tz_mismatch} calendar date shift(s)\n")
        f.write("  due to midnight-crossing activities, confirming that absolute UTC timestamps\n")
        f.write("  provide a safer ordering criteria than naive date strings.\n\n")

        f.write("5. CONCLUSION & METHODOLOGICAL DIRECTIVE:\n")
        f.write("Only the 539 'matched_before_session' records represent genuine prospective data.\n")
        f.write("Any analysis including the remaining 233 sessions must be explicitly labeled\n")
        f.write("'retrospective/exploratory' rather than 'pre-session predictive validation'.\n")

    print(f"[SAVED] Audit report -> {report_path}")

    # 6. Generate Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart of categories
        cat_order = ["matched_before_session", "after_session", "unmatched", "during_session"]
        colors = ["#2ecc71", "#e74c3c", "#95a5a6", "#f39c12"]
        cat_counts = [counts.get(c, 0) for c in cat_order]
        labels = ["Matched Before\n(Prospective)", "After Session\n(Leakage)", "Unmatched\n(No Survey)", "During Session\n(Ambiguous)"]
        
        bars = ax1.bar(labels, cat_counts, color=colors, edgecolor="black", linewidth=1.2)
        ax1.set_ylabel("Session Count", fontsize=11, fontweight="bold")
        ax1.set_title("PMData Sessions Timing Classification (N=772)", fontsize=12, fontweight="bold")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h + 8, f"{h}\n({h/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Histogram of time gap for matched_before_session
        ax2.hist(before_gaps, bins=25, color="#3498db", edgecolor="black", linewidth=1.0)
        ax2.axvline(before_gaps.median(), color="red", linestyle="--", linewidth=1.5, label=f"Median: {before_gaps.median():.1f}h")
        ax2.axvline(before_gaps.mean(), color="orange", linestyle=":", linewidth=1.5, label=f"Mean: {before_gaps.mean():.1f}h")
        ax2.set_xlabel("Time Gap: Workout Start - Wellness Survey (Hours)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        ax2.set_title("Distribution of Pre-Session Lead Time (N=539)", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_path = FIGURES_DIR / "timing_audit_distribution.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[SAVED] Timing audit figure -> {plot_path}")
    except Exception as e:
        print(f"[WARNING] Could not save plot: {e}")

    return audited_df, df


if __name__ == "__main__":
    load_and_audit_timing()
