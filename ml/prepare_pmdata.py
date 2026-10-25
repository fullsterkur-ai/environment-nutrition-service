"""
prepare_pmdata.py  -  Merge PMData wellness + sRPE into a session-level dataset.

Read-only access to raw CSVs.  Output goes OUTSIDE the Git repository.

Matching strategy:
  For each usable sRPE session, find the latest wellness record from the
  SAME participant on the SAME calendar date.  If none exists, keep the
  session with wellness fields set to NaN (no cross-participant or
  cross-date matching).

Does NOT impute, does NOT merge with synthetic TEQS data.
"""

import os
import sys
import ast
import json
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime

# ── Windows console encoding fix ─────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── paths ────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.paths import (PMDATA_RAW_DIR, PMDATA_MERGED_CSV, PMDATA_MERGE_REPORT,  # noqa: E402
                      require_dir)

require_dir(PMDATA_RAW_DIR, "Raw PMData folder (p01..p16)", "PMDATA_RAW_DIR")
DATA_ROOT    = str(PMDATA_RAW_DIR)
OUTPUT_CSV   = str(PMDATA_MERGED_CSV)
REPORT_PATH  = str(PMDATA_MERGE_REPORT)
PARTICIPANTS = [f"p{i:02d}" for i in range(1, 17)]

# ── report buffer ────────────────────────────────────────────────────────────
report_buf = StringIO()


def log(msg: str = "") -> None:
    """Print to terminal AND write to report buffer."""
    print(msg)
    report_buf.write(msg + "\n")


def section(title: str) -> None:
    log(f"\n{'=' * 72}")
    log(f"  {title}")
    log(f"{'=' * 72}")


# ── helpers ──────────────────────────────────────────────────────────────────
def parse_activity_names(raw: str) -> list[str]:
    """Safely parse the JSON-like activity_names field.

    Returns a list of activity strings.  Never silently discards names.
    """
    if pd.isna(raw):
        return []
    raw = str(raw).strip()
    # Try JSON first (handles single-quoted Python lists too after replacement)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
        return [str(parsed).strip()]
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to ast.literal_eval for Python-style lists
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed]
        return [str(parsed).strip()]
    except (ValueError, SyntaxError):
        pass
    # Last resort: return as single-element list
    return [raw.strip()]


def primary_activity(names: list[str]) -> str:
    """Return the most specific activity name (last non-category element).

    PMData convention: ['individual', 'running'] -> 'running'
                       ['team', 'soccer']        -> 'soccer'
    """
    category_words = {"individual", "team", "group", "other"}
    specific = [n for n in names if n.lower() not in category_words]
    if specific:
        return specific[-1]
    return names[-1] if names else "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#   PHASE 1 : Load all raw data
# ══════════════════════════════════════════════════════════════════════════════
section("Phase 1 - Loading raw data")

all_wellness: list[pd.DataFrame] = []
all_srpe: list[pd.DataFrame] = []
raw_wellness_total = 0
raw_srpe_total = 0
dropped_per_participant: dict[str, dict[str, int]] = {}

for pid in PARTICIPANTS:
    w_path = os.path.join(DATA_ROOT, pid, "pmsys", "wellness.csv")
    s_path = os.path.join(DATA_ROOT, pid, "pmsys", "srpe.csv")

    # -- wellness ----------------------------------------------------------
    wdf = pd.read_csv(w_path)
    wdf["participant_id"] = pid
    raw_wellness_total += len(wdf)
    all_wellness.append(wdf)

    # -- srpe --------------------------------------------------------------
    sdf = pd.read_csv(s_path)
    sdf["participant_id"] = pid
    raw_srpe_total += len(sdf)

    n_before = len(sdf)
    # Drop rows where perceived_exertion OR duration_min is missing
    mask_rpe_miss = sdf["perceived_exertion"].isna()
    mask_dur_miss = sdf["duration_min"].isna()
    mask_drop = mask_rpe_miss | mask_dur_miss
    n_dropped = int(mask_drop.sum())
    sdf_clean = sdf[~mask_drop].copy()

    dropped_per_participant[pid] = {
        "raw": n_before,
        "dropped_rpe_missing": int(mask_rpe_miss.sum()),
        "dropped_dur_missing": int(mask_dur_miss.sum()),
        "dropped_total": n_dropped,
        "usable": len(sdf_clean),
    }
    log(f"  {pid}  wellness={len(wdf):>4d}   srpe_raw={n_before:>4d}   "
        f"dropped={n_dropped:>3d}   srpe_usable={len(sdf_clean):>4d}")

    all_srpe.append(sdf_clean)

wellness_all = pd.concat(all_wellness, ignore_index=True)
srpe_all = pd.concat(all_srpe, ignore_index=True)

usable_srpe_total = len(srpe_all)
log(f"\n  Raw wellness total  : {raw_wellness_total}")
log(f"  Raw sRPE total      : {raw_srpe_total}")
log(f"  Usable sRPE total   : {usable_srpe_total}")

# ══════════════════════════════════════════════════════════════════════════════
#   PHASE 2 : Parse datetimes
# ══════════════════════════════════════════════════════════════════════════════
section("Phase 2 - Parsing datetimes")

wellness_all["wellness_datetime"] = pd.to_datetime(
    wellness_all["effective_time_frame"], utc=True, errors="coerce"
)
wellness_all["wellness_date"] = wellness_all["wellness_datetime"].dt.date

srpe_all["session_datetime"] = pd.to_datetime(
    srpe_all["end_date_time"], utc=True, errors="coerce"
)
srpe_all["session_date"] = srpe_all["session_datetime"].dt.date

n_bad_well_dt = int(wellness_all["wellness_datetime"].isna().sum())
n_bad_srpe_dt = int(srpe_all["session_datetime"].isna().sum())
log(f"  Unparseable wellness datetimes : {n_bad_well_dt}")
log(f"  Unparseable sRPE datetimes     : {n_bad_srpe_dt}")

# ══════════════════════════════════════════════════════════════════════════════
#   PHASE 3 : Parse activity names
# ══════════════════════════════════════════════════════════════════════════════
section("Phase 3 - Parsing activity names")

srpe_all["activity_names_raw"] = srpe_all["activity_names"].astype(str)
srpe_all["_parsed_activities"] = srpe_all["activity_names"].apply(parse_activity_names)
srpe_all["primary_activity"] = srpe_all["_parsed_activities"].apply(primary_activity)

activity_counts = srpe_all["primary_activity"].value_counts()
log("  Activity distribution:")
for act, cnt in activity_counts.items():
    log(f"    {act:30s}  {cnt:>5d}  ({100 * cnt / len(srpe_all):5.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
#   PHASE 4 : Same-date wellness matching
# ══════════════════════════════════════════════════════════════════════════════
section("Phase 4 - Wellness matching (same participant, same date)")

# For each participant+date, keep the LATEST wellness record
wellness_all_sorted = wellness_all.sort_values("wellness_datetime")
latest_wellness = (
    wellness_all_sorted
    .groupby(["participant_id", "wellness_date"], as_index=False)
    .last()
)

# Build a lookup key
latest_wellness["_merge_key"] = (
    latest_wellness["participant_id"] + "_" +
    latest_wellness["wellness_date"].astype(str)
)
srpe_all["_merge_key"] = (
    srpe_all["participant_id"] + "_" +
    srpe_all["session_date"].astype(str)
)

# Select wellness columns to bring in
wellness_cols_to_merge = [
    "_merge_key", "wellness_datetime",
    "fatigue", "mood", "readiness",
    "sleep_duration_h", "sleep_quality", "soreness", "stress",
]

merged = srpe_all.merge(
    latest_wellness[wellness_cols_to_merge],
    on="_merge_key",
    how="left",  # keep ALL usable sRPE sessions
)

n_matched = int(merged["wellness_datetime"].notna().sum())
n_unmatched = len(merged) - n_matched
pct_matched = 100 * n_matched / len(merged) if len(merged) > 0 else 0
pct_unmatched = 100 * n_unmatched / len(merged) if len(merged) > 0 else 0

log(f"  Sessions with same-day wellness match : {n_matched:>5d}  ({pct_matched:.1f}%)")
log(f"  Sessions WITHOUT wellness match       : {n_unmatched:>5d}  ({pct_unmatched:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
#   PHASE 5 : Build final dataset
# ══════════════════════════════════════════════════════════════════════════════
section("Phase 5 - Building final dataset")

# Rename columns to match the required schema
merged = merged.rename(columns={
    "perceived_exertion": "session_rpe",
    "duration_min": "session_duration_min",
})

final_columns = [
    "participant_id",
    "session_datetime",
    "session_date",
    "activity_names_raw",
    "primary_activity",
    "session_rpe",
    "session_duration_min",
    "wellness_datetime",
    "fatigue",
    "mood",
    "readiness",
    "sleep_duration_h",
    "sleep_quality",
    "soreness",
    "stress",
]

final = merged[final_columns].copy()

# ── Remove exact duplicates ──────────────────────────────────────────────────
n_before_dedup = len(final)
final = final.drop_duplicates()
n_duplicates_removed = n_before_dedup - len(final)
log(f"  Exact duplicates removed : {n_duplicates_removed}")

# ── Sort for reproducibility ─────────────────────────────────────────────────
final = final.sort_values(
    ["participant_id", "session_datetime"]
).reset_index(drop=True)

log(f"  Final dataset rows       : {len(final)}")
log(f"  Final participant count  : {final['participant_id'].nunique()}")

# ══════════════════════════════════════════════════════════════════════════════
#   PHASE 6 : Save
# ══════════════════════════════════════════════════════════════════════════════
section("Phase 6 - Saving outputs")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
final.to_csv(OUTPUT_CSV, index=False)
log(f"  Merged CSV -> {OUTPUT_CSV}")

# ══════════════════════════════════════════════════════════════════════════════
#   DATA-QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════════
section("Data-Quality Report")

log(f"\n  --- Record Counts ---")
log(f"  Raw wellness records          : {raw_wellness_total}")
log(f"  Raw sRPE records              : {raw_srpe_total}")
log(f"  Usable sRPE (after drop)      : {usable_srpe_total}")
log(f"  Final merged rows             : {len(final)}")
log(f"  Final participants            : {final['participant_id'].nunique()}")
log(f"  Duplicates removed            : {n_duplicates_removed}")

log(f"\n  --- Dropped sRPE Rows per Participant ---")
for pid in PARTICIPANTS:
    d = dropped_per_participant[pid]
    log(f"    {pid}  raw={d['raw']:>4d}  dropped={d['dropped_total']:>3d}  "
        f"(rpe_miss={d['dropped_rpe_missing']}, dur_miss={d['dropped_dur_missing']})  "
        f"usable={d['usable']:>4d}")

log(f"\n  --- Wellness Matching ---")
log(f"  Sessions with same-day match  : {n_matched:>5d}  ({pct_matched:.1f}%)")
log(f"  Sessions without match        : {n_unmatched:>5d}  ({pct_unmatched:.1f}%)")

log(f"\n  --- Missing Values per Final Column ---")
for col in final_columns:
    n_miss = int(final[col].isna().sum())
    pct = 100 * n_miss / len(final) if len(final) > 0 else 0
    log(f"    {col:30s}  {n_miss:>5d}  ({pct:5.1f}%)")

log(f"\n  --- Activity Distribution ---")
for act, cnt in activity_counts.items():
    log(f"    {act:30s}  {cnt:>5d}  ({100 * cnt / usable_srpe_total:5.1f}%)")

log(f"\n  --- Session RPE Descriptive Statistics ---")
rpe_desc = final["session_rpe"].describe()
for stat_name, val in rpe_desc.items():
    log(f"    {stat_name:10s}  {val:8.2f}")

log(f"\n  --- Duration (min) Descriptive Statistics ---")
dur_desc = final["session_duration_min"].describe()
for stat_name, val in dur_desc.items():
    log(f"    {stat_name:10s}  {val:8.2f}")

log(f"\n  --- Participant-Level Session Counts ---")
pid_counts = final["participant_id"].value_counts().sort_index()
for pid, cnt in pid_counts.items():
    log(f"    {pid}  {cnt:>5d} sessions")

# ── save report ──────────────────────────────────────────────────────────────
section("Report Saved")
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as fh:
    fh.write(report_buf.getvalue())
log(f"  [OK] Full report -> {REPORT_PATH}")
