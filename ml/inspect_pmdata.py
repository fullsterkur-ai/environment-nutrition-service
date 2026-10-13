"""
inspect_pmdata.py  -  Read-only profiling of the PMData wellness & sRPE CSVs.

* Iterates p01-p16
* Reports row counts, column names, dtypes, missing values
* Checks column-structure consistency across participants
* Shows first 3 rows from p01
* Saves a full report to pmdata_profile.txt
* Does NOT modify any raw file
"""

import os
import sys
import pandas as pd
from io import StringIO
from datetime import datetime

# Force UTF-8 on Windows console to avoid charmap errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── paths ────────────────────────────────────────────────────────────────────
DATA_ROOT = r"C:\Users\rivinma\Desktop\ResearchData\pmdata_selected"
REPORT_PATH = r"C:\Users\rivinma\Desktop\ResearchData\pmdata_profile.txt"
PARTICIPANTS = [f"p{i:02d}" for i in range(1, 17)]

# ── accumulators ─────────────────────────────────────────────────────────────
wellness_col_sets: dict[str, list[str]] = {}
srpe_col_sets: dict[str, list[str]] = {}

wellness_total_rows = 0
srpe_total_rows = 0

missing_files: list[str] = []

report_buf = StringIO()


def log(msg: str = "") -> None:
    """Print to terminal AND write to report buffer."""
    print(msg)
    report_buf.write(msg + "\n")


def section(title: str) -> None:
    log(f"\n{'=' * 72}")
    log(f"  {title}")
    log(f"{'=' * 72}")


# ── main loop ────────────────────────────────────────────────────────────────
section("PMData Inspection Report")
log(f"Generated: {datetime.now().isoformat()}")
log(f"Data root: {DATA_ROOT}")

p01_wellness_df: pd.DataFrame | None = None
p01_srpe_df: pd.DataFrame | None = None

for pid in PARTICIPANTS:
    section(f"Participant: {pid}")

    wellness_path = os.path.join(DATA_ROOT, pid, "pmsys", "wellness.csv")
    srpe_path = os.path.join(DATA_ROOT, pid, "pmsys", "srpe.csv")

    # ── wellness ─────────────────────────────────────────────────────────
    if os.path.isfile(wellness_path):
        wdf = pd.read_csv(wellness_path)
        n_rows = len(wdf)
        wellness_total_rows += n_rows
        wellness_col_sets[pid] = list(wdf.columns)

        log(f"  wellness.csv  ->  {n_rows} rows,  {len(wdf.columns)} columns")
        log(f"  Columns : {list(wdf.columns)}")
        log(f"  Dtypes  :")
        for col in wdf.columns:
            log(f"    {col:30s}  {str(wdf[col].dtype)}")
        log(f"  Missing values:")
        for col in wdf.columns:
            n_miss = int(wdf[col].isna().sum())
            pct = 100 * n_miss / n_rows if n_rows > 0 else 0
            log(f"    {col:30s}  {n_miss:>5d}  ({pct:5.1f}%)")

        if pid == "p01":
            p01_wellness_df = wdf
    else:
        missing_files.append(wellness_path)
        log(f"  *** wellness.csv  ->  FILE MISSING ***")

    # ── srpe ─────────────────────────────────────────────────────────────
    if os.path.isfile(srpe_path):
        sdf = pd.read_csv(srpe_path)
        n_rows_s = len(sdf)
        srpe_total_rows += n_rows_s
        srpe_col_sets[pid] = list(sdf.columns)

        log(f"  srpe.csv      ->  {n_rows_s} rows,  {len(sdf.columns)} columns")
        log(f"  Columns : {list(sdf.columns)}")
        log(f"  Dtypes  :")
        for col in sdf.columns:
            log(f"    {col:30s}  {str(sdf[col].dtype)}")
        log(f"  Missing values:")
        for col in sdf.columns:
            n_miss = int(sdf[col].isna().sum())
            pct = 100 * n_miss / n_rows_s if n_rows_s > 0 else 0
            log(f"    {col:30s}  {n_miss:>5d}  ({pct:5.1f}%)")

        if pid == "p01":
            p01_srpe_df = sdf
    else:
        missing_files.append(srpe_path)
        log(f"  *** srpe.csv  ->  FILE MISSING ***")

# ── column consistency check ─────────────────────────────────────────────────
section("Column-Structure Consistency")

if wellness_col_sets:
    ref_pid = list(wellness_col_sets.keys())[0]
    ref_cols = wellness_col_sets[ref_pid]
    wellness_consistent = True
    for pid, cols in wellness_col_sets.items():
        if cols != ref_cols:
            wellness_consistent = False
            log(f"  MISMATCH in wellness — {pid} has {cols}  (ref {ref_pid} has {ref_cols})")
    if wellness_consistent:
        log(f"  wellness.csv columns are CONSISTENT across all {len(wellness_col_sets)} participants.")
        log(f"  Reference columns: {ref_cols}")

if srpe_col_sets:
    ref_pid_s = list(srpe_col_sets.keys())[0]
    ref_cols_s = srpe_col_sets[ref_pid_s]
    srpe_consistent = True
    for pid, cols in srpe_col_sets.items():
        if cols != ref_cols_s:
            srpe_consistent = False
            log(f"  MISMATCH in srpe — {pid} has {cols}  (ref {ref_pid_s} has {ref_cols_s})")
    if srpe_consistent:
        log(f"  srpe.csv columns are CONSISTENT across all {len(srpe_col_sets)} participants.")
        log(f"  Reference columns: {ref_cols_s}")

# ── sample rows from p01 ─────────────────────────────────────────────────────
section("Sample Rows — p01 wellness.csv (first 3)")
if p01_wellness_df is not None:
    log(p01_wellness_df.head(3).to_string(index=False))
else:
    log("  (not available)")

section("Sample Rows — p01 srpe.csv (first 3)")
if p01_srpe_df is not None:
    log(p01_srpe_df.head(3).to_string(index=False))
else:
    log("  (not available)")

# ── missing files ─────────────────────────────────────────────────────────────
section("Missing Files")
if missing_files:
    for f in missing_files:
        log(f"  • {f}")
else:
    log("  None — all expected files found.")

# ── final summary ─────────────────────────────────────────────────────────────
section("Final Summary")
log(f"  Participants scanned   : {len(PARTICIPANTS)}")
log(f"  Total wellness rows    : {wellness_total_rows}")
log(f"  Total sRPE rows        : {srpe_total_rows}")
log(f"  Missing files          : {len(missing_files)}")
if wellness_col_sets:
    log(f"  Wellness columns       : {wellness_col_sets[list(wellness_col_sets.keys())[0]]}")
if srpe_col_sets:
    log(f"  sRPE columns           : {srpe_col_sets[list(srpe_col_sets.keys())[0]]}")

# ── save report ──────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as fh:
    fh.write(report_buf.getvalue())
log(f"\n[OK] Full report saved to: {REPORT_PATH}")
