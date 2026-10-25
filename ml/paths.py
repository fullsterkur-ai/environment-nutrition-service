"""Machine-independent path configuration for the research scripts and tests.

Project artifacts are always resolved relative to the repository root. External research
data (PMData, optionally SoccerMon) is NOT part of the repository; it is located through
environment variables so any machine can point at its own copy:

    RESEARCH_DATA_DIR        parent folder of the external datasets
                             (default: <workspace parent>/ResearchData, i.e. the folder that sits
                             next to the ResearchProject folder)
    PMDATA_RAW_DIR           raw PMData participants (p01..p16)      default: RESEARCH_DATA_DIR/pmdata_selected
    PMDATA_EXPERIMENTS_DIR   audited CSV + experiment outputs         default: RESEARCH_DATA_DIR/pmdata_experiments
    PMDATA_MERGED_CSV        merged session table                     default: RESEARCH_DATA_DIR/pmdata_merged_sessions.csv
    SOCCERMON_DIR            optional SoccerMon data (no script uses it yet)

Nothing is created at import time.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CSV = ROOT / "data" / "TEQS_Synthetic_5000_Development_Dataset.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
PP1_DIR = ARTIFACTS_DIR / "pp1_model_optimization"


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else default


RESEARCH_DATA_DIR = _env_path("RESEARCH_DATA_DIR", ROOT.parent.parent / "ResearchData")
PMDATA_RAW_DIR = _env_path("PMDATA_RAW_DIR", RESEARCH_DATA_DIR / "pmdata_selected")
PMDATA_EXPERIMENTS_DIR = _env_path("PMDATA_EXPERIMENTS_DIR", RESEARCH_DATA_DIR / "pmdata_experiments")
PMDATA_MERGED_CSV = _env_path("PMDATA_MERGED_CSV", RESEARCH_DATA_DIR / "pmdata_merged_sessions.csv")
PMDATA_MERGE_REPORT = RESEARCH_DATA_DIR / "pmdata_merge_report.txt"
PMDATA_PROFILE_REPORT = RESEARCH_DATA_DIR / "pmdata_profile.txt"
AUDITED_CSV = PMDATA_EXPERIMENTS_DIR / "pmdata_audited_prospective_sessions.csv"
SOCCERMON_DIR = _env_path("SOCCERMON_DIR", RESEARCH_DATA_DIR / "soccermon")


def _missing_message(path: Path, what: str, env_var: str) -> str:
    return (f"{what} not found at '{path}'. External research data is not stored in the repository. "
            f"Copy it to that location or set the environment variable {env_var} to its folder "
            f"(see README, section 'External research data').")


def require_file(path: Path, what: str, env_var: str) -> Path:
    if not Path(path).is_file():
        raise FileNotFoundError(_missing_message(path, what, env_var))
    return Path(path)


def require_dir(path: Path, what: str, env_var: str) -> Path:
    if not Path(path).is_dir():
        raise FileNotFoundError(_missing_message(path, what, env_var))
    return Path(path)
