"""
tests/test_pmdata_pipeline.py

Automated Research Pipeline Tests for PMData Audited Modeling.
Student IT23166660 - FullStrength.ai TEQS Component.

Covers:
1. Date parsing and session start time inference.
2. Absence of future/retrospective wellness matches in the audited subset.
3. Grouped-split integrity (zero participant leakage between train and test folds).
4. Prevention of post-session feature leakage (no duration, rpe, or participant_id in X).
5. Integrity of raw PMData and synthetic datasets.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

EXP_DIR = Path(r"C:\Users\rivinma\Desktop\ResearchData\pmdata_experiments")
AUDITED_CSV = EXP_DIR / "pmdata_audited_prospective_sessions.csv"
SYNTHETIC_CSV = Path(r"C:\Users\rivinma\Desktop\ResearchProject\environment-nutrition-service\data\TEQS_Synthetic_5000_Development_Dataset.csv")
RAW_DATA_ROOT = Path(r"C:\Users\rivinma\Desktop\ResearchData\pmdata_selected")


class TestPMDataPipeline(unittest.TestCase):

    def test_date_parsing_and_workout_start(self):
        """Test ISO8601 parsing and inferred workout start time calculation."""
        end_str = "2019-11-05T22:51:54.710Z"
        duration_min = 30

        end_dt = pd.to_datetime(end_str, format="ISO8601", utc=True)
        start_dt = end_dt - pd.to_timedelta(duration_min, unit="m")

        expected_start = pd.to_datetime("2019-11-05T22:21:54.710Z", format="ISO8601", utc=True)
        self.assertEqual(start_dt, expected_start)

        # Check time gap in hours
        wellness_str = "2019-11-05T08:31:40.751Z"
        wellness_dt = pd.to_datetime(wellness_str, format="ISO8601", utc=True)
        gap_hours = (start_dt - wellness_dt).total_seconds() / 3600.0

        self.assertGreater(gap_hours, 0)
        self.assertAlmostEqual(gap_hours, 13.837, places=2)

    def test_absence_of_future_wellness_matches(self):
        """Verify that every row in the audited prospective dataset occurred strictly BEFORE session start."""
        self.assertTrue(AUDITED_CSV.is_file(), f"Audited dataset missing: {AUDITED_CSV}")
        df = pd.read_csv(AUDITED_CSV)

        s_start = pd.to_datetime(df["session_start_datetime"], format="ISO8601", utc=True)
        w_time = pd.to_datetime(df["wellness_datetime"], format="ISO8601", utc=True)

        # Time gap must be strictly positive for all rows
        gaps = (s_start - w_time).dt.total_seconds() / 3600.0
        self.assertTrue((gaps > 0).all(), "Detected retrospective or concurrent wellness record in audited dataset!")
        self.assertEqual(len(df), 539, f"Expected exactly 539 audited prospective sessions, got {len(df)}")
        self.assertEqual(df["participant_id"].nunique(), 15, "Expected all 15 active participants to be present")

    def test_grouped_split_integrity(self):
        """Verify that GroupKFold strictly isolates participants between training and validation folds."""
        df = pd.read_csv(AUDITED_CSV)
        gkf = GroupKFold(n_splits=5)

        X = df[["fatigue", "mood", "readiness"]]
        y = df["session_rpe"]
        groups = df["participant_id"]

        for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            trn_pts = set(groups.iloc[trn_idx].unique())
            val_pts = set(groups.iloc[val_idx].unique())

            # Intersection must be completely empty
            overlap = trn_pts.intersection(val_pts)
            self.assertEqual(len(overlap), 0, f"Fold {fold+1} has participant overlap: {overlap}")
            self.assertGreater(len(trn_pts), 0)
            self.assertGreater(len(val_pts), 0)

    def test_prevention_of_feature_leakage(self):
        """Verify that forbidden post-session columns are not used as model predictors."""
        df = pd.read_csv(AUDITED_CSV)

        forbidden_features = {
            "session_duration_min",
            "duration_min",
            "actual_duration_min",
            "session_rpe",
            "completion_status",
            "performance_deviation_pct",
            "participant_id",
        }

        # Check modeled feature list from train_pmdata_models
        from ml.train_pmdata_models import NUMERIC_FEATURES, CATEGORICAL_FEATURES
        modeled_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)

        leaked = forbidden_features.intersection(modeled_features)
        self.assertEqual(len(leaked), 0, f"LEAKAGE DETECTED in modeled features: {leaked}")

    def test_dataset_preservation(self):
        """Verify synthetic data and raw PMData files remain intact and unmodified."""
        self.assertTrue(SYNTHETIC_CSV.is_file(), "Synthetic CSV is missing!")
        syn_df = pd.read_csv(SYNTHETIC_CSV)
        self.assertEqual(len(syn_df), 5000, "Synthetic dataset row count modified!")

        # Verify raw PMData root exists
        self.assertTrue(RAW_DATA_ROOT.is_dir(), "Raw PMData directory missing!")
        p01_w = RAW_DATA_ROOT / "p01" / "pmsys" / "wellness.csv"
        self.assertTrue(p01_w.is_file(), "p01 wellness.csv missing!")


if __name__ == "__main__":
    unittest.main()
