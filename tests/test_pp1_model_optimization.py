"""Tests for the PP1 optimisation artifacts (synthetic development experiment)."""
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from ml.pp1_features import RAW_FEATURES, engineer, ENGINEERED_NUMERIC
from ml.pp1_splits import GROUP, TARGET, make_splits

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts" / "pp1_model_optimization"
LEAK = {"session_id", "timestamp", "actual_duration_min", "session_rpe_1_10",
        "completion_status", "performance_deviation_pct", "data_type", "participant_id"}


def test_no_leakage_columns_in_features():
    assert not LEAK & set(RAW_FEATURES)


def test_splits_are_participant_disjoint_and_cover_all():
    df, parts = make_splits()
    ids = [set(p[GROUP]) for p in parts.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(p) for p in parts.values()) == len(df)
    assert {k: p[GROUP].nunique() for k, p in parts.items()} == {
        "dev": 150, "sealed_holdout": 50, "legacy_holdout": 50}


def test_sealed_holdout_not_in_legacy_test_set():
    _, parts = make_splits()
    legacy = set(pd.read_csv(ROOT / "artifacts" / "test_participants.csv")[GROUP])
    assert not set(parts["sealed_holdout"][GROUP]) & legacy


def test_feature_engineering_is_stateless_and_complete():
    df, _ = make_splits()
    out = engineer(df[RAW_FEATURES].head(20))
    assert set(ENGINEERED_NUMERIC) <= set(out.columns)
    assert not out[ENGINEERED_NUMERIC].isna().any().any()


@pytest.mark.skipif(not (EXP / "best_rpe_model_pp1.joblib").is_file(), reason="run ml.pp1_optimize first")
def test_saved_model_loads_and_predicts_reproducible_metrics():
    model = joblib.load(EXP / "best_rpe_model_pp1.joblib")
    _, parts = make_splits()
    sealed = parts["sealed_holdout"]
    pred = model.predict(sealed[RAW_FEATURES])
    from sklearn.metrics import r2_score
    saved = json.loads((EXP / "final_holdout_metrics.json").read_text())
    expected = saved["holdout_evaluations"]["sealed_holdout/final_model"]["R2"]
    assert abs(r2_score(sealed[TARGET], pred) - expected) < 1e-9


@pytest.mark.skipif(not (EXP / "final_holdout_metrics.json").is_file(), reason="run ml.pp1_optimize first")
def test_result_file_flag_matches_metric():
    saved = json.loads((EXP / "final_holdout_metrics.json").read_text())
    r2 = saved["holdout_evaluations"]["sealed_holdout/final_model"]["R2"]
    assert saved["target_R2_0.60_achieved_on_sealed_holdout"] == (r2 >= 0.60)


@pytest.mark.skipif(not (EXP / "best_rpe_model_pp1.joblib").is_file(), reason="run ml.pp1_optimize first")
def test_single_record_prediction_in_plausible_range():
    model = joblib.load(EXP / "best_rpe_model_pp1.joblib")
    row = pd.DataFrame([{"temperature_c": 31, "humidity_pct": 78, "air_quality_index": 60,
                         "sleep_hours": 7, "sleep_quality_1_5": 3, "mood_readiness_1_5": 3,
                         "planned_duration_min": 70, "planned_intensity_1_10": 7.5,
                         "workout_type": "Squat"}])
    assert 1.0 <= float(model.predict(row)[0]) <= 10.0


@pytest.mark.skipif(not (EXP / "interpretable_metrics.json").is_file(), reason="run ml.pp1_report_artifacts first")
def test_interpretable_tolerance_accuracy_metric():
    data = json.loads((EXP / "interpretable_metrics.json").read_text())
    assert "metric_label" in data
    assert data["metric_label"] == "Predictions within ±1 RPE point (%)"
    dev_final = data["synthetic_dev_groupkfold_oof"]["final_model"]
    assert "within_pm1" in dev_final
    pct = dev_final["within_pm1"]["pct"]
    assert pct >= 85.0  # 86.9% of predictions within +-1 RPE
    mean_pred = data["synthetic_dev_groupkfold_oof"]["mean_predictor"]["within_pm1"]["pct"]
    assert pct > mean_pred  # Outperforms constant mean baseline


@pytest.mark.skipif(not (EXP / "actual_vs_predicted_holdout.csv").is_file(), reason="run ml.pp1_optimize first")
def test_visual_reporting_artifacts_exist():
    # Verify core visual plotting datasets exist and are non-empty
    for csv_name in ("actual_vs_predicted_holdout.csv", "experiment_comparison.csv", "feature_importance.csv", "learning_curve.csv"):
        f = EXP / csv_name
        assert f.is_file(), f"Missing visual data artifact: {csv_name}"
        assert f.stat().st_size > 0

    # Verify high-resolution generated visual representation charts
    for png_name in ("actual_vs_predicted.png", "model_comparison.png", "feature_importance.png", "learning_curve.png"):
        p = EXP / png_name
        assert p.is_file(), f"Missing visual chart: {png_name}"
        assert p.stat().st_size > 1024  # Valid PNG image size

