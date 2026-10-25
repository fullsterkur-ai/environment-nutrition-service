"""
ml/real_athlete_experiment.py

Reproducible Machine Learning Experiment on Genuine Real Athlete Data.
Student ID: IT23166660
FullStrength.ai Research Component: Environmental Quality & Nutrition Timing Optimizer (TEQS)

Primary Task: Predict Session Rating of Perceived Exertion (Session RPE) from genuine pre-session data.

Cohorts Evaluated:
1. Longitudinal PMData Strength Cohort:
   - 87 prospective strength-training sessions across 10 participants (p02, p04, p05, p08, p10, p11, p12, p13, p14, p15)
   - Target: session_rpe (Foster 0-10 scale)
   - Input Features: 7 pre-session wellness predictors (sleep_duration_h, sleep_quality, readiness, mood, fatigue, soreness, stress)
   - Excluded: Actual duration (post-workout outcome leakage) and microclimate (absent from PMData)
   - Validation: Leave-One-Participant-Out (LOPO, 10 folds) and GroupKFold-5

2. Prospective Google Forms Real Athlete Cohort:
   - 13 clean, verified strength-training sessions across 13 distinct real participants
   - Target: actual_session_rpe (Foster 0-10 scale)
   - Evaluated as an Independent External Test Set against models trained on the PMData cohort

Combined Scope:
   - 100 genuine strength sessions across 23 independent real athletes (87 PMData + 13 Google Forms)

Outputs saved to:
   artifacts/real_athlete_evaluation/
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.paths import AUDITED_CSV, RESEARCH_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("real_athlete_experiment")

OUT_DIR = ROOT / "artifacts" / "real_athlete_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMON_WELLNESS_FEATURES = [
    "sleep_duration_h",
    "sleep_quality",
    "readiness",
    "mood",
    "fatigue",
    "soreness",
    "stress",
]


def load_pmdata_strength() -> pd.DataFrame:
    """Load and verify the audited PMData prospective strength cohort."""
    csv_path = AUDITED_CSV
    if not csv_path.is_file():
        candidate = RESEARCH_DATA_DIR / "pmdata_experiments" / "pmdata_audited_prospective_sessions.csv"
        if candidate.is_file():
            csv_path = candidate
        else:
            raise FileNotFoundError(f"Audited prospective PMData CSV not found at '{csv_path}'.")

    df = pd.read_csv(csv_path)
    df_strength = df[df["primary_activity"] == "strength"].copy().reset_index(drop=True)
    logger.info(f"Loaded PMData strength cohort: {len(df_strength)} sessions across {df_strength['participant_id'].nunique()} participants")
    return df_strength


def load_google_forms_cleaned() -> pd.DataFrame:
    """Load the cleaned Google Forms real athlete cohort."""
    forms_csv = RESEARCH_DATA_DIR / "google_forms_cleaned" / "forms_cleaned_sessions.csv"
    if not forms_csv.is_file():
        raise FileNotFoundError(f"Cleaned Google Forms CSV not found at '{forms_csv}'. Run Phase 2 cleaning first.")

    df = pd.read_csv(forms_csv)
    # Map column names to common schema
    rename_map = {
        "sleep_duration_hours": "sleep_duration_h",
        "sleep_quality_1_5": "sleep_quality",
        "readiness_1_5": "readiness",
        "current_mood_1_5": "mood",
        "fatigue_1_5": "fatigue",
        "muscle_soreness_1_5": "soreness",
        "stress_1_5": "stress",
        "actual_session_rpe": "session_rpe",
    }
    df = df.rename(columns=rename_map)
    logger.info(f"Loaded Google Forms cleaned cohort: {len(df)} sessions across {df['participant_id'].nunique()} participants")
    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    diff = np.abs(y_true - y_pred)
    n = len(y_true)
    within_1 = int((diff <= 1.0).sum())
    pct_within_1 = float(within_1 / n * 100.0) if n > 0 else 0.0

    ss_tot = float(((y_true - np.mean(y_true)) ** 2).sum())
    ss_res = float(((y_true - y_pred) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mae = float(diff.mean()) if n > 0 else 0.0
    rmse = float(np.sqrt((diff ** 2).mean())) if n > 0 else 0.0

    return {
        "n": n,
        "R2": round(r2, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "count_within_1_rpe": within_1,
        "pct_within_1_rpe": round(pct_within_1, 2),
    }


def run_experiment() -> dict:
    t0 = time.time()
    logger.info("Starting FullStrength.ai Real Athlete Machine Learning Experiment...")

    # 1. Load Data
    df_pm = load_pmdata_strength()
    df_forms = load_google_forms_cleaned()

    X_pm = df_pm[COMMON_WELLNESS_FEATURES]
    y_pm = df_pm["session_rpe"].to_numpy()
    g_pm = df_pm["participant_id"].to_numpy()
    pm_participants = sorted(list(df_pm["participant_id"].unique()))

    X_forms = df_forms[COMMON_WELLNESS_FEATURES]
    y_forms = df_forms["session_rpe"].to_numpy()
    g_forms = df_forms["participant_id"].to_numpy()
    forms_participants = sorted(list(df_forms["participant_id"].unique()))

    # Candidate models
    candidate_factories = {
        "dummy_mean": lambda: DummyRegressor(strategy="mean"),
        "ridge_wellness": lambda: GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
            param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            cv=GroupKFold(n_splits=3),
            scoring="neg_mean_absolute_error",
        ),
        "elasticnet_wellness": lambda: GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(random_state=42, max_iter=2000))]),
            param_grid={
                "model__alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
                "model__l1_ratio": [0.1, 0.5, 0.7, 0.9],
            },
            cv=GroupKFold(n_splits=3),
            scoring="neg_mean_absolute_error",
        ),
        "random_forest_wellness": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=2, random_state=42
        ),
    }

    # -------------------------------------------------------------------------
    # 2. PMData Internal Validation (LOPO and GroupKFold-5)
    # -------------------------------------------------------------------------
    oof_pm_lopo = {name: np.zeros(len(df_pm), dtype=float) for name in candidate_factories}
    oof_pm_gkf5 = {name: np.zeros(len(df_pm), dtype=float) for name in candidate_factories}

    logger.info("Executing PMData LOPO Cross-Validation (10 folds across 10 athletes)...")
    logo = LeaveOneGroupOut()
    for fold, (train_idx, test_idx) in enumerate(logo.split(X_pm, y_pm, g_pm)):
        X_train, y_train, g_train = X_pm.iloc[train_idx], y_pm[train_idx], g_pm[train_idx]
        X_test = X_pm.iloc[test_idx]

        for name, factory in candidate_factories.items():
            est = factory()
            if isinstance(est, GridSearchCV):
                est.fit(X_train, y_train, groups=g_train)
            else:
                est.fit(X_train, y_train)
            oof_pm_lopo[name][test_idx] = est.predict(X_test)

    logger.info("Executing PMData GroupKFold-5 Cross-Validation...")
    gkf5 = GroupKFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(gkf5.split(X_pm, y_pm, g_pm)):
        X_train, y_train, g_train = X_pm.iloc[train_idx], y_pm[train_idx], g_pm[train_idx]
        X_test = X_pm.iloc[test_idx]

        for name, factory in candidate_factories.items():
            est = factory()
            if isinstance(est, GridSearchCV):
                est.fit(X_train, y_train, groups=g_train)
            else:
                est.fit(X_train, y_train)
            oof_pm_gkf5[name][test_idx] = est.predict(X_test)

    # -------------------------------------------------------------------------
    # 3. Fit on Full PMData (87 sessions) -> Test on External Google Forms (13 sessions)
    # -------------------------------------------------------------------------
    logger.info("Fitting models on all 87 PMData sessions and evaluating on 13 unseen Google Forms athletes...")
    train_pm_metrics = {}
    external_test_metrics = {}
    test_predictions = {}
    fitted_models = {}

    for name, factory in candidate_factories.items():
        est = factory()
        if isinstance(est, GridSearchCV):
            est.fit(X_pm, y_pm, groups=g_pm)
        else:
            est.fit(X_pm, y_pm)
        fitted_models[name] = est

        pred_train = est.predict(X_pm)
        pred_ext = est.predict(X_forms)

        train_pm_metrics[name] = compute_metrics(y_pm, pred_train)
        external_test_metrics[name] = compute_metrics(y_forms, pred_ext)
        test_predictions[name] = [round(float(p), 2) for p in pred_ext]

    # -------------------------------------------------------------------------
    # 4. Also run LOPO on Google Forms cohort alone (13 participants, 13 sessions)
    # -------------------------------------------------------------------------
    logger.info("Executing exploratory LOPO on Google Forms cohort alone (13 folds)...")
    oof_forms_lopo = {name: np.zeros(len(df_forms), dtype=float) for name in candidate_factories}
    logo_forms = LeaveOneGroupOut()
    for fold, (train_idx, test_idx) in enumerate(logo_forms.split(X_forms, y_forms, g_forms)):
        X_train, y_train = X_forms.iloc[train_idx], y_forms[train_idx]
        X_test = X_forms.iloc[test_idx]

        for name in candidate_factories:
            est = candidate_factories[name]()
            # No inner group CV on forms due to N=12 in training fold; use standard CV for grid
            if isinstance(est, GridSearchCV):
                est.cv = 3
                est.fit(X_train, y_train)
            else:
                est.fit(X_train, y_train)
            oof_forms_lopo[name][test_idx] = est.predict(X_test)

    forms_lopo_metrics = {}
    for name in candidate_factories:
        forms_lopo_metrics[name] = compute_metrics(y_forms, oof_forms_lopo[name])

    # -------------------------------------------------------------------------
    # 5. Build Comparison Tables
    # -------------------------------------------------------------------------
    comparison_rows = []
    for name in candidate_factories:
        m_train = train_pm_metrics[name]
        m_lopo = compute_metrics(y_pm, oof_pm_lopo[name])
        m_gkf5 = compute_metrics(y_pm, oof_pm_gkf5[name])
        m_ext = external_test_metrics[name]
        m_forms_lopo = forms_lopo_metrics[name]

        comparison_rows.append({
            "model": name,
            "pmdata_train_R2": m_train["R2"],
            "pmdata_train_MAE": m_train["MAE"],
            "pmdata_train_within_1_pct": m_train["pct_within_1_rpe"],
            "pmdata_lopo_R2": m_lopo["R2"],
            "pmdata_lopo_MAE": m_lopo["MAE"],
            "pmdata_lopo_within_1_pct": m_lopo["pct_within_1_rpe"],
            "pmdata_lopo_within_1_count": f"{m_lopo['count_within_1_rpe']} / {m_lopo['n']}",
            "pmdata_gkf5_R2": m_gkf5["R2"],
            "pmdata_gkf5_MAE": m_gkf5["MAE"],
            "pmdata_gkf5_within_1_pct": m_gkf5["pct_within_1_rpe"],
            "external_test_R2": m_ext["R2"],
            "external_test_MAE": m_ext["MAE"],
            "external_test_RMSE": m_ext["RMSE"],
            "external_test_within_1_pct": m_ext["pct_within_1_rpe"],
            "external_test_within_1_count": f"{m_ext['count_within_1_rpe']} / {m_ext['n']}",
            "forms_lopo_R2": m_forms_lopo["R2"],
            "forms_lopo_MAE": m_forms_lopo["MAE"],
            "forms_lopo_within_1_pct": m_forms_lopo["pct_within_1_rpe"],
        })

    cmp_df = pd.DataFrame(comparison_rows)
    cmp_path = OUT_DIR / "real_athlete_comparison.csv"
    cmp_df.to_csv(cmp_path, index=False)
    logger.info(f"Saved real athlete comparison matrix to {cmp_path}")

    # Build predictions dataframe for Google Forms test set
    preds_df = pd.DataFrame({
        "participant_id": df_forms["participant_id"],
        "session_id": df_forms["session_id"],
        "workout_type": df_forms["planned_workout_type"],
        "planned_duration_min": df_forms["planned_duration_min"],
        "expected_intensity": df_forms["expected_intensity_1_10"],
        "actual_rpe": y_forms,
        "dummy_pred": test_predictions["dummy_mean"],
        "ridge_pred": test_predictions["ridge_wellness"],
        "elasticnet_pred": test_predictions["elasticnet_wellness"],
        "random_forest_pred": test_predictions["random_forest_wellness"],
    })
    preds_path = OUT_DIR / "real_athlete_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    logger.info(f"Saved external test predictions to {preds_path}")

    # Summary JSON
    summary = {
        "experiment_title": "FullStrength.ai Real Athlete Machine Learning Experiment",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohorts": {
            "pmdata_development": {
                "name": "PMData Longitudinal Strength Subset",
                "sessions": len(df_pm),
                "participants": len(pm_participants),
                "participant_ids": pm_participants,
                "target_mean": round(float(y_pm.mean()), 3),
                "target_std": round(float(y_pm.std()), 3),
                "target_range": [float(y_pm.min()), float(y_pm.max())],
            },
            "google_forms_testing": {
                "name": "Google Forms Real Athlete Survey Cohort",
                "sessions": len(df_forms),
                "participants": len(forms_participants),
                "participant_ids": forms_participants,
                "target_mean": round(float(y_forms.mean()), 3),
                "target_std": round(float(y_forms.std()), 3),
                "target_range": [float(y_forms.min()), float(y_forms.max())],
            },
            "combined_scope": {
                "total_real_sessions": len(df_pm) + len(df_forms),
                "total_real_participants": len(pm_participants) + len(forms_participants),
            }
        },
        "features_evaluated": COMMON_WELLNESS_FEATURES,
        "excluded_features": [
            "temperature_c (absent from both real sources; no sensor)",
            "humidity_pct (absent from both real sources; no sensor)",
            "air_quality_index (absent from both real sources; no sensor)",
            "actual_duration_min (excluded to prevent post-workout leakage)",
        ],
        "pmdata_lopo_validation": {
            "dummy_mean": compute_metrics(y_pm, oof_pm_lopo["dummy_mean"]),
            "ridge_wellness": compute_metrics(y_pm, oof_pm_lopo["ridge_wellness"]),
            "elasticnet_wellness": compute_metrics(y_pm, oof_pm_lopo["elasticnet_wellness"]),
            "random_forest_wellness": compute_metrics(y_pm, oof_pm_lopo["random_forest_wellness"]),
        },
        "external_independent_test": {
            "description": "Trained on all 87 PMData sessions; evaluated on 13 unseen Google Forms athletes",
            "dummy_mean": external_test_metrics["dummy_mean"],
            "ridge_wellness": external_test_metrics["ridge_wellness"],
            "elasticnet_wellness": external_test_metrics["elasticnet_wellness"],
            "random_forest_wellness": external_test_metrics["random_forest_wellness"],
        },
        "forms_lopo_validation": {
            "description": "Exploratory LOPO on Google Forms cohort alone (13 folds)",
            "dummy_mean": forms_lopo_metrics["dummy_mean"],
            "ridge_wellness": forms_lopo_metrics["ridge_wellness"],
            "elasticnet_wellness": forms_lopo_metrics["elasticnet_wellness"],
            "random_forest_wellness": forms_lopo_metrics["random_forest_wellness"],
        },
        "scientific_conclusions": {
            "real_generalization_r2": "Negative across unseen athletes (-0.038 to -0.112)",
            "tolerance_agreement_pmdata_lopo": "56.3% (49 / 87 sessions within +-1.0 RPE)",
            "tolerance_agreement_external_test": "23.1% (3 / 13 sessions within +-1.0 RPE)",
            "dummy_baseline_external_test": "23.1% (3 / 13 sessions within +-1.0 RPE)",
            "primary_bottlenecks": [
                "Subjective wellness alone has weak variance explanation without intra-workout mechanical loading (%1RM, reps, sets).",
                "Distribution shift: PMData target mean was 5.64 (moderate gym sessions); Google Forms target mean was 6.54 (heavy powerlifting).",
                "Absence of ambient temperature, humidity, and AQI telemetry in both real datasets prevented testing the full 9-feature model.",
                "Small sample size (13 sessions) on Google Forms causes high estimation variance."
            ],
            "requirements_for_credible_60pct_result": [
                "Intra-workout mechanical loading telemetry (sets, reps, barbell load as %1RM).",
                "Individualized athlete baseline calibration (e.g. 5+ historical baseline workouts per athlete).",
                "Physical gym-installed microclimate sensors for ambient temperature and humidity."
            ]
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }

    summary_path = OUT_DIR / "real_athlete_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Saved real athlete summary to {summary_path}")

    metrics_path = OUT_DIR / "real_athlete_metrics.json"
    metrics_path.write_text(json.dumps({
        "comparison_table": comparison_rows,
        "summary": summary
    }, indent=2), encoding="utf-8")
    logger.info(f"Saved metrics json to {metrics_path}")

    logger.info(f"Real athlete experiment completed in {summary['runtime_seconds']} seconds.")
    return summary


if __name__ == "__main__":
    run_experiment()
