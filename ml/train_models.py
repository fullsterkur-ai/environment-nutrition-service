
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, GroupKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.inspection import permutation_importance


RANDOM_STATE = 23166660

# IMPORTANT:
# Only pre-session variables are used as model inputs.
# Post-session variables are targets/calibration outputs and MUST NOT be used as predictors.
REGRESSION_TARGET = "session_rpe_1_10"
CLASSIFICATION_TARGET = "completion_status"
GROUP_COLUMN = "participant_id"

NUMERIC_FEATURES = [
    "temperature_c",
    "humidity_pct",
    "air_quality_index",
    "sleep_hours",
    "sleep_quality_1_5",
    "mood_readiness_1_5",
    "planned_duration_min",
    "planned_intensity_1_10",
]

CATEGORICAL_FEATURES = ["workout_type"]

LEAKAGE_COLUMNS = [
    "session_id",
    "timestamp",
    "actual_duration_min",            # known after/during session
    "session_rpe_1_10",              # regression target
    "completion_status",             # classification target
    "performance_deviation_pct",      # post-session outcome
    "data_type",
]


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [
        REGRESSION_TARGET, CLASSIFICATION_TARGET, GROUP_COLUMN
    ])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Basic type coercion
    for col in NUMERIC_FEATURES + [REGRESSION_TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def group_train_test_split(df: pd.DataFrame, test_size=0.20):
    """
    Split by participant, not by row.
    This prevents sessions from the same participant appearing in both train and test sets.
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(
        splitter.split(df, groups=df[GROUP_COLUMN])
    )
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def evaluate_regression(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_regression_models(train_df, test_df, outdir: Path):
    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[REGRESSION_TARGET]
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[REGRESSION_TARGET]

    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=350,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    fitted = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = evaluate_regression(y_test, pred)
        results[name] = metrics
        fitted[name] = pipe
        print(f"[Regression] {name}: {metrics}")

    best_name = min(results, key=lambda n: results[n]["MAE"])
    best_pipe = fitted[best_name]
    joblib.dump(best_pipe, outdir / "best_rpe_model.joblib")

    # Permutation importance on raw feature columns; robust and model-agnostic.
    perm = permutation_importance(
        best_pipe,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    imp = pd.DataFrame({
        "feature": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    imp.to_csv(outdir / "rpe_feature_importance.csv", index=False)

    return {
        "best_model": best_name,
        "metrics": results,
    }


def train_completion_classifier(train_df, test_df, outdir: Path):
    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[CLASSIFICATION_TARGET].astype(str)
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[CLASSIFICATION_TARGET].astype(str)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting_classifier": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    fitted = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
        }
        results[name] = metrics
        fitted[name] = pipe
        print(f"[Classification] {name}: {metrics}")

    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    best_pipe = fitted[best_name]
    joblib.dump(best_pipe, outdir / "best_completion_model.joblib")

    report = classification_report(
        y_test,
        best_pipe.predict(X_test),
        output_dict=True,
        zero_division=0
    )
    with open(outdir / "completion_classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return {
        "best_model": best_name,
        "metrics": results,
    }


def tune_random_forest_regressor(train_df, outdir: Path):
    """
    OPTIONAL advanced step:
    participant-grouped CV + randomized hyperparameter search.
    Run this only after the basic pipeline works.
    """
    X = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = train_df[REGRESSION_TARGET]
    groups = train_df[GROUP_COLUMN]

    pipe = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    params = {
        "model__n_estimators": [200, 350, 500, 700],
        "model__max_depth": [None, 6, 10, 16, 24],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", 0.6, 0.8, 1.0],
    }

    cv = GroupKFold(n_splits=5)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=params,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y, groups=groups)

    joblib.dump(search.best_estimator_, outdir / "tuned_rpe_model.joblib")
    with open(outdir / "tuned_rpe_params.json", "w") as f:
        json.dump(search.best_params_, f, indent=2)

    return search.best_params_, -search.best_score_


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--outdir", default="artifacts", help="Output directory")
    parser.add_argument("--tune", action="store_true", help="Run optional RF hyperparameter tuning")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv)
    train_df, test_df = group_train_test_split(df)

    # Save split membership for reproducibility.
    train_df[[GROUP_COLUMN]].drop_duplicates().to_csv(outdir / "train_participants.csv", index=False)
    test_df[[GROUP_COLUMN]].drop_duplicates().to_csv(outdir / "test_participants.csv", index=False)

    regression_summary = train_regression_models(train_df, test_df, outdir)
    classification_summary = train_completion_classifier(train_df, test_df, outdir)

    summary = {
        "IMPORTANT_NOTE": (
            "Metrics produced from synthetic data are for software/pipeline testing only "
            "and must not be reported as scientific model performance."
        ),
        "rows_total": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_participants": int(train_df[GROUP_COLUMN].nunique()),
        "test_participants": int(test_df[GROUP_COLUMN].nunique()),
        "regression": regression_summary,
        "classification": classification_summary,
    }

    if args.tune:
        best_params, cv_mae = tune_random_forest_regressor(train_df, outdir)
        summary["optional_tuning"] = {
            "best_params": best_params,
            "group_cv_mae": float(cv_mae),
        }

    with open(outdir / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved artifacts to:", outdir.resolve())
    print("IMPORTANT: synthetic metrics are NOT research evidence.")


if __name__ == "__main__":
    main()
