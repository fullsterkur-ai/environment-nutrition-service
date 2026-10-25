"""
Pre-session feature engineering for the PP1 RPE optimisation.

Only columns known BEFORE the workout are used. All constants below are fixed,
domain-motivated reference points (NOT fitted on any data), so the transformer is
stateless and cannot leak information from validation / test participants.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BASE_NUMERIC = [
    "temperature_c", "humidity_pct", "air_quality_index", "sleep_hours",
    "sleep_quality_1_5", "mood_readiness_1_5", "planned_duration_min",
    "planned_intensity_1_10",
]
CATEGORICAL = ["workout_type"]
RAW_FEATURES = BASE_NUMERIC + CATEGORICAL

ENGINEERED_NUMERIC = [
    "planned_load",            # intensity x duration (planned training load, /100)
    "intensity_sq",            # non-linear intensity effect
    "duration_log",            # diminishing effect of long sessions
    "heat_humidity",           # temperature x humidity interaction (/100)
    "heat_excess_sq",          # squared excess above a 24 C reference
    "humidity_excess",         # humidity above 60 %
    "aqi_excess",              # AQI above 50 (moderate threshold)
    "sleep_deficit",           # shortfall relative to 7 h
    "sleep_oversleep",         # excess above 9 h
    "recovery_index",          # sleep_quality x mood readiness (/5)
    "readiness_mean",          # mean of sleep quality & readiness
    "heat_x_intensity",        # heat stress amplified by hard planned work
    "poor_recovery_x_intensity",  # low readiness amplified by hard planned work
    "env_stress_index",        # combined heat + humidity + AQI burden (0..~3)
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t, h, a = out["temperature_c"], out["humidity_pct"], out["air_quality_index"]
    sh, sq, mr = out["sleep_hours"], out["sleep_quality_1_5"], out["mood_readiness_1_5"]
    dur, inten = out["planned_duration_min"], out["planned_intensity_1_10"]

    out["planned_load"] = inten * dur / 100.0
    out["intensity_sq"] = inten ** 2
    out["duration_log"] = np.log1p(dur)
    out["heat_humidity"] = t * h / 100.0
    out["heat_excess_sq"] = np.clip(t - 24.0, 0, None) ** 2
    out["humidity_excess"] = np.clip(h - 60.0, 0, None)
    out["aqi_excess"] = np.clip(a - 50.0, 0, None)
    out["sleep_deficit"] = np.clip(7.0 - sh, 0, None)
    out["sleep_oversleep"] = np.clip(sh - 9.0, 0, None)
    out["recovery_index"] = sq * mr / 5.0
    out["readiness_mean"] = (sq + mr) / 2.0
    out["heat_x_intensity"] = np.clip(t - 24.0, 0, None) * inten / 10.0
    out["poor_recovery_x_intensity"] = (5.0 - out["readiness_mean"]) * inten / 5.0
    out["env_stress_index"] = (
        np.clip(t - 24.0, 0, None) / 12.0
        + np.clip(h - 60.0, 0, None) / 40.0
        + np.clip(a - 50.0, 0, None) / 100.0
    )
    return out


class PreSessionFeatures(BaseEstimator, TransformerMixin):
    """Stateless feature builder. enabled=False passes the raw 9 inputs through."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        missing = [c for c in RAW_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Missing pre-session columns: {missing}")
        X = X[RAW_FEATURES]
        return engineer(X) if self.enabled else X

    def get_feature_names_out(self, input_features=None):
        return np.array(RAW_FEATURES + (ENGINEERED_NUMERIC if self.enabled else []))


def numeric_columns(enabled: bool):
    return BASE_NUMERIC + (ENGINEERED_NUMERIC if enabled else [])
