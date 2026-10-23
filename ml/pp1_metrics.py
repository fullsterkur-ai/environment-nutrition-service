"""Interpretable error metrics for the PP1 RPE regression (computed from saved predictions).

The primary research target stays regression (R2 / MAE / RMSE on session RPE). The extra,
easier-to-explain metric is:

    "Predictions within ±1 RPE point (%)"

= 100 * (# predictions with |actual - predicted| <= 1.0) / (# predictions in that split).
It is NOT a classification accuracy. Because RPE has a standard deviation of about 1.0, even a
constant mean predictor scores high on this metric, so it must always be shown next to the
mean-predictor baseline.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TOLERANCE = 1.0
WITHIN_LABEL = "Predictions within ±1 RPE point (%)"
_EPS = 1e-9


def within_tolerance(y_true, y_pred, tol: float = TOLERANCE) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = int(len(y_true))
    k = int((np.abs(y_true - y_pred) <= tol + _EPS).sum())
    return {"label": WITHIN_LABEL, "tolerance": tol, "count_within": k, "n": n,
            "pct": 100.0 * k / n if n else float("nan")}


def regression_summary(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "n": int(len(y_true)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "within_pm1": within_tolerance(y_true, y_pred),
        "bias_mean_error": float(err.mean()),
        "abs_error_quantiles": {f"p{q}": float(np.percentile(abs_err, q)) for q in (50, 75, 90, 95, 99)},
        "share_abs_error_gt_2": float((abs_err > 2.0).mean()),
    }
