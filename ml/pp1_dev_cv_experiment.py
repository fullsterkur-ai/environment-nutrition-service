"""Small, bounded DEVELOPMENT-ONLY experiment for the R2 >= 0.60 question.

Run from the project root:
    python -m ml.pp1_dev_cv_experiment

Scientific rules enforced here:
  * Only the 150 DEV participants are loaded/used (``make_splits()["dev"]``). The sealed and
    legacy holdouts are never read, so they cannot influence anything.
  * Participant-grouped GroupKFold(5) with the SAME folds as ml/pp1_optimize.py.
  * Hyper-parameters of every NEW candidate are chosen by an INNER GroupKFold(3) on the
    outer-training participants only (nested CV), so the outer score is not selection-biased.
  * Only the 9 legitimate pre-session inputs are used. No participant identity, session RPE,
    actual duration, timestamp or any post-session field. Target/labels/data are untouched.
  * Results are DEVELOPMENT-CV evidence. A newly chosen candidate would still need data that
    was not used for selection before any claim about the 0.60 target can be made.

Outputs (new file): artifacts/pp1_model_optimization/dev_cv_new_experiment.json
"""
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, SplineTransformer, StandardScaler

from ml.paths import PP1_DIR
from ml.pp1_features import BASE_NUMERIC, CATEGORICAL, RAW_FEATURES
from ml.pp1_metrics import regression_summary
from ml.pp1_optimize import make_pipe
from ml.pp1_splits import GROUP, TARGET, make_splits

SEED = 23166660
OUTER_FOLDS, INNER_FOLDS, N_BOOT = 5, 3, 1000
GRID = {"model__alpha": [0.003, 0.01, 0.03, 0.1, 0.3], "model__l1_ratio": [0.2, 0.6, 0.9]}


def _cat():
    return Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))])


def spline_pipe():
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sp", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
                    ("sc", StandardScaler())])
    return Pipeline([("pre", ColumnTransformer([("num", num, BASE_NUMERIC), ("cat", _cat(), CATEGORICAL)])),
                     ("model", ElasticNet(max_iter=50000))])


def interaction_pipe():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)), ("sc2", StandardScaler())])
    return Pipeline([("pre", ColumnTransformer([("num", num, BASE_NUMERIC), ("cat", _cat(), CATEGORICAL)])),
                     ("model", ElasticNet(max_iter=50000))])


def nested_oof(pipe, X, y, g, cv_outer, search: bool):
    oof = np.zeros(len(y))
    chosen = []
    for tr, te in cv_outer.split(X, y, g):
        if search:
            gs = GridSearchCV(clone(pipe), GRID, cv=GroupKFold(n_splits=INNER_FOLDS), scoring="r2", n_jobs=-1)
            gs.fit(X.iloc[tr], y.iloc[tr], groups=g.iloc[tr])
            est = gs.best_estimator_
            chosen.append({k: float(v) for k, v in gs.best_params_.items()})
        else:
            est = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        oof[te] = est.predict(X.iloc[te])
    return oof, chosen


def fold_r2(y, oof, g, cv_outer, X):
    return [float(r2_score(y.iloc[te], oof[te])) for _, te in cv_outer.split(X, y, g)]


def participant_bootstrap_delta(y, a, b, g):
    """95% CI of R2(a) - R2(b) resampling participants (a, b are OOF predictions)."""
    rng = np.random.RandomState(SEED)
    idx = {pid: np.flatnonzero(g.values == pid) for pid in g.unique()}
    ids = np.array(list(idx))
    deltas = []
    for _ in range(N_BOOT):
        rows = np.concatenate([idx[p] for p in rng.choice(ids, len(ids), replace=True)])
        deltas.append(r2_score(y.values[rows], a[rows]) - r2_score(y.values[rows], b[rows]))
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def participant_offset_ceiling(y, oof, g):
    """Optimistic ceiling for ANY participant-level correction: subtract each participant's mean
    residual (computed in-sample, so it is looser than anything achievable)."""
    r = y - oof
    r_adj = r - r.groupby(g).transform("mean")
    sst = float(((y - y.mean()) ** 2).sum())
    return {"R2_current": float(1 - (r ** 2).sum() / sst),
            "R2_if_participant_offsets_were_known_in_sample": float(1 - (r_adj ** 2).sum() / sst)}


def main():
    t0 = time.time()
    dev = make_splits()[1]["dev"]  # DEV ONLY - holdouts are never used here
    X, y, g = dev[RAW_FEATURES], dev[TARGET], dev[GROUP]
    cv = GroupKFold(n_splits=OUTER_FOLDS)  # same folds as ml/pp1_optimize.py
    frozen = joblib.load(PP1_DIR / "best_rpe_model_pp1.joblib")

    specs = [
        ("mean_baseline", DummyRegressor(strategy="mean"), False, "Mean predictor"),
        ("elasticnet_fe_frozen", frozen, False,
         "Existing frozen ElasticNet+engineered features (its hyper-parameters were tuned on these same dev folds)"),
        ("elasticnet_fe_nested", make_pipe(ElasticNet(max_iter=20000), True), True,
         "Existing recipe re-tuned with the SAME nested protocol as the new candidates (fair reference)"),
        ("spline_elasticnet_nested", spline_pipe(), True,
         "NEW: cubic spline basis (4 knots) on the 8 numeric inputs + one-hot workout type + ElasticNet"),
        ("pairwise_poly_elasticnet_nested", interaction_pipe(), True,
         "NEW: all squares + pairwise interactions of the 8 numeric inputs + one-hot workout type + ElasticNet"),
    ]
    results, oofs = {}, {}
    for name, est, search, desc in specs:
        ts = time.time()
        oof, chosen = nested_oof(est, X, y, g, cv, search)
        folds = fold_r2(y, oof, g, cv, X)
        s = regression_summary(y, oof)
        results[name] = {"description": desc, "nested_hyperparameter_search": search,
                         "pooled_oof": s, "fold_R2": folds,
                         "fold_R2_mean": float(np.mean(folds)), "fold_R2_std": float(np.std(folds)),
                         "chosen_params_per_fold": chosen, "seconds": round(time.time() - ts, 1)}
        oofs[name] = oof
        print(f"{name:34s} fold-R2 {np.mean(folds):.4f} ± {np.std(folds):.4f} | pooled R2 {s['R2']:.4f} "
              f"MAE {s['MAE']:.3f} RMSE {s['RMSE']:.3f} within±1 {s['within_pm1']['pct']:.2f}%", flush=True)

    ref = "elasticnet_fe_nested"
    comparisons = {}
    for name in ("spline_elasticnet_nested", "pairwise_poly_elasticnet_nested"):
        d_folds = [a - b for a, b in zip(results[name]["fold_R2"], results[ref]["fold_R2"])]
        comparisons[f"{name}_minus_{ref}"] = {
            "mean_fold_R2_delta": float(np.mean(d_folds)), "fold_deltas": d_folds,
            "folds_better": int(sum(d > 0 for d in d_folds)),
            "pooled_R2_delta": results[name]["pooled_oof"]["R2"] - results[ref]["pooled_oof"]["R2"],
            "participant_bootstrap_95CI_of_pooled_R2_delta": participant_bootstrap_delta(y, oofs[name], oofs[ref], g),
        }

    ceiling = participant_offset_ceiling(y, oofs["elasticnet_fe_frozen"], g)
    best_new = max(("spline_elasticnet_nested", "pairwise_poly_elasticnet_nested"),
                   key=lambda n: results[n]["fold_R2_mean"])
    out = {
        "IMPORTANT_NOTE": ("DEVELOPMENT-CV evidence on SYNTHETIC data only. The sealed and legacy holdouts were NOT "
                           "used. No independent evaluation data exists for any newly chosen candidate, so no claim "
                           "about the R2 >= 0.60 target is made."),
        "protocol": {"data": "dev participants only", "outer_cv": f"GroupKFold({OUTER_FOLDS}) (same folds as PP1)",
                     "inner_cv_for_new_candidates": f"GroupKFold({INNER_FOLDS}) on outer-training participants",
                     "grid": GRID, "inputs": RAW_FEATURES,
                     "n_rows": int(len(dev)), "n_participants": int(g.nunique())},
        "candidates": results,
        "comparisons_vs_nested_reference": comparisons,
        "participant_offset_ceiling_dev": ceiling,
        "best_new_candidate_by_dev_cv": best_new,
        "best_new_candidate_fold_R2_mean": results[best_new]["fold_R2_mean"],
        "reaches_0.60_in_dev_cv": bool(results[best_new]["fold_R2_mean"] >= 0.60),
        "independent_evaluation_available": False,
        "target_0.60_independently_achieved": False,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (PP1_DIR / "dev_cv_new_experiment.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\ncomparisons vs nested reference:")
    print(json.dumps(comparisons, indent=2))
    print("\nparticipant-offset ceiling (dev, in-sample, optimistic):", ceiling)
    print("reaches 0.60 in dev CV:", out["reaches_0.60_in_dev_cv"], "| runtime", out["runtime_seconds"], "s")


if __name__ == "__main__":
    main()
