"""
PP1 REAL PMData experiment (kept completely separate from the synthetic experiment).

Run from the project root:
    python -m ml.pp1_pmdata_experiment

 * Data: audited prospective PMData sessions (no environment columns are added).
 * Evaluation: NESTED participant-grouped CV. Outer: GroupKFold(5, shuffled) repeated with
   5 seeds + leave-one-participant-out (LOPO). Inner: GroupKFold(3) hyper-parameter search
   on the outer-training participants only.
 * Every candidate is compared with DummyRegressor(mean). Results are reported even if
   they remain weak or negative. Nothing here is synthetic.
"""
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml.paths import AUDITED_CSV, require_file  # noqa: E402  (external data dir via PMDATA_EXPERIMENTS_DIR)
OUT = ROOT / "artifacts" / "pp1_model_optimization" / "real_pmdata"
SEED = 23166660
WELLNESS = ["fatigue", "mood", "readiness", "sleep_duration_h", "sleep_quality", "soreness", "stress"]
TARGET, GROUP = "session_rpe", "participant_id"


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Defensible pre-session composites from the same wellness questionnaire (no new data)."""
    X = X.copy()
    X["wellness_composite"] = (X["readiness"] + X["mood"] + X["sleep_quality"]
                               - X["fatigue"] - X["soreness"] - X["stress"]) / 6.0
    X["fatigue_soreness"] = (X["fatigue"] + X["soreness"]) / 2.0
    X["sleep_deficit"] = np.clip(7.5 - X["sleep_duration_h"], 0, None)
    X["readiness_minus_fatigue"] = X["readiness"] - X["fatigue"]
    return X


COMPOSITES = ["wellness_composite", "fatigue_soreness", "sleep_deficit", "readiness_minus_fatigue"]


def pipe(model, features, with_activity=False, engineered=False, scale=True):
    ft = FunctionTransformer(add_features if engineered else (lambda x: x), validate=False)
    num = features + (COMPOSITES if engineered else [])
    tfs = [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler() if scale else "passthrough")]), num)]
    if with_activity:
        tfs.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                     ("oh", OneHotEncoder(handle_unknown="ignore"))]),
                    ["primary_activity"]))
    return Pipeline([("ft", ft), ("pre", ColumnTransformer(tfs)), ("model", model)])


def tuned(estimator, grid):
    return GridSearchCV(estimator, grid, cv=GroupKFold(n_splits=3), scoring="neg_mean_squared_error",
                        n_jobs=1, refit=True)


def candidates():
    c = {"dummy_mean": (pipe(DummyRegressor(), WELLNESS), None)}
    for eng in (False, True):
        for act in (False, True):
            tag = f"{'eng' if eng else 'raw'}_{'act' if act else 'wellness'}"
            c[f"ridge_{tag}"] = (pipe(Ridge(), WELLNESS, act, eng),
                                 {"model__alpha": [1, 10, 30, 100, 300, 1000]})
            c[f"elasticnet_{tag}"] = (pipe(ElasticNet(max_iter=20000), WELLNESS, act, eng),
                                      {"model__alpha": [0.01, 0.05, 0.2, 0.5, 1.0],
                                       "model__l1_ratio": [0.2, 0.6, 0.9]})
    for act in (False, True):
        tag = f"raw_{'act' if act else 'wellness'}"
        c[f"random_forest_shallow_{tag}"] = (
            pipe(RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
                 WELLNESS, act, False, scale=False),
            {"model__max_depth": [2, 3, 4], "model__min_samples_leaf": [15, 30, 60]})
        c[f"hist_gb_shallow_{tag}"] = (
            pipe(HistGradientBoostingRegressor(random_state=SEED, max_iter=60, learning_rate=0.03),
                 WELLNESS, act, False, scale=False),
            {"model__max_depth": [2, 3], "model__min_samples_leaf": [30, 60],
             "model__l2_regularization": [1.0, 20.0]})
    return c


def metr(y, p):
    return {"R2": r2_score(y, p), "MAE": mean_absolute_error(y, p),
            "RMSE": mean_squared_error(y, p) ** 0.5}


def run_outer(name, est, grid, df, splits):
    oof = np.full(len(df), np.nan)
    fold_r2 = []
    for tr, va in splits:
        m = tuned(clone(est), grid) if grid else clone(est)
        m.fit(df.iloc[tr], df[TARGET].iloc[tr], **({"groups": df[GROUP].iloc[tr]} if grid else {}))
        p = m.predict(df.iloc[va])
        oof[va] = p
        if len(va) > 1:
            fold_r2.append(r2_score(df[TARGET].iloc[va], p))
    return oof, fold_r2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    require_file(AUDITED_CSV, "Audited PMData sessions CSV", "PMDATA_EXPERIMENTS_DIR")
    df = pd.read_csv(AUDITED_CSV).reset_index(drop=True)
    assert not ({"session_duration_min", "duration_min", "actual_duration_min"} & set(WELLNESS))
    X = df[[GROUP, "primary_activity"] + WELLNESS]
    y = df[TARGET]
    print(f"PMData audited: {len(df)} sessions, {df[GROUP].nunique()} participants", flush=True)
    t0 = time.time()

    rep_splits = []
    for r in range(5):
        gk = GroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)
        s = list(gk.split(df, y, df[GROUP]))
        for tr, va in s:
            assert set(df[GROUP].iloc[tr]).isdisjoint(set(df[GROUP].iloc[va]))
        rep_splits.append(s)
    lopo = list(LeaveOneGroupOut().split(df, y, df[GROUP]))

    rows, store = [], {}
    for name, (est, grid) in candidates().items():
        ts = time.time()
        rep_pooled, rep_mae, rep_rmse, rep_fold = [], [], [], []
        last_oof = None
        for s in rep_splits:
            oof, fr2 = run_outer(name, est, grid, df, s)
            m = metr(y, oof)
            rep_pooled.append(m["R2"]); rep_mae.append(m["MAE"]); rep_rmse.append(m["RMSE"])
            rep_fold.extend(fr2); last_oof = oof
        lo_oof, _ = run_outer(name, est, grid, df, lopo)
        lm = metr(y, lo_oof)
        rows.append({
            "candidate": name,
            "cv_scheme": "nested GroupKFold(5, shuffled) x5 repeats",
            "pooled_oof_R2_mean": np.mean(rep_pooled), "pooled_oof_R2_std_over_repeats": np.std(rep_pooled),
            "mean_fold_R2": np.mean(rep_fold), "pooled_oof_MAE_mean": np.mean(rep_mae),
            "pooled_oof_RMSE_mean": np.mean(rep_rmse),
            "lopo_pooled_R2": lm["R2"], "lopo_MAE": lm["MAE"], "lopo_RMSE": lm["RMSE"],
            "seconds": round(time.time() - ts, 1)})
        store[name] = lo_oof
        print(f"{name:42s} pooled R2={np.mean(rep_pooled):+.4f} MAE={np.mean(rep_mae):.3f} "
              f"| LOPO R2={lm['R2']:+.4f}", flush=True)

    res = pd.DataFrame(rows)
    dummy = res.loc[res.candidate == "dummy_mean"].iloc[0]
    res["R2_gain_vs_dummy"] = res["pooled_oof_R2_mean"] - dummy["pooled_oof_R2_mean"]
    res["MAE_gain_vs_dummy"] = dummy["pooled_oof_MAE_mean"] - res["pooled_oof_MAE_mean"]
    res = res.sort_values("pooled_oof_R2_mean", ascending=False)
    res.to_csv(OUT / "pmdata_experiment_comparison.csv", index=False)

    best = res.iloc[0]["candidate"]
    pd.DataFrame({GROUP: df[GROUP], "actual_rpe": y, **{f"lopo_oof_{k}": v for k, v in store.items()}}
                 ).to_csv(OUT / "pmdata_lopo_oof_predictions.csv", index=False)
    summary = {
        "IMPORTANT_NOTE": "REAL PMData (not synthetic). Separate from the synthetic experiment. "
                          "No environment measurements were added. Results reported as computed.",
        "sessions": int(len(df)), "participants": int(df[GROUP].nunique()),
        "target_std": float(y.std()),
        "best_candidate_by_pooled_oof_R2": best,
        "best_pooled_oof_R2": float(res.iloc[0]["pooled_oof_R2_mean"]),
        "best_MAE": float(res.iloc[0]["pooled_oof_MAE_mean"]),
        "best_RMSE": float(res.iloc[0]["pooled_oof_RMSE_mean"]),
        "dummy_pooled_oof_R2": float(dummy["pooled_oof_R2_mean"]),
        "dummy_MAE": float(dummy["pooled_oof_MAE_mean"]),
        "any_candidate_R2_above_0.10": bool((res["pooled_oof_R2_mean"] > 0.10).any()),
        "note_selection": "'best' is the maximum over ~15 candidates on the same 15 participants; "
                          "it is optimistically biased and should not be read as real performance.",
        "seconds": round(time.time() - t0, 1),
    }
    with open(OUT / "pmdata_experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
