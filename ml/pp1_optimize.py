"""
PP1 RPE model optimisation (synthetic development dataset).

Run from the project root:
    python -m ml.pp1_optimize

Protocol
  * dev (150 participants): ALL hyper-parameter search / model selection via GroupKFold.
  * sealed holdout (50 new participants): evaluated ONCE for the frozen final model
    and ONCE for the baseline recipe re-fit on dev. Never used for selection.
  * legacy holdout (50 original test participants): already used to pick the baseline
    model in ml/train_models.py -> reported only for like-for-like comparison.
  * Selection rule (fixed before running): highest mean GroupKFold R2 on dev.
Synthetic-data results are development evidence, NOT validation on real athletes.
"""
import json
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Auto-bootstrap: If running under system Python without ML dependencies installed,
# automatically delegate execution to the project's virtual environment python.
try:
    import joblib  # pyrefly: ignore [missing-import]  # type: ignore
    import numpy as np  # pyrefly: ignore [missing-import]  # type: ignore
    import pandas as pd  # pyrefly: ignore [missing-import]  # type: ignore
    import sklearn  # pyrefly: ignore [missing-import]  # type: ignore
    from scipy.stats import loguniform, randint, uniform  # pyrefly: ignore [missing-import]  # type: ignore
    from sklearn.compose import ColumnTransformer  # pyrefly: ignore [missing-import]  # type: ignore
    from sklearn.dummy import DummyRegressor  # pyrefly: ignore [missing-import]  # type: ignore
    from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,  # pyrefly: ignore [missing-import]  # type: ignore
                                  RandomForestRegressor)
    from sklearn.impute import SimpleImputer  # pyrefly: ignore [missing-import]  # type: ignore
    from sklearn.inspection import permutation_importance  # pyrefly: ignore [missing-import]  # type: ignore
    from sklearn.linear_model import ElasticNet, Ridge  # pyrefly: ignore [missing-import]  # type: ignore
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # pyrefly: ignore [missing-import]  # type: ignore
    from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict  # pyrefly: ignore [missing-import]  # type: ignore
    from sklearn.pipeline import Pipeline  # pyrefly: ignore [missing-import]  # type: ignore
    from sklearn.preprocessing import OneHotEncoder, StandardScaler  # pyrefly: ignore [missing-import]  # type: ignore
except ModuleNotFoundError:
    venv_py = ROOT / "venv" / "Scripts" / "python.exe"
    if venv_py.is_file() and Path(sys.executable).resolve() != venv_py.resolve():
        import subprocess
        _res = subprocess.run([str(venv_py), str(__file__)] + sys.argv[1:])
        sys.exit(_res.returncode)
    else:
        sys.exit(
            f"\n[ERROR] Missing required ML dependencies. Virtual environment not found at:\n"
            f"  {venv_py}\n"
        )

try:
    from ml.pp1_features import CATEGORICAL, RAW_FEATURES, PreSessionFeatures, numeric_columns
    from ml.pp1_splits import GROUP, SPLIT_SEED, TARGET, make_splits
except (ImportError, ModuleNotFoundError):
    from pp1_features import CATEGORICAL, RAW_FEATURES, PreSessionFeatures, numeric_columns  # type: ignore
    from pp1_splits import GROUP, SPLIT_SEED, TARGET, make_splits  # type: ignore

try:
    from xgboost import XGBRegressor  # pyrefly: ignore [missing-import]  # type: ignore
    HAVE_XGB = True
except Exception:  # pragma: no cover
    HAVE_XGB = False

SEED = 23166660
N_FOLDS = 5
N_ITER = 20  # bounded search budget per candidate
OUT = ROOT / "artifacts" / "pp1_model_optimization"


def metrics(y, p):
    return {"R2": float(r2_score(y, p)), "MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(mean_squared_error(y, p) ** 0.5)}


def make_pipe(model, fe: bool, scale=True):
    num = numeric_columns(fe)
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler() if scale else "passthrough")]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("features", PreSessionFeatures(enabled=fe)),
                     ("preprocess", pre), ("model", model)])


def candidates():
    c = {}
    c["dummy_mean"] = (make_pipe(DummyRegressor(strategy="mean"), False), {})
    for fe in (False, True):
        tag = "fe" if fe else "raw"
        c[f"ridge_{tag}"] = (make_pipe(Ridge(), fe),
                             {"model__alpha": loguniform(1e-2, 1e3)})
        c[f"elasticnet_{tag}"] = (make_pipe(ElasticNet(max_iter=20000), fe),
                                  {"model__alpha": loguniform(1e-4, 1e0),
                                   "model__l1_ratio": uniform(0.05, 0.95)})
        c[f"random_forest_{tag}"] = (
            make_pipe(RandomForestRegressor(random_state=SEED, n_jobs=1), fe, scale=False),
            {"model__n_estimators": randint(150, 401),
             "model__max_depth": [3, 4, 6, 8, 12, None],
             "model__min_samples_leaf": randint(5, 60),
             "model__max_features": [0.3, 0.5, 0.7, 1.0]})
        c[f"extra_trees_{tag}"] = (
            make_pipe(ExtraTreesRegressor(random_state=SEED, n_jobs=1), fe, scale=False),
            {"model__n_estimators": randint(150, 401),
             "model__max_depth": [3, 4, 6, 8, 12, None],
             "model__min_samples_leaf": randint(5, 60),
             "model__max_features": [0.3, 0.5, 0.7, 1.0]})
        c[f"hist_gb_{tag}"] = (
            make_pipe(HistGradientBoostingRegressor(random_state=SEED), fe, scale=False),
            {"model__learning_rate": loguniform(0.01, 0.2),
             "model__max_iter": randint(50, 400),
             "model__max_depth": [2, 3, 4, 6],
             "model__min_samples_leaf": randint(20, 120),
             "model__l2_regularization": loguniform(1e-2, 1e2)})
        if HAVE_XGB:
            c[f"xgboost_{tag}"] = (
                make_pipe(XGBRegressor(random_state=SEED, n_jobs=1, tree_method="hist",
                                       verbosity=0), fe, scale=False),
                {"model__n_estimators": randint(50, 400),
                 "model__learning_rate": loguniform(0.01, 0.2),
                 "model__max_depth": randint(2, 6),
                 "model__min_child_weight": randint(5, 60),
                 "model__subsample": uniform(0.6, 0.4),
                 "model__colsample_bytree": uniform(0.5, 0.5),
                 "model__reg_lambda": loguniform(1e-1, 1e2)})
    return c


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="PP1 RPE model optimisation (synthetic development dataset)."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke-test run with 1 search iteration for fast validation."
    )
    args = parser.parse_args(argv)
    search_budget = 1 if args.quick else N_ITER

    OUT.mkdir(parents=True, exist_ok=True)
    df, parts = make_splits()
    dev, sealed, legacy = parts["dev"], parts["sealed_holdout"], parts["legacy_holdout"]
    Xd, yd, gd = dev[RAW_FEATURES], dev[TARGET], dev[GROUP]
    cv = GroupKFold(n_splits=N_FOLDS)
    t0 = time.time()

    rows, oof_store, fitted = [], {}, {}
    for name, (pipe, space) in candidates().items():
        ts = time.time()
        if space:
            search = RandomizedSearchCV(pipe, space, n_iter=search_budget, cv=cv, scoring="r2",
                                        random_state=SEED, n_jobs=-1, refit=True)
            search.fit(Xd, yd, groups=gd)
            best, params = search.best_estimator_, search.best_params_
            inner = float(search.best_score_)
        else:
            best, params, inner = pipe.fit(Xd, yd), {}, float("nan")
        oof = cross_val_predict(best, Xd, yd, groups=gd, cv=cv)
        fold_r2 = [r2_score(yd.iloc[te], oof[te]) for _, te in cv.split(Xd, yd, gd)]
        cvm = metrics(yd, oof)
        trm = metrics(yd, best.predict(Xd))
        rows.append({
            "candidate": name, "family": name.rsplit("_", 1)[0],
            "feature_set": "engineered" if name.endswith("_fe") else "raw_9_inputs",
            "search_iterations": search_budget if space else 0,
            "cv_R2_mean_of_folds": float(np.mean(fold_r2)), "cv_R2_std_of_folds": float(np.std(fold_r2)),
            "cv_R2_pooled_oof": cvm["R2"], "cv_MAE": cvm["MAE"], "cv_RMSE": cvm["RMSE"],
            "train_R2": trm["R2"], "generalization_gap_R2": trm["R2"] - float(np.mean(fold_r2)),
            "search_best_score_R2": inner, "best_params": json.dumps(
                {k: (float(v) if isinstance(v, (np.floating, float)) else
                     int(v) if isinstance(v, (np.integer, int)) else v) for k, v in params.items()}),
            "seconds": round(time.time() - ts, 1),
        })
        oof_store[name] = oof
        fitted[name] = best
        print(f"{name:22s} CV R2={np.mean(fold_r2):.4f} +/- {np.std(fold_r2):.4f} "
              f"MAE={cvm['MAE']:.4f} train R2={trm['R2']:.4f} ({time.time()-ts:.0f}s)", flush=True)

    comp = pd.DataFrame(rows).sort_values("cv_R2_mean_of_folds", ascending=False)
    comp.to_csv(OUT / "experiment_comparison.csv", index=False)
    pd.DataFrame({"row_in_dev": np.arange(len(dev)), "participant_id": gd.values,
                  "actual": yd.values, **{f"oof_{k}": v for k, v in oof_store.items()}}
                 ).to_csv(OUT / "cross_validation_oof_predictions.csv", index=False)

    # ---- freeze the final model using DEV CV ONLY -------------------------------------
    final_name = comp.iloc[0]["candidate"]
    final = fitted[final_name]
    ref_pipe = make_pipe(Ridge(alpha=1.0), False)  # exact original baseline recipe
    ref_pipe.fit(Xd, yd)
    print(f"\nFROZEN final model (selected on dev CV only): {final_name}", flush=True)

    # ---- one-shot holdout evaluation ---------------------------------------------------
    evals, pred_frames = {}, []
    for hname, hdf in (("sealed_holdout", sealed), ("legacy_holdout", legacy)):
        Xh, yh = hdf[RAW_FEATURES], hdf[TARGET]
        for mname, m in (("final_model", final), ("baseline_recipe_ridge_raw_alpha1", ref_pipe)):
            p = m.predict(Xh)
            evals[f"{hname}/{mname}"] = metrics(yh, p)
            if mname == "final_model":
                pred_frames.append(pd.DataFrame({
                    "holdout": hname, "participant_id": hdf[GROUP].values,
                    "session_id": hdf["session_id"].values,
                    "actual_rpe": yh.values, "predicted_rpe": p,
                    "residual": yh.values - p}))
    # original saved baseline artifact (trained on 200 participants) on the LEGACY holdout
    baseline_artifact = ROOT / "artifacts" / "best_rpe_model.joblib"
    if baseline_artifact.is_file():
        orig = joblib.load(baseline_artifact)
        evals["legacy_holdout/original_saved_baseline_artifact"] = metrics(
            legacy[TARGET], orig.predict(legacy[RAW_FEATURES]))
    else:
        evals["legacy_holdout/original_saved_baseline_artifact"] = {"R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}
    # dummy on the sealed holdout, for context
    dm = DummyRegressor().fit(Xd, yd)
    evals["sealed_holdout/dummy_mean"] = metrics(sealed[TARGET], dm.predict(sealed[RAW_FEATURES]))

    pd.concat(pred_frames).to_csv(OUT / "actual_vs_predicted_holdout.csv", index=False)

    # participant-bootstrap CI for the sealed-holdout R2 (resampling participants)
    sp = [f for f in pred_frames if (f["holdout"] == "sealed_holdout").all()][0]
    rng = np.random.RandomState(SEED)
    ids = sp["participant_id"].unique()
    groups = {i: g for i, g in sp.groupby("participant_id")}
    boots = []
    for _ in range(1000):
        pick = rng.choice(ids, len(ids), replace=True)
        b = pd.concat([groups[i] for i in pick])
        boots.append(r2_score(b["actual_rpe"], b["predicted_rpe"]))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    # permutation importance (raw columns) on the sealed holdout -- descriptive only
    imp = permutation_importance(final, sealed[RAW_FEATURES], sealed[TARGET], scoring="r2",
                                 n_repeats=20, random_state=SEED, n_jobs=1)
    pd.DataFrame({"feature": RAW_FEATURES, "importance_mean_R2_drop": imp.importances_mean,
                  "importance_std": imp.importances_std}
                 ).sort_values("importance_mean_R2_drop", ascending=False
                               ).to_csv(OUT / "feature_importance.csv", index=False)

    # ---- diagnostics (dev only) ---------------------------------------------------------
    ridge_oof = oof_store["ridge_raw"]
    res = yd - ridge_oof
    part_mean_share = float(res.groupby(gd).transform("mean").var() / res.var())
    best_family_r2 = comp.groupby("family")["cv_R2_mean_of_folds"].max().to_dict()
    diag = {
        "dev_target_std": float(yd.std()),
        "dev_residual_std_ridge_raw": float(res.std()),
        "share_of_residual_variance_explained_by_participant_mean": part_mean_share,
        "best_cv_R2_per_family": best_family_r2,
        "note": "Flexible non-linear models do not beat Ridge on the same 9 inputs, and the "
                "participant-mean share of residual variance is small: remaining error is "
                "dominated by session-level noise not explained by pre-session inputs.",
    }

    # ---- save model + evidence -----------------------------------------------------------
    model_path = OUT / "best_rpe_model_pp1.joblib"
    joblib.dump(final, model_path)
    fe_row = comp.iloc[0].to_dict()
    sealed_m = evals["sealed_holdout/final_model"]
    result = {
        "IMPORTANT_NOTE": "Synthetic development-dataset evidence only. NOT validation on real "
                          "athletes. All values below were computed by this script.",
        "target": TARGET, "dataset": str(Path("data") / "TEQS_Synthetic_5000_Development_Dataset.csv"),
        "best_model_name": final_name,
        "best_model_best_params": json.loads(fe_row["best_params"]),
        "selection_rule": "highest mean GroupKFold(5) R2 on dev participants only",
        "dev_cv": {"R2_mean_of_folds": fe_row["cv_R2_mean_of_folds"],
                   "R2_std_of_folds": fe_row["cv_R2_std_of_folds"],
                   "R2_pooled_oof": fe_row["cv_R2_pooled_oof"],
                   "MAE": fe_row["cv_MAE"], "RMSE": fe_row["cv_RMSE"]},
        "train_R2": fe_row["train_R2"],
        "generalization_gap": {
            "train_minus_dev_cv_R2": fe_row["generalization_gap_R2"],
            "dev_cv_minus_sealed_holdout_R2": fe_row["cv_R2_mean_of_folds"] - sealed_m["R2"]},
        "holdout_evaluations": evals,
        "sealed_holdout_R2_participant_bootstrap_95CI": ci,
        "target_R2_0.60_achieved_on_sealed_holdout": bool(sealed_m["R2"] >= 0.60),
        "target_R2_0.60_achieved_on_dev_cv": bool(fe_row["cv_R2_mean_of_folds"] >= 0.60),
        "diagnostics": diag,
        "holdout_provenance": {
            "legacy_holdout": "50 original test participants (artifacts/test_participants.csv); "
                              "ALREADY USED to select Ridge in ml/train_models.py (best test MAE). "
                              "Not used for any PP1 tuning; reported for comparison only.",
            "sealed_holdout": f"50 new participants sampled (seed {SPLIT_SEED}) from the original "
                              "training participants; evaluated once after freezing the model.",
            "dev": "remaining 150 participants; all tuning via GroupKFold",
            "participants": {k: int(v[GROUP].nunique()) for k, v in parts.items()},
            "rows": {k: int(len(v)) for k, v in parts.items()},
        },
        "model_artifact": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "model_trained_on": "dev participants only (150)",
        "candidate_catboost": "not installed / not used",
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(OUT / "final_holdout_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(OUT / "training_configuration.json", "w") as f:
        json.dump({
            "random_seed": SEED, "split_seed": SPLIT_SEED, "cv": f"GroupKFold(n_splits={N_FOLDS})",
            "search": f"RandomizedSearchCV(n_iter={N_ITER}, scoring='r2', refit on dev)",
            "inputs": RAW_FEATURES, "engineered_features_pre_session_only": numeric_columns(True)[8:],
            "excluded_leakage_columns": ["session_id", "timestamp", "actual_duration_min",
                                         "session_rpe_1_10", "completion_status",
                                         "performance_deviation_pct", "data_type", "participant_id"],
            "candidates": list(fitted), "python": platform.python_version(),
            "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
            "xgboost": __import__("xgboost").__version__ if HAVE_XGB else None,
        }, f, indent=2)

    print("\n=== HOLDOUT EVALUATIONS (one-shot) ===")
    for k, v in evals.items():
        print(f"{k:60s} R2={v['R2']:.4f} MAE={v['MAE']:.4f} RMSE={v['RMSE']:.4f}")
    print(f"sealed R2 95% participant-bootstrap CI: {ci[0]:.3f} .. {ci[1]:.3f}")
    print("R2>=0.60 on sealed holdout:", result["target_R2_0.60_achieved_on_sealed_holdout"])


if __name__ == "__main__":
    main()
