"""PP1 report artifacts derived from ALREADY SAVED experiment outputs.

Run from the project root:
    python -m ml.pp1_report_artifacts

Writes (new files only; nothing existing is overwritten):
  artifacts/pp1_model_optimization/interpretable_metrics.json
  artifacts/pp1_model_optimization/learning_curve.csv

Rules followed:
  * No model selection / tuning. The frozen model is only evaluated.
  * Predictions are read from the saved out-of-sample prediction files; numbers are
    cross-checked against final_holdout_metrics.json.
  * The learning curve fits CLONES of the frozen pipeline (same hyper-parameters) on
    participant-grouped DEV data only. The saved model artifact is not modified and the
    sealed/legacy holdouts are not touched.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, learning_curve

from ml.paths import ARTIFACTS_DIR, PP1_DIR, SYNTHETIC_CSV
from ml.pp1_features import RAW_FEATURES
from ml.pp1_metrics import WITHIN_LABEL, TOLERANCE, regression_summary, within_tolerance
from ml.pp1_splits import GROUP, TARGET, make_splits

SEED = 23166660
N_FOLDS = 5


def _check(a: float, b: float, what: str) -> None:
    if abs(a - b) > 1e-9:
        raise AssertionError(f"{what}: recomputed {a} != saved {b}")


def synthetic_blocks(res: dict) -> dict:
    final = res["best_model_name"]
    oof = pd.read_csv(PP1_DIR / "cross_validation_oof_predictions.csv")
    avp = pd.read_csv(PP1_DIR / "actual_vs_predicted_holdout.csv")
    dev_mean = float(oof["actual"].mean())  # constant mean predictor fitted on dev targets only

    blocks = {}
    dev = {
        "dataset": "SYNTHETIC development dataset (not real athletes)",
        "split": f"GroupKFold({N_FOLDS}) out-of-fold predictions on the dev participants",
        "participants": int(oof["participant_id"].nunique()),
        "final_model": regression_summary(oof["actual"], oof[f"oof_{final}"]),
        "ridge_raw_baseline": regression_summary(oof["actual"], oof["oof_ridge_raw"]),
        "mean_predictor": regression_summary(oof["actual"], oof["oof_dummy_mean"]),
    }
    blocks["synthetic_dev_groupkfold_oof"] = dev

    for hname, key in (("sealed_holdout", "synthetic_sealed_holdout"), ("legacy_holdout", "synthetic_legacy_holdout")):
        h = avp[avp["holdout"] == hname]
        summ = regression_summary(h["actual_rpe"], h["predicted_rpe"])
        saved = res["holdout_evaluations"][f"{hname}/final_model"]
        _check(summ["R2"], saved["R2"], f"{hname} R2")
        _check(summ["MAE"], saved["MAE"], f"{hname} MAE")
        _check(summ["RMSE"], saved["RMSE"], f"{hname} RMSE")
        blocks[key] = {
            "dataset": "SYNTHETIC development dataset (not real athletes)",
            "split": ("50 participants never used for tuning; evaluated once to report the frozen model"
                      if hname == "sealed_holdout" else
                      "50 original test participants; already used to pick the baseline in train_models.py (comparison only)"),
            "participants": int(h["participant_id"].nunique()),
            "final_model": summ,
            "mean_predictor_fit_on_dev": regression_summary(h["actual_rpe"], np.full(len(h), dev_mean)),
        }
    blocks["synthetic_sealed_holdout"]["note"] = (
        "Already consulted to report the optimised model; do not use for any further model selection.")
    return blocks


def real_pmdata_block() -> dict:
    real = PP1_DIR / "real_pmdata"
    lopo_path, summ_path = real / "pmdata_lopo_oof_predictions.csv", real / "pmdata_experiment_summary.json"
    if not (lopo_path.is_file() and summ_path.is_file()):
        return {"available": False,
                "reason": "run `python -m ml.pp1_pmdata_experiment` (needs external PMData, see README)"}
    lopo = pd.read_csv(lopo_path)
    summary = json.loads(summ_path.read_text(encoding="utf-8"))
    best = summary["best_candidate_by_pooled_oof_R2"]
    cols = [c for c in lopo.columns if c.startswith("lopo_oof_")]
    table = {c.replace("lopo_oof_", ""): regression_summary(lopo["actual_rpe"], lopo[c]) for c in cols}
    return {
        "available": True,
        "dataset": "REAL PMData (audited prospective sessions; no environment data) - separate from synthetic",
        "split": "leave-one-participant-out out-of-fold predictions",
        "participants": int(lopo["participant_id"].nunique()),
        "n": int(len(lopo)),
        "mean_predictor": table["dummy_mean"],
        "best_candidate_name": best,
        "best_candidate": table[best],
        "all_candidates": table,
        "note": ("The best candidate is the maximum over many candidates on the same 15 participants and is "
                 "optimistically biased. Real-data skill is essentially nil."),
    }


def completion_block() -> dict:
    """Reference only (the completion classifier is not shown in the dashboard)."""
    summ_path = ARTIFACTS_DIR / "model_summary.json"
    test_ids = set(pd.read_csv(ARTIFACTS_DIR / "test_participants.csv")[GROUP])
    df = pd.read_csv(SYNTHETIC_CSV)
    y = df[df[GROUP].isin(test_ids)]["completion_status"]
    majority = y.value_counts().idxmax()
    const = np.full(len(y), majority)
    stored = json.loads(summ_path.read_text(encoding="utf-8"))["classification"]
    return {
        "dataset": "SYNTHETIC, legacy test participants",
        "n": int(len(y)),
        "class_share": y.value_counts(normalize=True).round(4).to_dict(),
        "majority_class_baseline": {"class": majority,
                                    "accuracy": float(accuracy_score(y, const)),
                                    "macro_f1": float(f1_score(y, const, average="macro", zero_division=0))},
        "stored_model_metrics": stored["metrics"],
        "stored_best_model": stored["best_model"],
        "note": "Accuracy must be read against the majority-class baseline; macro F1 is the primary metric.",
    }


def learning_curve_csv(res: dict) -> pd.DataFrame:
    model = joblib.load(PP1_DIR / "best_rpe_model_pp1.joblib")
    dev = make_splits()[1]["dev"]  # DEV ONLY
    X, y, g = dev[RAW_FEATURES], dev[TARGET], dev[GROUP]
    sizes, tr, va = learning_curve(
        clone(model), X, y, groups=g, cv=GroupKFold(n_splits=N_FOLDS),
        train_sizes=np.linspace(0.1, 1.0, 8), scoring="r2", shuffle=True, random_state=SEED, n_jobs=1)
    return pd.DataFrame({
        "train_rows_per_fold": sizes,
        "train_R2_mean": tr.mean(axis=1), "train_R2_std": tr.std(axis=1),
        "validation_R2_mean": va.mean(axis=1), "validation_R2_std": va.std(axis=1),
        "estimator": res["best_model_name"],
        "protocol": f"GroupKFold({N_FOLDS}) on dev participants; fixed hyper-parameters of the frozen model",
    })


def main():
    res = json.loads((PP1_DIR / "final_holdout_metrics.json").read_text(encoding="utf-8"))
    out = {
        "metric_label": WITHIN_LABEL,
        "metric_definition": (f"100 * (# predictions with |actual - predicted| <= {TOLERANCE}) / (# predictions "
                              "in the split). Not a classification accuracy; always compare with the "
                              "mean-predictor baseline."),
        "primary_research_metric": "regression R2 on session RPE (MAE / RMSE reported alongside)",
        "frozen_model": res["best_model_name"],
        **synthetic_blocks(res),
        "real_pmdata_lopo": real_pmdata_block(),
        "completion_classification_reference": completion_block(),
    }
    (PP1_DIR / "interpretable_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lc = learning_curve_csv(res)
    lc.to_csv(PP1_DIR / "learning_curve.csv", index=False)

    def line(name, s):
        w = s["within_pm1"]
        print(f"{name:42s} n={s['n']:5d} R2={s['R2']:.4f} MAE={s['MAE']:.3f} RMSE={s['RMSE']:.3f} "
              f"within±1={w['pct']:.2f}% ({w['count_within']}/{w['n']})")

    for key in ("synthetic_dev_groupkfold_oof", "synthetic_sealed_holdout", "synthetic_legacy_holdout"):
        b = out[key]
        print(f"\n[{key}] {b['split']} | participants={b['participants']}")
        line("  final model", b["final_model"])
        line("  mean predictor", b.get("mean_predictor", b.get("mean_predictor_fit_on_dev")))
        if "ridge_raw_baseline" in b:
            line("  ridge raw baseline", b["ridge_raw_baseline"])
    pm = out["real_pmdata_lopo"]
    if pm.get("available"):
        print(f"\n[REAL PMData LOPO] participants={pm['participants']} n={pm['n']}")
        line("  mean predictor", pm["mean_predictor"])
        line(f"  best: {pm['best_candidate_name']}", pm["best_candidate"])
    c = out["completion_classification_reference"]
    print("\n[completion classification reference]", c["majority_class_baseline"], c["stored_model_metrics"])
    print("\nlearning curve:\n", lc.drop(columns=["estimator", "protocol"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
