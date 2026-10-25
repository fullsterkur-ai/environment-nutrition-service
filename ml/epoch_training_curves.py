"""
ml/epoch_training_curves.py

Epoch-by-epoch training trace (loss / accuracy curves) for the PP1 RPE model.

WHY THIS FILE EXISTS
--------------------
The deployed PP1 model is a scikit-learn ElasticNet. ElasticNet is solved by
coordinate descent in one call to .fit(), so it has NO epochs and therefore no
Keras-style "loss vs val_loss" history.

To show the training process visually, this script trains an *iterative twin*
of the deployed model:

    same pre-session features  (PreSessionFeatures)
    same preprocessing         (median impute + StandardScaler + one-hot)
    same penalty               (elastic-net, same alpha and l1_ratio as deployed)
    optimiser                  SGDRegressor, one pass over the training data = 1 epoch

After every epoch we measure, on REAL computed predictions:
    loss          = mean squared error on training participants
    val_loss      = mean squared error on held-out validation participants
    accuracy      = % of training predictions within +/-1 RPE point
    val_accuracy  = % of validation predictions within +/-1 RPE point

"accuracy" here is the project's tolerance metric (pp1_metrics.within_tolerance),
NOT classification accuracy.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
* It does NOT retrain, overwrite or replace the deployed model
  (artifacts/pp1_model_optimization/best_rpe_model_pp1.joblib is only READ).
* It does NOT touch the sealed or legacy holdout participants.
* It does NOT modify the dataset.
* Data is the SYNTHETIC development dataset (data_type == SYNTHETIC).

Outputs (new folder, nothing existing is overwritten):
    artifacts/training_curves/epoch_history.csv
    artifacts/training_curves/epoch_summary.json
    artifacts/training_curves/epoch_loss_accuracy_curve.png

Run:
    .\\venv\\Scripts\\python -m ml.epoch_training_curves
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.pp1_metrics import within_tolerance  # noqa: E402
from ml.pp1_splits import GROUP, TARGET, make_splits  # noqa: E402

DEPLOYED_MODEL = ROOT / "artifacts" / "pp1_model_optimization" / "best_rpe_model_pp1.joblib"
OUT_DIR = ROOT / "artifacts" / "training_curves"
HISTORY_CSV = OUT_DIR / "epoch_history.csv"
SUMMARY_JSON = OUT_DIR / "epoch_summary.json"
CURVE_PNG = OUT_DIR / "epoch_loss_accuracy_curve.png"

SEED = 23166660
EPOCHS = 150
VAL_FRACTION = 0.20      # of DEV participants
ETA0 = 2e-5              # small constant step -> gradual, visible convergence over ~80 epochs


def _acc(y, p) -> float:
    return within_tolerance(y, p)["pct"]


def run(epochs: int = EPOCHS) -> pd.DataFrame:
    started = time.time()
    _, parts = make_splits()
    dev = parts["dev"]                       # sealed / legacy holdouts are NOT used

    # 1. participant-grouped train / validation split inside DEV only
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    tr_idx, va_idx = next(gss.split(dev, groups=dev[GROUP]))
    train, val = dev.iloc[tr_idx], dev.iloc[va_idx]
    assert not set(train[GROUP]) & set(val[GROUP]), "participant leakage"

    # 2. reuse the deployed pipeline's (unfitted) feature + preprocessing steps
    deployed = joblib.load(DEPLOYED_MODEL)            # read-only
    alpha = float(deployed.named_steps["model"].alpha)
    l1_ratio = float(deployed.named_steps["model"].l1_ratio)
    features = clone(deployed.named_steps["features"])
    preprocess = clone(deployed.named_steps["preprocess"])

    X_tr = preprocess.fit_transform(features.fit_transform(train))   # fit on TRAIN only
    X_va = preprocess.transform(features.transform(val))
    y_tr = train[TARGET].to_numpy(float)
    y_va = val[TARGET].to_numpy(float)

    # 3. iterative twin: elastic-net penalty, one partial_fit pass = one epoch
    sgd = SGDRegressor(loss="squared_error", penalty="elasticnet", alpha=alpha,
                       l1_ratio=l1_ratio, learning_rate="constant", eta0=ETA0,
                       random_state=SEED)
    rng = np.random.RandomState(SEED)
    rows = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y_tr))
        sgd.partial_fit(X_tr[order], y_tr[order])
        p_tr, p_va = sgd.predict(X_tr), sgd.predict(X_va)
        rows.append({
            "epoch": epoch,
            "loss": mean_squared_error(y_tr, p_tr),
            "val_loss": mean_squared_error(y_va, p_va),
            "accuracy": _acc(y_tr, p_tr),
            "val_accuracy": _acc(y_va, p_va),
        })
    hist = pd.DataFrame(rows)

    # 4. reference lines on the SAME split (all genuinely computed)
    enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000).fit(X_tr, y_tr)
    dummy = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    ref = {
        "elasticnet_closed_form": {
            "val_loss": mean_squared_error(y_va, enet.predict(X_va)),
            "val_accuracy": _acc(y_va, enet.predict(X_va)),
        },
        "dummy_mean": {
            "val_loss": mean_squared_error(y_va, dummy.predict(X_va)),
            "val_accuracy": _acc(y_va, dummy.predict(X_va)),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist.to_csv(HISTORY_CSV, index=False)
    best = hist.loc[hist["val_loss"].idxmin()]
    summary = {
        "IMPORTANT_NOTE": ("Synthetic development data only. Iterative SGD twin of the deployed "
                           "ElasticNet, used to visualise training. The deployed model was NOT "
                           "retrained or modified. Sealed/legacy holdouts were NOT used."),
        "dataset": "data/TEQS_Synthetic_5000_Development_Dataset.csv (SYNTHETIC)",
        "split": f"GroupShuffleSplit(test_size={VAL_FRACTION}, seed={SEED}) on DEV participants",
        "participants": {"train": int(train[GROUP].nunique()), "validation": int(val[GROUP].nunique())},
        "rows": {"train": int(len(train)), "validation": int(len(val))},
        "optimiser": f"SGDRegressor(penalty=elasticnet, alpha={alpha:.6g}, l1_ratio={l1_ratio:.4f}, "
                     f"learning_rate=constant, eta0={ETA0})",
        "epochs": epochs,
        "accuracy_definition": "% of predictions within +/-1 RPE point (tolerance metric, not classification)",
        "first_epoch": hist.iloc[0].to_dict(),
        "final_epoch": hist.iloc[-1].to_dict(),
        "best_val_loss_epoch": best.to_dict(),
        "reference_same_split": ref,
        "runtime_seconds": round(time.time() - started, 1),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(hist, ref, CURVE_PNG)
    return hist


def plot(hist: pd.DataFrame, ref: dict, out: Path) -> Path:
    """Keras-style two-panel figure: loss/val_loss and accuracy/val_accuracy vs epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=140)
    e = hist["epoch"]
    a1.plot(e, hist["loss"], color="#1f77b4", label="loss")
    a1.plot(e, hist["val_loss"], color="#ff7f0e", label="val_loss")
    a1.axhline(ref["elasticnet_closed_form"]["val_loss"], color="gray", ls=":",
               label="deployed-recipe ElasticNet val_loss")
    a1.set_yscale("log")
    a1.set_xlabel("epoch"); a1.set_ylabel("MSE (log scale)")
    a1.set_title("Loss curve"); a1.grid(alpha=.3); a1.legend()

    a2.plot(e, hist["accuracy"], color="#1f77b4", label="accuracy")
    a2.plot(e, hist["val_accuracy"], color="#ff7f0e", label="val_accuracy")
    a2.axhline(ref["dummy_mean"]["val_accuracy"], color="#d62728", ls="--",
               label="DummyRegressor val_accuracy")
    a2.set_xlabel("epoch"); a2.set_ylabel("% predictions within ±1 RPE")
    a2.set_title("Accuracy curve (±1 RPE tolerance)"); a2.grid(alpha=.3); a2.legend()

    fig.suptitle("FullStrength.ai RPE model — epoch training trace (SYNTHETIC development data)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    h = run()
    print(h.iloc[[0, 9, 49, 99, -1]].round(4).to_string(index=False))
    print(f"saved: {HISTORY_CSV}\nsaved: {SUMMARY_JSON}\nsaved: {CURVE_PNG}")
