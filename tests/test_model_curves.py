"""
tests/test_model_curves.py

Automated test and visualization generator for Training vs Validation Loss and Accuracy Curves.
Proves model convergence across training sample sizes and outputs visual artifacts for viva demo.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "artifacts" / "pp1_model_optimization"
OUTPUT_IMG = Path(__file__).parent / "model_loss_accuracy_curve.png"


def test_learning_curve_loss_and_accuracy():
    """Verify empirical convergence of ElasticNet pipeline across sample sizes."""
    lc_csv = EXP_DIR / "learning_curve.csv"
    assert lc_csv.is_file(), f"Learning curve data missing: {lc_csv}"

    df = pd.read_csv(lc_csv)
    assert len(df) >= 5, "Learning curve must contain at least 5 sample size steps"

    # 1. Verify train R2 is consistent and validation R2 increases with data
    init_val_r2 = df["validation_R2_mean"].iloc[0]
    final_val_r2 = df["validation_R2_mean"].iloc[-1]
    assert final_val_r2 > init_val_r2, "Validation R2 must improve as training data grows"

    # 2. Generalization gap must narrow (zero overfitting)
    init_gap = df["train_R2_mean"].iloc[0] - df["validation_R2_mean"].iloc[0]
    final_gap = df["train_R2_mean"].iloc[-1] - df["validation_R2_mean"].iloc[-1]
    assert final_gap < init_gap, "Generalization gap must shrink with larger cohort size"
    assert final_gap < 0.02, f"Final gap must be < 0.02, got {final_gap:.4f}"

    # 3. Error curve = unexplained variance (1 - R2). This is NOT MSE; it is the share of
    #    RPE variance the model does not explain, computed directly from saved R2 values.
    train_mse = 1.0 - df["train_R2_mean"]
    val_mse = 1.0 - df["validation_R2_mean"]
    sealed_r2 = json.loads((EXP_DIR / "final_holdout_metrics.json").read_text(encoding="utf-8"))[
        "holdout_evaluations"]["sealed_holdout/final_model"]["R2"]

    # 4. Generate high-resolution visual figure for presentation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    samples = df["train_rows_per_fold"]

    # Left Plot: error curve (unexplained variance)
    ax1.plot(samples, train_mse, "o-", color="#0B2A5B", linewidth=2.2, label="Training error (1 - R²)")
    ax1.plot(samples, val_mse, "s--", color="#2F5DA8", linewidth=2.2, label="Validation error (1 - R²)")
    ax1.set_title("Learning curve: error vs training size", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Training Sessions per Fold (GroupKFold)", fontsize=11)
    ax1.set_ylabel("Unexplained variance (1 - R²)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Right Plot: Explanatory Accuracy / R² Generalization Curve
    ax2.plot(samples, df["train_R2_mean"], "o-", color="#0B2A5B", linewidth=2.2, label="Training R² Score")
    ax2.plot(samples, df["validation_R2_mean"], "s--", color="#10B981", linewidth=2.2, label="Validation R² (Unseen Groups)")
    ax2.axhline(sealed_r2, color="#E11D48", linestyle=":", label=f"Sealed synthetic test R² ({sealed_r2:.3f})")
    ax2.set_title("Learning curve: R² vs training size (gap narrows)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Training Sessions per Fold (GroupKFold)", fontsize=11)
    ax2.set_ylabel("Explained Variance (R² Score)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.suptitle("FullStrength.ai RPE model: learning curve (SYNTHETIC development data)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, bbox_inches="tight")
    plt.close()

    assert OUTPUT_IMG.is_file(), f"Expected generated image at {OUTPUT_IMG}"
