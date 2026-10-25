"""
tests/test_epoch_training_curves.py

Verifies the saved epoch training trace (loss / val_loss / accuracy / val_accuracy)
and re-draws the Keras-style curve into tests/model_curves/ so it can be opened
straight from the tests folder during the viva.

Source of every point: artifacts/training_curves/epoch_history.csv, produced by
    .\\venv\\Scripts\\python -m ml.epoch_training_curves
Nothing here retrains or changes the deployed model.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from ml.epoch_training_curves import HISTORY_CSV, SUMMARY_JSON, plot

TEST_IMG = Path(__file__).parent / "model_curves" / "epoch_loss_accuracy_curve.png"


@pytest.fixture(scope="module")
def history():
    if not HISTORY_CSV.is_file():
        pytest.skip("run `python -m ml.epoch_training_curves` first")
    return pd.read_csv(HISTORY_CSV)


@pytest.fixture(scope="module")
def summary():
    return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))


def test_history_has_keras_style_columns(history):
    assert list(history.columns) == ["epoch", "loss", "val_loss", "accuracy", "val_accuracy"]
    assert history["epoch"].is_monotonic_increasing and len(history) >= 50


def test_loss_decreases_during_training(history):
    assert history["loss"].iloc[-1] < history["loss"].iloc[0]
    assert history["val_loss"].iloc[-1] < history["val_loss"].iloc[0]


def test_accuracy_increases_during_training(history):
    assert history["val_accuracy"].iloc[-1] > history["val_accuracy"].iloc[0]


def test_no_large_overfitting_gap(history):
    last = history.iloc[-10:]
    assert (last["val_loss"] - last["loss"]).mean() < 0.10


def test_converged_model_beats_dummy_baseline(history, summary):
    dummy = summary["reference_same_split"]["dummy_mean"]
    assert history["val_loss"].iloc[-1] < dummy["val_loss"]
    assert history["val_accuracy"].iloc[-1] > dummy["val_accuracy"]


def test_provenance_is_marked_synthetic(summary):
    assert "SYNTHETIC" in summary["dataset"]
    assert "NOT" in summary["IMPORTANT_NOTE"]


def test_draw_curve_into_tests_folder(history, summary):
    out = plot(history, summary["reference_same_split"], TEST_IMG)
    assert out.is_file() and out.stat().st_size > 10_000
