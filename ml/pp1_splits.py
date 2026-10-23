"""
PP1 split definitions (shared by exploration, optimisation and the UI).

Holdout provenance (important for honest reporting):
  * The ORIGINAL baseline (ml/train_models.py, GroupShuffleSplit seed 23166660,
    20 % of participants) picked its "best" model (Ridge) by test MAE, so those
    50 participants were already used for model selection -> "LEGACY holdout".
  * For PP1 optimisation we therefore keep three disjoint participant sets:
        legacy_holdout : the 50 original test participants (never trained/tuned on here;
                         reported only so the baseline comparison is like-for-like)
        sealed_holdout : 50 NEW participants drawn (seeded) from the original train set;
                         touched exactly once, after the final model is frozen
        dev            : the remaining 150 participants; ALL tuning / model selection
                         uses GroupKFold on dev only.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "TEQS_Synthetic_5000_Development_Dataset.csv"
LEGACY_TEST = ROOT / "artifacts" / "test_participants.csv"
LEGACY_TRAIN = ROOT / "artifacts" / "train_participants.csv"

SPLIT_SEED = 20260601  # fixed, independent of the baseline seed
GROUP = "participant_id"
TARGET = "session_rpe_1_10"


def make_splits():
    df = pd.read_csv(CSV)
    legacy_test = set(pd.read_csv(LEGACY_TEST)[GROUP])
    legacy_train = sorted(pd.read_csv(LEGACY_TRAIN)[GROUP])
    assert not legacy_test & set(legacy_train)

    rng = np.random.RandomState(SPLIT_SEED)
    shuffled = list(rng.permutation(legacy_train))
    sealed = set(shuffled[:50])
    dev = set(shuffled[50:])

    parts = {
        "dev": df[df[GROUP].isin(dev)].reset_index(drop=True),
        "sealed_holdout": df[df[GROUP].isin(sealed)].reset_index(drop=True),
        "legacy_holdout": df[df[GROUP].isin(legacy_test)].reset_index(drop=True),
    }
    sets = [set(p[GROUP]) for p in parts.values()]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    return df, parts
