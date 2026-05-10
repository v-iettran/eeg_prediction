"""Deterministic subject-level train/val/test split for the eegmmidb dataset.

The split is materialised once by Notebook 00 into ``data/processed/splits.json``
and then read by every model notebook so all three classifiers train, validate,
and test on identical subject pools. Never re-shuffle in a downstream notebook.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# Subjects with non-standard sampling rates / annotation issues. These are
# well documented in BCI literature on this dataset and are dropped before
# splitting so they never enter any of train, val, or test.
KNOWN_BAD_SUBJECTS: list[int] = [88, 89, 92, 100, 104]

ALL_SUBJECTS: list[int] = list(range(1, 110))  # 1..109 inclusive

DEFAULT_SEED = 42
N_VAL = 12
N_TEST = 12


def build_subject_split(
    seed: int = DEFAULT_SEED,
    n_val: int = N_VAL,
    n_test: int = N_TEST,
    bad_subjects: list[int] | None = None,
) -> dict:
    """Build a deterministic subject-level split.

    Returns a dict with ``train``, ``val``, ``test`` (lists of subject ids)
    and ``dropped`` (the bad subjects excluded from the pool).
    """
    if bad_subjects is None:
        bad_subjects = list(KNOWN_BAD_SUBJECTS)

    pool = [s for s in ALL_SUBJECTS if s not in set(bad_subjects)]
    rng = random.Random(seed)
    shuffled = pool.copy()
    rng.shuffle(shuffled)

    test = sorted(shuffled[:n_test])
    val = sorted(shuffled[n_test : n_test + n_val])
    train = sorted(shuffled[n_test + n_val :])

    return {
        "seed": seed,
        "dropped": sorted(bad_subjects),
        "train": train,
        "val": val,
        "test": test,
    }


def save_split(split: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def split_indices_by_subject(subject_ids, split: dict):
    """Return (train_idx, val_idx, test_idx) for a flat array of subject ids per epoch."""
    import numpy as np

    subject_ids = np.asarray(subject_ids)
    train_set = set(split["train"])
    val_set = set(split["val"])
    test_set = set(split["test"])

    train_idx = np.where(np.isin(subject_ids, list(train_set)))[0]
    val_idx = np.where(np.isin(subject_ids, list(val_set)))[0]
    test_idx = np.where(np.isin(subject_ids, list(test_set)))[0]
    return train_idx, val_idx, test_idx
