"""Notebook 02 logic — CSP + LDA on 64-channel epochs.

Pipeline: ``mne.decoding.CSP`` (with Ledoit-Wolf shrinkage) → LDA. Light grid
over ``n_components in {4, 6, 8}`` via subject-disjoint GroupKFold per
window. Picks the winning window by val macro-F1 then evaluates on test.
Saves CSP scalp-pattern topomaps for the demo.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import compute_metrics, plot_confusion_matrix  # noqa: E402
from src.splits import load_split, split_indices_by_subject  # noqa: E402


PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
SEED = 42


def _load_window(win_label: str):
    arr = np.load(PROCESSED / f"epochs_{win_label}.npz", allow_pickle=False)
    return {
        "X_64ch": arr["X_64ch"],
        "y": arr["y"],
        "subject_ids": arr["subject_ids"],
        "ch_names_64": arr["ch_names_64"].tolist(),
        "sfreq": float(arr["sfreq"]),
        "tmin": float(arr["tmin"]),
        "tmax": float(arr["tmax"]),
    }


def _build_pipeline(n_components: int):
    return Pipeline([
        ("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


def _plot_csp_topomaps(csp: CSP, ch_names_64: list[str], sfreq: float, save_path: Path) -> None:
    info = mne.create_info(ch_names=ch_names_64, sfreq=sfreq, ch_types="eeg")
    info.set_montage("standard_1005")
    patterns = csp.patterns_  # (n_components, n_channels)
    n = patterns.shape[0]
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            mne.viz.plot_topomap(patterns[i], info, axes=ax, show=False, contours=4, cmap="RdBu_r")
            ax.set_title(f"CSP {i + 1}", fontsize=10)
        else:
            ax.axis("off")
    fig.suptitle("CSP scalp patterns (red/blue = polarity)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    np.random.seed(SEED)

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    split = load_split(PROCESSED / "splits.json")

    windows_evaluated: list[dict] = []
    per_window_artifacts: dict[str, dict] = {}

    for win_label in ["w0.0-4.0", "w0.5-2.5"]:
        print(f"\n=== Window {win_label} ===")
        bundle = _load_window(win_label)
        X = bundle["X_64ch"].astype(np.float64)  # CSP wants float64 covariances
        y = bundle["y"]; subj = bundle["subject_ids"]

        train_idx, val_idx, test_idx = split_indices_by_subject(subj, split)
        print(f"  epochs: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

        X_train = X[train_idx]; y_train = y[train_idx]; g_train = subj[train_idx]
        X_val = X[val_idx]; y_val = y[val_idx]
        X_test = X[test_idx]; y_test = y[test_idx]

        gkf = GroupKFold(n_splits=5)
        param_grid = {"csp__n_components": [4, 6, 8]}
        pipe = _build_pipeline(n_components=6)

        t0 = time.perf_counter()
        gs = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=gkf,
            scoring="f1_macro",
            n_jobs=1,  # CSP is parallel internally; outer parallelism causes RAM pressure
            refit=True,
        )
        gs.fit(X_train, y_train, groups=g_train)
        train_seconds = time.perf_counter() - t0
        print(f"  best CV params: {gs.best_params_}  best CV macroF1: {gs.best_score_:.4f}  ({train_seconds:.1f}s)")

        best_pipe = gs.best_estimator_

        val_pred = best_pipe.predict(X_val)
        val_metrics = compute_metrics(y_val, val_pred)
        print(f"  val accuracy: {val_metrics['accuracy']:.4f} | macroF1: {val_metrics['macro_f1']:.4f}")

        windows_evaluated.append({
            "window": [bundle["tmin"], bundle["tmax"]],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_confusion_matrix": val_metrics["confusion_matrix"],
            "training_time_sec": float(train_seconds),
            "best_n_components": int(gs.best_params_["csp__n_components"]),
        })
        per_window_artifacts[win_label] = {
            "pipe": best_pipe,
            "X_test": X_test,
            "y_test": y_test,
            "ch_names_64": bundle["ch_names_64"],
            "sfreq": bundle["sfreq"],
            "tmin_tmax": [bundle["tmin"], bundle["tmax"]],
        }

    winner_idx = int(np.argmax([w["val_macro_f1"] for w in windows_evaluated]))
    winner = windows_evaluated[winner_idx]
    win_label = f"w{winner['window'][0]:.1f}-{winner['window'][1]:.1f}"
    print(f"\n=== Winning window: {win_label} (val macroF1 {winner['val_macro_f1']:.4f}) ===")

    artifacts = per_window_artifacts[win_label]
    test_pred = artifacts["pipe"].predict(artifacts["X_test"])
    test_metrics = compute_metrics(artifacts["y_test"], test_pred)
    print("Test metrics:", test_metrics)

    # Persist model.
    model_path = MODELS_DIR / f"csp_lda_{win_label}.joblib"
    joblib.dump(artifacts["pipe"], model_path)
    print("Saved model →", model_path)

    # Figures.
    plot_confusion_matrix(test_metrics["confusion_matrix"], "CSP+LDA — test confusion", str(FIG_DIR / "csp_lda_confusion.png"))
    csp_step = artifacts["pipe"].named_steps["csp"]
    _plot_csp_topomaps(csp_step, artifacts["ch_names_64"], artifacts["sfreq"], FIG_DIR / "csp_topomaps.png")

    bundle = _load_window(win_label)
    train_idx, val_idx, test_idx = split_indices_by_subject(bundle["subject_ids"], split)

    result = {
        "model_name": "csp_lda",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "windows_evaluated": windows_evaluated,
        "winning_window": list(winner["window"]),
        "winning_n_components": winner["best_n_components"],
        "test_metrics": test_metrics,
        "model_config": {
            "csp": {
                "estimator": "mne.decoding.CSP",
                "n_components": winner["best_n_components"],
                "reg": "ledoit_wolf",
                "log": True,
                "norm_trace": False,
            },
            "classifier": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
            "channels": "all 64 EEG channels",
            "cv": "GroupKFold(n_splits=5) with subject-disjoint folds",
        },
        "n_train_epochs": int(len(train_idx)),
        "n_val_epochs": int(len(val_idx)),
        "n_test_epochs": int(len(test_idx)),
        "model_artifact": str(model_path.relative_to(ROOT)),
    }
    out_path = REPORTS_DIR / "csp_lda_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("Saved →", out_path)


if __name__ == "__main__":
    main()
