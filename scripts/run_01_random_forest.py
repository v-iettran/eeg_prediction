"""Notebook 01 logic — Random Forest on band-power features.

Trains and evaluates a RandomForestClassifier on log mu/beta band-power
features extracted from the 3-channel motor view (C3, Cz, C4) for both
windows. Picks the winning window by val macro-F1, then computes test
metrics on the winning window only.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import compute_metrics, plot_confusion_matrix, plot_feature_importance  # noqa: E402
from src.features import bandpower_features, feature_names, MU_BAND, BETA_BAND  # noqa: E402
from src.preprocessing import MOTOR_CHANNELS  # noqa: E402
from src.splits import load_split, split_indices_by_subject  # noqa: E402


PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
SEED = 42


def _load_window(win_label: str):
    arr = np.load(PROCESSED / f"epochs_{win_label}.npz", allow_pickle=False)
    return {
        "X_3ch": arr["X_3ch"],
        "X_64ch": arr["X_64ch"],
        "y": arr["y"],
        "subject_ids": arr["subject_ids"],
        "ch_names_64": arr["ch_names_64"].tolist(),
        "sfreq": float(arr["sfreq"]),
        "tmin": float(arr["tmin"]),
        "tmax": float(arr["tmax"]),
    }


def _build_pipeline(rf_kwargs):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(**rf_kwargs)),
    ])


def main() -> None:
    np.random.seed(SEED)

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    split = load_split(PROCESSED / "splits.json")
    print("Loaded split: train", len(split["train"]), "val", len(split["val"]), "test", len(split["test"]))

    feat_names = feature_names(MOTOR_CHANNELS)
    print("Feature names:", feat_names)

    windows_evaluated: list[dict] = []
    per_window_artifacts: dict[str, dict] = {}

    for win_label in ["w0.0-4.0", "w0.5-2.5"]:
        print(f"\n=== Window {win_label} ===")
        bundle = _load_window(win_label)
        X = bundle["X_3ch"]; y = bundle["y"]; subj = bundle["subject_ids"]
        sfreq = bundle["sfreq"]

        train_idx, val_idx, test_idx = split_indices_by_subject(subj, split)
        print(f"  epochs: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

        # Feature extraction.
        feats_all = bandpower_features(X, sfreq=sfreq, bands=(MU_BAND, BETA_BAND))
        F_train = feats_all[train_idx]; y_train = y[train_idx]; g_train = subj[train_idx]
        F_val = feats_all[val_idx]; y_val = y[val_idx]
        F_test = feats_all[test_idx]; y_test = y[test_idx]

        base_kwargs = dict(
            n_estimators=500,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        )

        # Subject-disjoint group K-fold for hyperparameter search.
        gkf = GroupKFold(n_splits=5)
        param_grid = {
            "rf__max_depth": [None, 10, 20],
            "rf__min_samples_leaf": [1, 2, 5],
        }
        pipe = _build_pipeline(base_kwargs)

        t0 = time.perf_counter()
        gs = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=gkf,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(F_train, y_train, groups=g_train)
        train_seconds = time.perf_counter() - t0
        print(f"  best CV params: {gs.best_params_}  best CV macroF1: {gs.best_score_:.4f}")
        print(f"  trained in {train_seconds:.1f}s")

        best_pipe = gs.best_estimator_

        # Validation metrics.
        val_pred = best_pipe.predict(F_val)
        val_metrics = compute_metrics(y_val, val_pred)
        print(f"  val accuracy: {val_metrics['accuracy']:.4f} | macroF1: {val_metrics['macro_f1']:.4f}")

        windows_evaluated.append({
            "window": [bundle["tmin"], bundle["tmax"]],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_confusion_matrix": val_metrics["confusion_matrix"],
            "training_time_sec": float(train_seconds),
            "best_params": gs.best_params_,
        })
        per_window_artifacts[win_label] = {
            "pipe": best_pipe,
            "F_test": F_test,
            "y_test": y_test,
            "feat_names": feat_names,
            "tmin_tmax": [bundle["tmin"], bundle["tmax"]],
        }

    # Pick winning window by val macroF1.
    winner_idx = int(np.argmax([w["val_macro_f1"] for w in windows_evaluated]))
    winner = windows_evaluated[winner_idx]
    win_label = f"w{winner['window'][0]:.1f}-{winner['window'][1]:.1f}"
    print(f"\n=== Winning window: {win_label} (val macroF1 {winner['val_macro_f1']:.4f}) ===")

    artifacts = per_window_artifacts[win_label]
    test_pred = artifacts["pipe"].predict(artifacts["F_test"])
    test_metrics = compute_metrics(artifacts["y_test"], test_pred)
    print("Test metrics:", test_metrics)

    # Persist model.
    model_path = MODELS_DIR / f"rf_{win_label}.joblib"
    joblib.dump(artifacts["pipe"], model_path)
    print("Saved model →", model_path)

    # Figures.
    plot_confusion_matrix(test_metrics["confusion_matrix"], "RF — test confusion", str(FIG_DIR / "rf_confusion.png"))
    rf = artifacts["pipe"].named_steps["rf"]
    plot_feature_importance(rf.feature_importances_, artifacts["feat_names"], str(FIG_DIR / "rf_feature_importance.png"))

    # Result JSON.
    n_train = sum(int(s) for s in [len(split_indices_by_subject(_load_window(win_label)["subject_ids"], split)[0])])
    bundle = _load_window(win_label)
    train_idx, val_idx, test_idx = split_indices_by_subject(bundle["subject_ids"], split)

    result = {
        "model_name": "random_forest",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "windows_evaluated": windows_evaluated,
        "winning_window": list(winner["window"]),
        "test_metrics": test_metrics,
        "model_config": {
            "estimator": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": 500,
            "class_weight": "balanced",
            "best_params": winner["best_params"],
            "features": "log mu (8-13Hz) + log beta (13-30Hz) band power, 3 motor channels (C3, Cz, C4)",
            "scaler": "StandardScaler (fit on train)",
            "cv": "GroupKFold(n_splits=5) with subject-disjoint folds",
        },
        "feature_importances": dict(zip(artifacts["feat_names"], [float(v) for v in rf.feature_importances_])),
        "n_train_epochs": int(len(train_idx)),
        "n_val_epochs": int(len(val_idx)),
        "n_test_epochs": int(len(test_idx)),
        "model_artifact": str(model_path.relative_to(ROOT)),
    }
    out_path = REPORTS_DIR / "rf_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("Saved →", out_path)


if __name__ == "__main__":
    main()
