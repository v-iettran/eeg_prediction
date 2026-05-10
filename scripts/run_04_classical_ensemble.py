"""Phase C — Classical model pool on enriched V2 features.

Trains RF v2, XGBoost, SVM (RBF), and Logistic Regression on 31
enriched features from the 3-channel motor view. All use [0.0, 4.0]
window only (v1 winner). Evaluates on val; test is reserved for the
ensemble.

Produces:
    models/v2_rf.joblib, v2_xgb.joblib, v2_svm.joblib, v2_lr.joblib
    reports/v2_classical_results.json
    reports/figures/v2_feature_importance.png
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import compute_metrics, plot_confusion_matrix, plot_feature_importance  # noqa: E402
from src.features_v2 import extract_features_v2, feature_names_v2  # noqa: E402
from src.splits import load_split, split_indices_by_subject  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
SEED = 42
WIN_LABEL = "w0.0-4.0"


def _load_v2_window():
    arr = np.load(PROCESSED / f"v2_epochs_{WIN_LABEL}.npz", allow_pickle=False)
    return {
        "X_3ch": arr["X_3ch"],
        "y": arr["y"],
        "subject_ids": arr["subject_ids"],
        "sfreq": float(arr["sfreq"]),
    }


MODEL_SPECS = {
    "v2_rf": {
        "clf": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=SEED
        ),
        "param_grid": {
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_leaf": [1, 2, 5],
        },
    },
    "v2_xgb": {
        "clf": XGBClassifier(
            n_estimators=300, eval_metric="logloss", random_state=SEED, verbosity=0
        ),
        "param_grid": {
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.1],
            "clf__subsample": [0.8, 1.0],
        },
    },
    "v2_svm": {
        "clf": SVC(
            kernel="rbf", class_weight="balanced", probability=True, random_state=SEED
        ),
        "param_grid": {
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", "auto"],
        },
    },
    "v2_lr": {
        "clf": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED, solver="saga"
        ),
        "param_grid": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l1", "l2"],
        },
    },
}


def main() -> None:
    np.random.seed(SEED)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    split = load_split(PROCESSED / "splits.json")
    bundle = _load_v2_window()
    X_3ch = bundle["X_3ch"]
    y = bundle["y"]
    subj = bundle["subject_ids"]
    sfreq = bundle["sfreq"]

    train_idx, val_idx, test_idx = split_indices_by_subject(subj, split)
    print(f"Epochs: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    print("Extracting v2 features (31-dim)...")
    t0 = time.perf_counter()
    feats_all = extract_features_v2(X_3ch, sfreq)
    print(f"  Feature extraction took {time.perf_counter() - t0:.1f}s, shape={feats_all.shape}")
    feat_names = feature_names_v2()

    F_train = feats_all[train_idx]; y_train = y[train_idx]; g_train = subj[train_idx]
    F_val = feats_all[val_idx]; y_val = y[val_idx]
    F_test = feats_all[test_idx]; y_test = y[test_idx]

    gkf = GroupKFold(n_splits=5)

    model_results: list[dict] = []

    for name, spec in MODEL_SPECS.items():
        print(f"\n=== {name} ===")
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", spec["clf"])])

        t0 = time.perf_counter()
        gs = GridSearchCV(
            pipe,
            param_grid=spec["param_grid"],
            cv=gkf,
            scoring="f1_macro",
            n_jobs=-1 if name != "v2_svm" else 1,
            refit=True,
        )
        gs.fit(F_train, y_train, groups=g_train)
        train_sec = time.perf_counter() - t0
        print(f"  best CV params: {gs.best_params_}")
        print(f"  best CV macroF1: {gs.best_score_:.4f} ({train_sec:.1f}s)")

        best_pipe = gs.best_estimator_

        val_pred = best_pipe.predict(F_val)
        val_metrics = compute_metrics(y_val, val_pred)
        print(f"  val accuracy: {val_metrics['accuracy']:.4f} | macroF1: {val_metrics['macro_f1']:.4f}")

        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(best_pipe, model_path)
        print(f"  saved → {model_path}")

        plot_confusion_matrix(
            val_metrics["confusion_matrix"],
            f"{name} — val confusion",
            str(FIG_DIR / f"{name}_val_confusion.png"),
        )

        entry = {
            "name": name,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_confusion_matrix": val_metrics["confusion_matrix"],
            "best_params": {k: str(v) for k, v in gs.best_params_.items()},
            "cv_best_score": float(gs.best_score_),
            "training_time_sec": float(train_sec),
            "model_artifact": str(model_path.relative_to(ROOT)),
        }

        if name == "v2_rf":
            rf_model = best_pipe.named_steps["clf"]
            entry["feature_importances"] = dict(
                zip(feat_names, [float(v) for v in rf_model.feature_importances_])
            )
            plot_feature_importance(
                rf_model.feature_importances_,
                feat_names,
                str(FIG_DIR / "v2_feature_importance.png"),
                top_k=15,
            )

        model_results.append(entry)

    best_model = max(model_results, key=lambda m: m["val_macro_f1"])
    print(f"\n=== Best classical model: {best_model['name']} (val macroF1 {best_model['val_macro_f1']:.4f}) ===")

    result = {
        "phase": "C",
        "data_version": "v2",
        "rejection_threshold_uv": 300,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "window": [0.0, 4.0],
        "n_features": 31,
        "feature_names": feat_names,
        "n_train_epochs": int(len(train_idx)),
        "n_val_epochs": int(len(val_idx)),
        "n_test_epochs": int(len(test_idx)),
        "models": model_results,
        "best_classical_model": best_model["name"],
    }
    out_path = REPORTS_DIR / "v2_classical_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
