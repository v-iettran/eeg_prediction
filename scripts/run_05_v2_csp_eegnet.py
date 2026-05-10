"""Phase D — Retrain CSP+LDA and EEGNet on V2 data.

Same architectures/hyperparameters as v1, only the data changes (more
epochs from the relaxed PTP threshold). Evaluates on val; test reserved
for the ensemble.

Produces:
    models/v2_csp_lda.joblib
    models/v2_eegnet.pt
    reports/v2_csp_eegnet_results.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import (  # noqa: E402
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
)
from src.models import (  # noqa: E402
    EEGNet,
    EEGNetConfig,
    apply_zscore,
    channel_zscore_stats,
    count_parameters,
    predict_eegnet,
    set_torch_seed,
    train_eegnet,
)
from src.splits import load_split, split_indices_by_subject  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
SEED = 42
WIN_LABEL = "w0.0-4.0"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_v2_window():
    arr = np.load(PROCESSED / f"v2_epochs_{WIN_LABEL}.npz", allow_pickle=False)
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


def main() -> None:
    np.random.seed(SEED)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    split = load_split(PROCESSED / "splits.json")
    device = _device()
    print(f"Device: {device}")

    bundle = _load_v2_window()
    X_64ch = bundle["X_64ch"]
    y = bundle["y"]
    subj = bundle["subject_ids"]
    ch_names_64 = bundle["ch_names_64"]
    sfreq = bundle["sfreq"]

    train_idx, val_idx, test_idx = split_indices_by_subject(subj, split)
    print(f"V2 epochs [{WIN_LABEL}]: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    results: dict = {
        "phase": "D",
        "data_version": "v2",
        "rejection_threshold_uv": 300,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "window": [0.0, 4.0],
        "n_train_epochs": int(len(train_idx)),
        "n_val_epochs": int(len(val_idx)),
        "n_test_epochs": int(len(test_idx)),
        "models": {},
    }

    # ===== CSP+LDA v2 =====
    print("\n" + "=" * 60)
    print("CSP+LDA v2")
    print("=" * 60)

    X_csp = X_64ch.astype(np.float64)
    X_train_csp = X_csp[train_idx]; y_train = y[train_idx]; g_train = subj[train_idx]
    X_val_csp = X_csp[val_idx]; y_val = y[val_idx]

    gkf = GroupKFold(n_splits=5)
    csp_pipe = Pipeline([
        ("csp", CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis()),
    ])

    t0 = time.perf_counter()
    gs = GridSearchCV(
        csp_pipe,
        param_grid={"csp__n_components": [4, 6, 8]},
        cv=gkf,
        scoring="f1_macro",
        n_jobs=1,
        refit=True,
    )
    gs.fit(X_train_csp, y_train, groups=g_train)
    csp_train_sec = time.perf_counter() - t0
    print(f"  best params: {gs.best_params_}  CV macroF1: {gs.best_score_:.4f} ({csp_train_sec:.1f}s)")

    best_csp = gs.best_estimator_
    val_pred_csp = best_csp.predict(X_val_csp)
    csp_val_metrics = compute_metrics(y_val, val_pred_csp)
    print(f"  val accuracy: {csp_val_metrics['accuracy']:.4f} | macroF1: {csp_val_metrics['macro_f1']:.4f}")

    csp_path = MODELS_DIR / "v2_csp_lda.joblib"
    joblib.dump(best_csp, csp_path)
    print(f"  saved → {csp_path}")

    plot_confusion_matrix(
        csp_val_metrics["confusion_matrix"],
        "CSP+LDA v2 — val confusion",
        str(FIG_DIR / "v2_csp_lda_val_confusion.png"),
    )

    results["models"]["v2_csp_lda"] = {
        "val_accuracy": csp_val_metrics["accuracy"],
        "val_macro_f1": csp_val_metrics["macro_f1"],
        "val_confusion_matrix": csp_val_metrics["confusion_matrix"],
        "best_n_components": int(gs.best_params_["csp__n_components"]),
        "training_time_sec": float(csp_train_sec),
        "model_artifact": str(csp_path.relative_to(ROOT)),
    }

    # ===== EEGNet v2 =====
    print("\n" + "=" * 60)
    print("EEGNet v2")
    print("=" * 60)

    X_train_raw = X_64ch[train_idx]
    X_val_raw = X_64ch[val_idx]

    mean, std = channel_zscore_stats(X_train_raw)
    X_train_z = apply_zscore(X_train_raw, mean, std)
    X_val_z = apply_zscore(X_val_raw, mean, std)

    n_samples = X_train_z.shape[2]
    cfg = EEGNetConfig(n_channels=64, n_samples=n_samples, n_classes=2)
    set_torch_seed(SEED)
    model = EEGNet(cfg)
    print(f"  params: {count_parameters(model)} | input shape: (64, {n_samples})")

    t0 = time.perf_counter()
    history = train_eegnet(
        model, X_train_z, y[train_idx], X_val_z, y[val_idx],
        device=device, max_epochs=200, batch_size=64, lr=1e-3,
        weight_decay=1e-4, patience=30, seed=SEED, verbose=True,
    )
    eegnet_train_sec = time.perf_counter() - t0

    val_pred_eeg = predict_eegnet(model, X_val_z, device)
    eeg_val_metrics = compute_metrics(y[val_idx], val_pred_eeg)
    print(f"  val accuracy: {eeg_val_metrics['accuracy']:.4f} | macroF1: {eeg_val_metrics['macro_f1']:.4f} ({eegnet_train_sec:.1f}s)")

    eegnet_path = MODELS_DIR / "v2_eegnet.pt"
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "n_channels": cfg.n_channels, "n_samples": cfg.n_samples,
            "n_classes": cfg.n_classes, "F1": cfg.F1, "D": cfg.D,
            "F2": cfg.F2, "kernel_length": cfg.kernel_length,
            "pool1": cfg.pool1, "pool2": cfg.pool2, "dropout": cfg.dropout,
        },
        "channel_mean": mean,
        "channel_std": std,
        "ch_names_64": ch_names_64,
        "sfreq": sfreq,
        "tmin_tmax": [bundle["tmin"], bundle["tmax"]],
    }
    torch.save(payload, eegnet_path)
    print(f"  saved → {eegnet_path}")

    plot_confusion_matrix(
        eeg_val_metrics["confusion_matrix"],
        "EEGNet v2 — val confusion",
        str(FIG_DIR / "v2_eegnet_val_confusion.png"),
    )
    plot_training_curves(history, str(FIG_DIR / "v2_eegnet_training_curves.png"))

    results["models"]["v2_eegnet"] = {
        "val_accuracy": eeg_val_metrics["accuracy"],
        "val_macro_f1": eeg_val_metrics["macro_f1"],
        "val_confusion_matrix": eeg_val_metrics["confusion_matrix"],
        "n_params": count_parameters(model),
        "n_train_history_epochs": len(history.train_loss),
        "training_time_sec": float(eegnet_train_sec),
        "model_artifact": str(eegnet_path.relative_to(ROOT)),
        "device": str(device),
    }

    out_path = REPORTS_DIR / "v2_csp_eegnet_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")
    print("Phase D complete.")


if __name__ == "__main__":
    main()
