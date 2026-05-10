"""Notebook 03 logic — EEGNet on raw 64-channel epochs.

Trains EEGNet (Lawhern 2018) per window with early stopping on val macro-F1.
Per-channel z-score normalisation is fit on train and saved alongside the
model so Streamlit can reproduce the inference path.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import compute_metrics, plot_confusion_matrix, plot_training_curves  # noqa: E402
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


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def main() -> None:
    set_torch_seed(SEED)

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    split = load_split(PROCESSED / "splits.json")
    device = _device()
    print("Device:", device)

    windows_evaluated: list[dict] = []
    per_window_artifacts: dict[str, dict] = {}

    for win_label in ["w0.0-4.0", "w0.5-2.5"]:
        print(f"\n=== Window {win_label} ===")
        bundle = _load_window(win_label)
        X = bundle["X_64ch"]; y = bundle["y"]; subj = bundle["subject_ids"]

        train_idx, val_idx, test_idx = split_indices_by_subject(subj, split)
        X_train_raw = X[train_idx]; y_train = y[train_idx]
        X_val_raw = X[val_idx]; y_val = y[val_idx]
        X_test_raw = X[test_idx]; y_test = y[test_idx]
        print(f"  epochs: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

        # Per-channel z-score from train only.
        mean, std = channel_zscore_stats(X_train_raw)
        X_train = apply_zscore(X_train_raw, mean, std)
        X_val = apply_zscore(X_val_raw, mean, std)
        X_test = apply_zscore(X_test_raw, mean, std)

        n_samples = X_train.shape[2]
        cfg = EEGNetConfig(n_channels=64, n_samples=n_samples, n_classes=2)
        set_torch_seed(SEED)
        model = EEGNet(cfg)
        print(f"  params: {count_parameters(model)} | input shape: (64, {n_samples})")

        t0 = time.perf_counter()
        history = train_eegnet(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            device=device,
            max_epochs=200,
            batch_size=64,
            lr=1e-3,
            weight_decay=1e-4,
            patience=20,
            seed=SEED,
            verbose=True,
        )
        train_seconds = time.perf_counter() - t0

        val_pred = predict_eegnet(model, X_val, device)
        val_metrics = compute_metrics(y_val, val_pred)
        print(f"  val accuracy: {val_metrics['accuracy']:.4f} | macroF1: {val_metrics['macro_f1']:.4f} | trained {train_seconds:.1f}s")

        windows_evaluated.append({
            "window": [bundle["tmin"], bundle["tmax"]],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_confusion_matrix": val_metrics["confusion_matrix"],
            "training_time_sec": float(train_seconds),
            "n_train_history_epochs": len(history.train_loss),
        })
        per_window_artifacts[win_label] = {
            "model": model,
            "cfg": cfg,
            "mean": mean,
            "std": std,
            "X_test": X_test,
            "y_test": y_test,
            "history": history,
            "ch_names_64": bundle["ch_names_64"],
            "sfreq": bundle["sfreq"],
            "tmin_tmax": [bundle["tmin"], bundle["tmax"]],
        }

        # Save training curve per window so we can compare both runs.
        plot_training_curves(history, str(FIG_DIR / f"eegnet_training_curves_{win_label}.png"))

    winner_idx = int(np.argmax([w["val_macro_f1"] for w in windows_evaluated]))
    winner = windows_evaluated[winner_idx]
    win_label = f"w{winner['window'][0]:.1f}-{winner['window'][1]:.1f}"
    print(f"\n=== Winning window: {win_label} (val macroF1 {winner['val_macro_f1']:.4f}) ===")

    artifacts = per_window_artifacts[win_label]
    test_pred = predict_eegnet(artifacts["model"], artifacts["X_test"], device)
    test_metrics = compute_metrics(artifacts["y_test"], test_pred)
    print("Test metrics:", test_metrics)

    # Persist model.
    cfg = artifacts["cfg"]
    payload = {
        "state_dict": artifacts["model"].state_dict(),
        "config": {
            "n_channels": cfg.n_channels,
            "n_samples": cfg.n_samples,
            "n_classes": cfg.n_classes,
            "F1": cfg.F1,
            "D": cfg.D,
            "F2": cfg.F2,
            "kernel_length": cfg.kernel_length,
            "pool1": cfg.pool1,
            "pool2": cfg.pool2,
            "dropout": cfg.dropout,
        },
        "channel_mean": artifacts["mean"],
        "channel_std": artifacts["std"],
        "ch_names_64": artifacts["ch_names_64"],
        "sfreq": artifacts["sfreq"],
        "tmin_tmax": artifacts["tmin_tmax"],
    }
    model_path = MODELS_DIR / f"eegnet_{win_label}.pt"
    torch.save(payload, model_path)
    print("Saved model →", model_path)

    plot_confusion_matrix(test_metrics["confusion_matrix"], "EEGNet — test confusion", str(FIG_DIR / "eegnet_confusion.png"))
    # Save the winning training curve under the canonical name too.
    plot_training_curves(artifacts["history"], str(FIG_DIR / "eegnet_training_curves.png"))

    bundle = _load_window(win_label)
    train_idx, val_idx, test_idx = split_indices_by_subject(bundle["subject_ids"], split)

    result = {
        "model_name": "eegnet",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "windows_evaluated": windows_evaluated,
        "winning_window": list(winner["window"]),
        "test_metrics": test_metrics,
        "model_config": {
            "architecture": "EEGNet (Lawhern 2018) F1=8 D=2 F2=16 kernel=64 dropout=0.5",
            "n_params": count_parameters(artifacts["model"]),
            "channels": "all 64 EEG channels",
            "normalization": "per-channel z-score, fit on train",
            "training": {
                "optimizer": "Adam",
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 64,
                "max_epochs": 200,
                "early_stopping_patience": 20,
                "seed": SEED,
                "device": str(device),
            },
        },
        "n_train_epochs": int(len(train_idx)),
        "n_val_epochs": int(len(val_idx)),
        "n_test_epochs": int(len(test_idx)),
        "model_artifact": str(model_path.relative_to(ROOT)),
    }
    out_path = REPORTS_DIR / "eegnet_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("Saved →", out_path)


if __name__ == "__main__":
    main()
