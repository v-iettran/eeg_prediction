"""Phase F (fixed) — Train EEG-TCNet and FBCNet on V2 EA-aligned data.

Fixes applied:
  - Euclidean Alignment already baked into v2_epochs_w0.0-4.0.npz
  - FBCNet uses same 8-30 Hz data as EEGNet/TCNet (no separate wide bandpass)
  - Overlapping filter bands [(8,12)...(26,30)] with 4-window variance
  - LR 1e-4 with 5-epoch linear warmup
  - Label smoothing 0.1
  - Patience 30
  - TCNet dropout 0.4 + MaxNorm(1.0) on spatial conv
  - FBCNet dropout 0.1

Produces:
    models/v2_tcnet.pt, models/v2_fbcnet.pt
    reports/v2_tcnet_fbcnet_results.json
    reports/figures/v2_tcnet_training_curves.png
    reports/figures/v2_fbcnet_training_curves.png
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import compute_metrics, plot_confusion_matrix, plot_training_curves  # noqa: E402
from src.models import (  # noqa: E402
    TrainHistory,
    apply_zscore,
    channel_zscore_stats,
    count_parameters,
    set_torch_seed,
)
from src.tcnet import EEGTCNet, TCNetConfig  # noqa: E402
from src.fbcnet import FBCNet, FBCNetConfig, FilterBankConfig, apply_filter_bank  # noqa: E402
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


def _iter_batches(X, y, batch_size, shuffle, rng):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        sel = idx[start:start + batch_size]
        yield X[sel], y[sel]


def _train_deep_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: torch.device,
    max_epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 30,
    warmup_epochs: int = 5,
    label_smoothing: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> TrainHistory:
    """Training loop with LR warmup, label smoothing, and configurable patience."""
    from sklearn.metrics import f1_score, accuracy_score

    set_torch_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Warmup: linear ramp from lr/10 to lr over warmup_epochs, then ReduceLROnPlateau
    def _warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        return 1.0

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=_warmup_lambda)
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=7)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    history = TrainHistory(train_loss=[], val_loss=[], val_macro_f1=[], val_accuracy=[])
    best_f1 = -np.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_without_improve = 0

    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.int64)).to(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in _iter_batches(X_train, y_train, batch_size, shuffle=True, rng=rng):
            xb_t = torch.from_numpy(xb.astype(np.float32)).to(device)
            yb_t = torch.from_numpy(yb.astype(np.int64)).to(device)
            optim.zero_grad()
            logits = model(xb_t)
            loss = loss_fn(logits, yb_t)
            loss.backward()
            optim.step()
            if hasattr(model, "apply_max_norm"):
                model.apply_max_norm()
            running += loss.item() * len(xb_t)
            n_seen += len(xb_t)
        train_loss = running / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            preds, losses = [], []
            for i in range(0, len(X_val_t), 256):
                xb = X_val_t[i:i + 256]
                yb = y_val_t[i:i + 256]
                logits = model(xb)
                losses.append(loss_fn(logits, yb).item() * len(xb))
                preds.append(logits.argmax(dim=1).cpu().numpy())
            val_loss = float(np.sum(losses) / len(X_val_t))
            val_pred = np.concatenate(preds)

        val_f1 = float(f1_score(y_val, val_pred, average="macro"))
        val_acc = float(accuracy_score(y_val, val_pred))

        history.train_loss.append(float(train_loss))
        history.val_loss.append(val_loss)
        history.val_macro_f1.append(val_f1)
        history.val_accuracy.append(val_acc)

        # Scheduler: warmup phase then plateau
        if epoch <= warmup_epochs:
            warmup_sched.step()
        else:
            plateau_sched.step(val_loss)

        improved = val_f1 > best_f1 + 1e-6
        if improved:
            best_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if verbose and (epoch == 1 or epoch % 10 == 0 or improved):
            cur_lr = optim.param_groups[0]["lr"]
            print(
                f"  epoch {epoch:3d} | lr {cur_lr:.2e} | train_loss {train_loss:.4f} "
                f"| val_loss {val_loss:.4f} | val_macroF1 {val_f1:.4f} "
                f"| val_acc {val_acc:.4f}{'  *' if improved else ''}"
            )

        if epochs_without_improve >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (best val macroF1 {best_f1:.4f})")
            break

    model.load_state_dict(best_state)
    return history


def _predict(model, X, device, batch_size=256):
    model.eval()
    out = []
    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            logits = model(Xt[i:i + batch_size])
            out.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    set_torch_seed(SEED)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    device = _device()
    split = load_split(PROCESSED / "splits.json")
    print(f"Device: {device}")

    results: dict = {
        "phase": "F-fixed",
        "data_version": "v2",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "window": [0.0, 4.0],
        "fixes": [
            "euclidean_alignment",
            "unified_8-30Hz_bandpass_for_fbcnet",
            "overlapping_filter_bands",
            "multi_window_variance",
            "lr_1e-4_with_warmup",
            "label_smoothing_0.1",
            "patience_30",
            "tcnet_dropout_0.4_maxnorm_1.0",
            "fbcnet_dropout_0.1",
        ],
        "models": {},
    }

    # ===== Load data (same for both models now) =====
    arr = np.load(PROCESSED / f"v2_epochs_{WIN_LABEL}.npz", allow_pickle=False)
    X_64ch = arr["X_64ch"]
    y = arr["y"]
    subj = arr["subject_ids"]
    sfreq = float(arr["sfreq"])

    train_idx, val_idx, test_idx = split_indices_by_subject(subj, split)
    print(f"V2 EA-aligned epochs: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # ===== Section 1: EEG-TCNet =====
    print("\n" + "=" * 60)
    print("EEG-TCNet (dropout=0.4, MaxNorm=1.0)")
    print("=" * 60)

    mean, std = channel_zscore_stats(X_64ch[train_idx])
    X_train_z = apply_zscore(X_64ch[train_idx], mean, std)
    X_val_z = apply_zscore(X_64ch[val_idx], mean, std)

    n_samples = X_train_z.shape[2]
    tcnet_cfg = TCNetConfig(n_channels=64, n_samples=n_samples, n_classes=2,
                            dropout=0.4, max_norm=1.0)
    set_torch_seed(SEED)
    tcnet = EEGTCNet(tcnet_cfg)
    print(f"  params: {count_parameters(tcnet)} | input shape: (64, {n_samples})")

    t0 = time.perf_counter()
    tcnet_history = _train_deep_model(
        tcnet, X_train_z, y[train_idx], X_val_z, y[val_idx],
        device=device, max_epochs=200, batch_size=64, lr=1e-4,
        weight_decay=1e-4, patience=30, warmup_epochs=5,
        label_smoothing=0.1, seed=SEED,
    )
    tcnet_train_sec = time.perf_counter() - t0

    val_pred_tcnet = _predict(tcnet, X_val_z, device)
    tcnet_val_metrics = compute_metrics(y[val_idx], val_pred_tcnet)
    print(f"  val accuracy: {tcnet_val_metrics['accuracy']:.4f} | macroF1: {tcnet_val_metrics['macro_f1']:.4f} ({tcnet_train_sec:.1f}s)")

    tcnet_path = MODELS_DIR / "v2_tcnet.pt"
    torch.save({
        "state_dict": tcnet.state_dict(),
        "config": {
            "n_channels": tcnet_cfg.n_channels, "n_samples": tcnet_cfg.n_samples,
            "n_classes": tcnet_cfg.n_classes, "F1": tcnet_cfg.F1, "D": tcnet_cfg.D,
            "F2": tcnet_cfg.F2, "temporal_kernel": tcnet_cfg.temporal_kernel,
            "pool1": tcnet_cfg.pool1, "tcn_kernel": tcnet_cfg.tcn_kernel,
            "tcn_dilations": list(tcnet_cfg.tcn_dilations), "dropout": tcnet_cfg.dropout,
            "max_norm": tcnet_cfg.max_norm,
        },
        "channel_mean": mean,
        "channel_std": std,
        "sfreq": sfreq,
    }, tcnet_path)
    print(f"  saved → {tcnet_path}")

    plot_training_curves(tcnet_history, str(FIG_DIR / "v2_tcnet_training_curves.png"))
    plot_confusion_matrix(
        tcnet_val_metrics["confusion_matrix"],
        "EEG-TCNet (fixed) — val confusion",
        str(FIG_DIR / "v2_tcnet_val_confusion.png"),
    )

    results["models"]["v2_tcnet"] = {
        "val_accuracy": tcnet_val_metrics["accuracy"],
        "val_macro_f1": tcnet_val_metrics["macro_f1"],
        "val_confusion_matrix": tcnet_val_metrics["confusion_matrix"],
        "n_params": count_parameters(tcnet),
        "n_train_history_epochs": len(tcnet_history.train_loss),
        "training_time_sec": float(tcnet_train_sec),
        "model_artifact": str(tcnet_path.relative_to(ROOT)),
    }

    # ===== Section 2: FBCNet (now on same 8-30 Hz data) =====
    print("\n" + "=" * 60)
    print("FBCNet (overlapping bands, multi-window variance, dropout=0.1)")
    print("=" * 60)

    fb_config = FilterBankConfig(sfreq=sfreq)
    print(f"  {len(fb_config.bands)} overlapping bands: {fb_config.bands[0]} ... {fb_config.bands[-1]}")

    t0 = time.perf_counter()
    X_fb_all = apply_filter_bank(X_64ch, fb_config)
    print(f"  Filter bank took {time.perf_counter() - t0:.1f}s, shape={X_fb_all.shape}")

    X_fb_train = X_fb_all[train_idx]
    X_fb_val = X_fb_all[val_idx]

    # Per-channel, per-band z-score normalization
    fb_mean = X_fb_train.mean(axis=(0, 3), keepdims=True).astype(np.float32)
    fb_std = X_fb_train.std(axis=(0, 3), keepdims=True).astype(np.float32)
    fb_std = np.where(fb_std < 1e-8, 1.0, fb_std)

    X_fb_train_z = ((X_fb_train - fb_mean) / fb_std).astype(np.float32)
    X_fb_val_z = ((X_fb_val - fb_mean) / fb_std).astype(np.float32)

    n_samples_fb = X_fb_train_z.shape[3]
    fbcnet_cfg = FBCNetConfig(
        n_channels=64, n_samples=n_samples_fb, n_classes=2,
        n_bands=len(fb_config.bands), m=4, n_windows=4, dropout=0.1,
    )
    set_torch_seed(SEED)
    fbcnet = FBCNet(fbcnet_cfg)
    n_features = fbcnet_cfg.n_bands * fbcnet_cfg.m * fbcnet_cfg.n_windows
    print(f"  params: {count_parameters(fbcnet)} | features: {n_features} ({fbcnet_cfg.n_bands}b × {fbcnet_cfg.m}m × {fbcnet_cfg.n_windows}w)")

    t0 = time.perf_counter()
    fbcnet_history = _train_deep_model(
        fbcnet, X_fb_train_z, y[train_idx], X_fb_val_z, y[val_idx],
        device=device, max_epochs=200, batch_size=64, lr=1e-4,
        weight_decay=1e-4, patience=30, warmup_epochs=5,
        label_smoothing=0.1, seed=SEED,
    )
    fbcnet_train_sec = time.perf_counter() - t0

    val_pred_fbc = _predict(fbcnet, X_fb_val_z, device)
    fbcnet_val_metrics = compute_metrics(y[val_idx], val_pred_fbc)
    print(f"  val accuracy: {fbcnet_val_metrics['accuracy']:.4f} | macroF1: {fbcnet_val_metrics['macro_f1']:.4f} ({fbcnet_train_sec:.1f}s)")

    fbcnet_path = MODELS_DIR / "v2_fbcnet.pt"
    torch.save({
        "state_dict": fbcnet.state_dict(),
        "config": {
            "n_channels": fbcnet_cfg.n_channels, "n_samples": fbcnet_cfg.n_samples,
            "n_classes": fbcnet_cfg.n_classes, "n_bands": fbcnet_cfg.n_bands,
            "m": fbcnet_cfg.m, "n_windows": fbcnet_cfg.n_windows,
            "dropout": fbcnet_cfg.dropout,
        },
        "filter_bank": {
            "bands": fb_config.bands,
            "order": fb_config.order,
            "sfreq": fb_config.sfreq,
        },
        "fb_mean": fb_mean,
        "fb_std": fb_std,
        "sfreq": sfreq,
    }, fbcnet_path)
    print(f"  saved → {fbcnet_path}")

    plot_training_curves(fbcnet_history, str(FIG_DIR / "v2_fbcnet_training_curves.png"))
    plot_confusion_matrix(
        fbcnet_val_metrics["confusion_matrix"],
        "FBCNet (fixed) — val confusion",
        str(FIG_DIR / "v2_fbcnet_val_confusion.png"),
    )

    results["models"]["v2_fbcnet"] = {
        "val_accuracy": fbcnet_val_metrics["accuracy"],
        "val_macro_f1": fbcnet_val_metrics["macro_f1"],
        "val_confusion_matrix": fbcnet_val_metrics["confusion_matrix"],
        "n_params": count_parameters(fbcnet),
        "n_train_history_epochs": len(fbcnet_history.train_loss),
        "training_time_sec": float(fbcnet_train_sec),
        "model_artifact": str(fbcnet_path.relative_to(ROOT)),
    }

    results["n_train_epochs"] = int(len(train_idx))
    results["n_val_epochs"] = int(len(val_idx))
    results["n_test_epochs"] = int(len(test_idx))

    out_path = REPORTS_DIR / "v2_tcnet_fbcnet_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")
    print("Phase F (fixed) complete.")


if __name__ == "__main__":
    main()
