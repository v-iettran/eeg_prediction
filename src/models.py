"""EEGNet (Lawhern et al. 2018) implementation + minimal training loop.

Architecture follows PIPELINE_PLAN.md sec. 7.1 — 64 channels, F1=8, D=2,
F2=16, dropout=0.5, with a ``MaxNorm`` constraint on the depthwise spatial
conv as in the original Keras reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


class _Conv2dWithMaxNorm(nn.Conv2d):
    """Depthwise Conv2d that clamps each output filter's L2 norm to ``max_norm``.

    Applied via ``apply_max_norm()`` after each optimiser step to mimic Keras'
    ``kernel_constraint=max_norm(1.0)`` used in the original EEGNet.
    """

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    @torch.no_grad()
    def apply_max_norm(self) -> None:
        # weight shape: (out_channels, in_channels // groups, kH, kW)
        w = self.weight
        norm = w.norm(p=2, dim=(1, 2, 3), keepdim=True).clamp(min=1e-12)
        desired = norm.clamp(max=self.max_norm)
        w.mul_(desired / norm)


@dataclass
class EEGNetConfig:
    n_channels: int = 64
    n_samples: int = 641  # gets overridden at construction
    n_classes: int = 2
    F1: int = 8
    D: int = 2
    F2: int = 16
    kernel_length: int = 64  # 0.4 s at 160 Hz
    pool1: int = 4
    pool2: int = 8
    dropout: float = 0.5


class EEGNet(nn.Module):
    """EEGNet with a (n_channels, 1) depthwise spatial conv across all channels."""

    def __init__(self, cfg: EEGNetConfig):
        super().__init__()
        self.cfg = cfg

        # Block 1 — temporal conv (1 x kernel_length).
        self.conv_temporal = nn.Conv2d(
            1,
            cfg.F1,
            kernel_size=(1, cfg.kernel_length),
            padding=(0, cfg.kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(cfg.F1)

        # Block 2 — depthwise spatial conv across all channels.
        self.conv_spatial = _Conv2dWithMaxNorm(
            cfg.F1,
            cfg.F1 * cfg.D,
            kernel_size=(cfg.n_channels, 1),
            groups=cfg.F1,
            bias=False,
            max_norm=1.0,
        )
        self.bn2 = nn.BatchNorm2d(cfg.F1 * cfg.D)
        self.pool1 = nn.AvgPool2d((1, cfg.pool1))
        self.drop1 = nn.Dropout(cfg.dropout)

        # Block 3 — separable conv (depthwise temporal, then pointwise).
        sep_kernel = 16
        self.conv_separable_depth = nn.Conv2d(
            cfg.F1 * cfg.D,
            cfg.F1 * cfg.D,
            kernel_size=(1, sep_kernel),
            groups=cfg.F1 * cfg.D,
            padding=(0, sep_kernel // 2),
            bias=False,
        )
        self.conv_separable_point = nn.Conv2d(
            cfg.F1 * cfg.D,
            cfg.F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(cfg.F2)
        self.pool2 = nn.AvgPool2d((1, cfg.pool2))
        self.drop2 = nn.Dropout(cfg.dropout)

        # Determine flattened size once with a dummy forward.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, cfg.n_channels, cfg.n_samples)
            flat = self._forward_features(dummy).flatten(1).shape[1]
        self.classifier = nn.Linear(flat, cfg.n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_temporal(x)
        # padding 'same' for even kernel length yields one extra sample; trim it.
        if x.shape[-1] != self.cfg.n_samples:
            x = x[..., : self.cfg.n_samples]
        x = self.bn1(x)

        x = self.conv_spatial(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv_separable_depth(x)
        # again strip an extra sample if 'same' padding overflows.
        target_len = x.shape[-1]
        if target_len != x.shape[-1]:
            x = x[..., :target_len]
        x = self.conv_separable_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # add channel dim → (B, 1, C, T)
        feats = self._forward_features(x)
        return self.classifier(feats.flatten(1))

    def apply_max_norm(self) -> None:
        self.conv_spatial.apply_max_norm()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def set_torch_seed(seed: int = 42) -> None:
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def channel_zscore_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std fit on training data only.

    X_train shape: (n_epochs, n_channels, n_samples). Returns (mean, std)
    each shape (n_channels, 1) so they broadcast onto epochs.
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True)[0]  # (n_channels, 1)
    std = X_train.std(axis=(0, 2), keepdims=True)[0]
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def _iter_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        sel = idx[start : start + batch_size]
        yield X[sel], y[sel]


@dataclass
class TrainHistory:
    train_loss: list[float]
    val_loss: list[float]
    val_macro_f1: list[float]
    val_accuracy: list[float]


def train_eegnet(
    model: EEGNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: torch.device,
    max_epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> TrainHistory:
    """Train EEGNet with early stopping on val macro-F1.

    Returns the training history. The model is restored to its best-val-F1
    weights in-place before returning.
    """
    from sklearn.metrics import f1_score, accuracy_score

    set_torch_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=5
    )
    loss_fn = nn.CrossEntropyLoss()

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
            model.apply_max_norm()
            running += loss.item() * len(xb_t)
            n_seen += len(xb_t)
        train_loss = running / max(n_seen, 1)

        # Validation.
        model.eval()
        with torch.no_grad():
            # Validate in chunks to be safe on small GPUs.
            preds = []
            losses = []
            for i in range(0, len(X_val_t), 256):
                xb = X_val_t[i : i + 256]
                yb = y_val_t[i : i + 256]
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

        scheduler.step(val_loss)

        improved = val_f1 > best_f1 + 1e-6
        if improved:
            best_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if verbose and (epoch == 1 or epoch % 10 == 0 or improved):
            print(
                f"  epoch {epoch:3d} | train_loss {train_loss:.4f} "
                f"| val_loss {val_loss:.4f} | val_macroF1 {val_f1:.4f} "
                f"| val_acc {val_acc:.4f}{'  *' if improved else ''}"
            )

        if epochs_without_improve >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (best val macroF1 {best_f1:.4f})")
            break

    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict_eegnet(model: EEGNet, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    for i in range(0, len(Xt), batch_size):
        logits = model(Xt[i : i + batch_size])
        out.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,), dtype=np.int64)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
