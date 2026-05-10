"""FBCNet (Mane et al. 2021) — Filter Bank CSP Network.

Architecture:
    Preprocessing: Overlapping filter bank within the 8-30 Hz range
    Block 1: Per-band depthwise spatial convolution
    Block 2: Multi-window LogVarLayer (segment time into windows → variance → log)
    Classifier: Linear

The filter bank is applied offline (not a learned layer).
Uses the same 8-30 Hz bandpass data as EEGNet and TCNet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt


# --- Filter Bank (preprocessing, not nn.Module) ---

@dataclass
class FilterBankConfig:
    """Overlapping 4-Hz-wide bands spanning the mu-beta range."""
    bands: list[tuple[float, float]] = field(default_factory=lambda: [
        (8, 12), (10, 14), (12, 16), (14, 18), (16, 20),
        (18, 22), (20, 24), (22, 26), (24, 28), (26, 30),
    ])
    order: int = 5
    sfreq: float = 160.0


def apply_filter_bank(
    X: np.ndarray,
    config: FilterBankConfig,
) -> np.ndarray:
    """Apply multi-band filter bank to epochs already bandpassed at 8-30 Hz.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_samples)
    config : FilterBankConfig

    Returns
    -------
    X_fb : ndarray, shape (n_epochs, n_bands, n_channels, n_samples)
    """
    n_epochs, n_ch, n_samples = X.shape
    n_bands = len(config.bands)
    nyq = config.sfreq / 2.0

    X_fb = np.empty((n_epochs, n_bands, n_ch, n_samples), dtype=np.float32)

    for bi, (lo, hi) in enumerate(config.bands):
        b, a = butter(config.order, [lo / nyq, hi / nyq], btype="bandpass")
        X_fb[:, bi] = filtfilt(b, a, X, axis=-1).astype(np.float32)

    return X_fb


# --- FBCNet nn.Module ---

@dataclass
class FBCNetConfig:
    n_channels: int = 64
    n_samples: int = 641
    n_classes: int = 2
    n_bands: int = 10
    m: int = 4          # spatial filters per band
    n_windows: int = 4  # temporal segments for multi-window variance
    dropout: float = 0.1


class _MultiWindowLogVarLayer(nn.Module):
    """Segment time into n_windows non-overlapping chunks, compute log-variance per chunk.

    Captures temporal ERD/ERS dynamics instead of collapsing all time info.
    """

    def __init__(self, n_windows: int = 4):
        super().__init__()
        self.n_windows = n_windows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features, time)
        B, F, T = x.shape
        w = self.n_windows
        # Trim samples to make evenly divisible
        usable = (T // w) * w
        x = x[:, :, :usable].reshape(B, F, w, usable // w)
        # Variance per window → log
        return torch.log(x.var(dim=-1) + 1e-6)  # (B, F, n_windows)


class FBCNet(nn.Module):
    """Filter Bank CSP Network with multi-window temporal variance."""

    def __init__(self, cfg: FBCNetConfig):
        super().__init__()
        self.cfg = cfg

        self.spatial_conv = nn.Conv2d(
            cfg.n_bands,
            cfg.n_bands * cfg.m,
            kernel_size=(cfg.n_channels, 1),
            groups=cfg.n_bands,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(cfg.n_bands * cfg.m)
        self.drop = nn.Dropout(cfg.dropout)

        self.logvar = _MultiWindowLogVarLayer(n_windows=cfg.n_windows)

        # Features: n_bands * m * n_windows
        n_features = cfg.n_bands * cfg.m * cfg.n_windows
        self.classifier = nn.Linear(n_features, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_bands, n_channels, n_samples)
        x = self.spatial_conv(x)   # (batch, n_bands*m, 1, n_samples)
        x = self.bn(x)
        x = x.squeeze(2)           # (batch, n_bands*m, n_samples)
        x = self.logvar(x)         # (batch, n_bands*m, n_windows)
        x = x.reshape(x.size(0), -1)  # (batch, n_bands*m*n_windows)
        x = self.drop(x)
        return self.classifier(x)

    @torch.no_grad()
    def apply_max_norm(self) -> None:
        """Compatibility stub."""
        pass
