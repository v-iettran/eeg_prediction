"""EEG-TCNet (Ingolfsson et al. 2020) — EEGNet frontend + Temporal Convolutional Network.

Architecture:
    Block 1 — Temporal conv (kernel 32, shorter than EEGNet's 64)
    Block 2 — Depthwise spatial conv across all 64 channels
    Block 3 — TCN with dilated causal convolutions [1, 2, 4, 8]
    Classifier — Global average pooling + Linear
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TCNetConfig:
    n_channels: int = 64
    n_samples: int = 641
    n_classes: int = 2
    F1: int = 8
    D: int = 2
    F2: int = 16
    temporal_kernel: int = 32
    pool1: int = 4
    tcn_kernel: int = 4
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.4
    max_norm: float = 1.0


class _CausalConv1d(nn.Module):
    """Conv1d with causal (left-only) padding."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class _TemporalBlock(nn.Module):
    """Two causal convolutions with residual connection."""

    def __init__(self, n_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = _CausalConv1d(n_channels, n_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.conv2 = _CausalConv1d(n_channels, n_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(n_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop(F.elu(self.bn1(self.conv1(x))))
        out = self.drop(F.elu(self.bn2(self.conv2(out))))
        return out + residual


class EEGTCNet(nn.Module):
    """EEG-TCNet: EEGNet blocks 1-2 → TCN → global avg pool → classifier."""

    def __init__(self, cfg: TCNetConfig):
        super().__init__()
        self.cfg = cfg

        # Block 1 — temporal conv
        self.conv_temporal = nn.Conv2d(
            1, cfg.F1,
            kernel_size=(1, cfg.temporal_kernel),
            padding=(0, cfg.temporal_kernel // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(cfg.F1)

        # Block 2 — depthwise spatial conv
        self.conv_spatial = nn.Conv2d(
            cfg.F1, cfg.F1 * cfg.D,
            kernel_size=(cfg.n_channels, 1),
            groups=cfg.F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(cfg.F1 * cfg.D)
        self.pool1 = nn.AvgPool2d((1, cfg.pool1))
        self.drop1 = nn.Dropout(cfg.dropout)

        # Block 3 — TCN
        self.tcn_blocks = nn.Sequential(*[
            _TemporalBlock(cfg.F2, cfg.tcn_kernel, d, cfg.dropout)
            for d in cfg.tcn_dilations
        ])

        self.classifier = nn.Linear(cfg.F2, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Block 1
        x = self.conv_temporal(x)
        if x.shape[-1] != self.cfg.n_samples:
            x = x[..., :self.cfg.n_samples]
        x = self.bn1(x)

        # Block 2
        x = self.conv_spatial(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Squeeze spatial dim: (B, F2, 1, T') → (B, F2, T')
        x = x.squeeze(2)

        # Block 3 — TCN
        x = self.tcn_blocks(x)

        # Global average pooling over time → (B, F2)
        x = x.mean(dim=-1)

        return self.classifier(x)

    @torch.no_grad()
    def apply_max_norm(self) -> None:
        """Clip spatial conv weight norms to cfg.max_norm (stabilises early training)."""
        w = self.conv_spatial.weight.data  # (out_ch, in_ch/groups, kH, kW)
        flat = w.view(w.size(0), -1)
        norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        desired = norms.clamp(max=self.cfg.max_norm)
        flat.mul_(desired / norms)
