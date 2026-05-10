"""Enriched feature extraction for V2 classical models.

Computes 31 features per epoch from the 3-channel motor view (C3, Cz, C4):
  - 9 per-channel features × 3 channels = 27
  - 4 inter-channel features
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch, coherence
from scipy.stats import kurtosis

from src.features import MU_BAND, BETA_BAND

CHANNEL_NAMES = ("C3", "Cz", "C4")

_PER_CH_FEAT_NAMES = [
    "log_mu_power",
    "log_beta_power",
    "mu_rel_power",
    "beta_rel_power",
    "mu_beta_ratio",
    "variance",
    "kurtosis",
    "hjorth_mobility",
    "hjorth_complexity",
]

_INTER_CH_FEAT_NAMES = [
    "C3C4_mu_coherence",
    "C3C4_beta_coherence",
    "C3C4_mu_asymmetry",
    "C3C4_beta_asymmetry",
]


def feature_names_v2() -> list[str]:
    """Return 31 human-readable feature names in extraction order."""
    names: list[str] = []
    for ch in CHANNEL_NAMES:
        for feat in _PER_CH_FEAT_NAMES:
            names.append(f"{ch}_{feat}")
    names.extend(_INTER_CH_FEAT_NAMES)
    return names


def _band_power_welch(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    mask = (freqs >= fmin) & (freqs <= fmax)
    return psd[..., mask].mean(axis=-1)


def _hjorth_params(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Hjorth mobility and complexity along the last axis.

    x: (..., n_samples)
    Returns: (mobility, complexity) each shape (...)
    """
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)

    var_x = np.var(x, axis=-1)
    var_dx = np.var(dx, axis=-1)
    var_ddx = np.var(ddx, axis=-1)

    std_x = np.sqrt(np.maximum(var_x, 1e-20))
    std_dx = np.sqrt(np.maximum(var_dx, 1e-20))
    std_ddx = np.sqrt(np.maximum(var_ddx, 1e-20))

    mobility = std_dx / std_x
    mobility_dx = std_ddx / std_dx
    complexity = mobility_dx / np.maximum(mobility, 1e-20)

    return mobility, complexity


def extract_features_v2(X_3ch: np.ndarray, sfreq: float) -> np.ndarray:
    """Extract 31 enriched features per epoch.

    Parameters
    ----------
    X_3ch : ndarray, shape (n_epochs, 3, n_samples)
        Channels are [C3, Cz, C4].
    sfreq : float
        Sampling rate in Hz.

    Returns
    -------
    feats : ndarray, shape (n_epochs, 31), float64
    """
    if X_3ch.ndim != 3 or X_3ch.shape[1] != 3:
        raise ValueError(f"Expected (n_epochs, 3, n_samples), got {X_3ch.shape}")

    n_epochs, n_ch, n_samples = X_3ch.shape
    nperseg = min(256, n_samples)

    freqs_w, psd = welch(X_3ch, fs=sfreq, nperseg=nperseg, axis=-1)

    mu_power = _band_power_welch(psd, freqs_w, *MU_BAND)       # (n_epochs, 3)
    beta_power = _band_power_welch(psd, freqs_w, *BETA_BAND)   # (n_epochs, 3)
    total_power = _band_power_welch(psd, freqs_w, MU_BAND[0], BETA_BAND[1])

    log_mu = np.log(np.maximum(mu_power, 1e-20))
    log_beta = np.log(np.maximum(beta_power, 1e-20))

    total_safe = np.maximum(total_power, 1e-20)
    mu_rel = mu_power / total_safe
    beta_rel = beta_power / total_safe
    mu_beta_ratio = mu_power / np.maximum(beta_power, 1e-20)

    var_epoch = np.var(X_3ch, axis=-1)          # (n_epochs, 3)
    kurt_epoch = kurtosis(X_3ch, axis=-1, fisher=True)  # (n_epochs, 3)

    mobility, complexity = _hjorth_params(X_3ch)  # each (n_epochs, 3)

    per_ch = np.stack([
        log_mu, log_beta,
        mu_rel, beta_rel, mu_beta_ratio,
        var_epoch, kurt_epoch,
        mobility, complexity,
    ], axis=-1)  # (n_epochs, 3, 9)
    per_ch_flat = per_ch.reshape(n_epochs, n_ch * len(_PER_CH_FEAT_NAMES))  # (n_epochs, 27)

    # --- inter-channel features (C3=idx0, C4=idx2) ---
    c3_data = X_3ch[:, 0, :]  # (n_epochs, n_samples)
    c4_data = X_3ch[:, 2, :]

    mu_coh = np.empty(n_epochs, dtype=np.float64)
    beta_coh = np.empty(n_epochs, dtype=np.float64)
    for i in range(n_epochs):
        f_coh, cxy = coherence(c3_data[i], c4_data[i], fs=sfreq, nperseg=nperseg)
        mu_mask = (f_coh >= MU_BAND[0]) & (f_coh <= MU_BAND[1])
        beta_mask = (f_coh >= BETA_BAND[0]) & (f_coh <= BETA_BAND[1])
        mu_coh[i] = cxy[mu_mask].mean() if mu_mask.any() else 0.0
        beta_coh[i] = cxy[beta_mask].mean() if beta_mask.any() else 0.0

    mu_asym = log_mu[:, 2] - log_mu[:, 0]      # log(mu_C4) - log(mu_C3)
    beta_asym = log_beta[:, 2] - log_beta[:, 0]

    inter_ch = np.stack([mu_coh, beta_coh, mu_asym, beta_asym], axis=-1)  # (n_epochs, 4)

    feats = np.hstack([per_ch_flat, inter_ch]).astype(np.float64)  # (n_epochs, 31)
    assert feats.shape == (n_epochs, 31), feats.shape
    return feats
