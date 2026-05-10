"""Feature extraction utilities.

Currently exposes :func:`bandpower_features` for the Random Forest notebook.
CSP itself comes from :class:`mne.decoding.CSP` so it is wired up directly
inside the CSP notebook rather than re-implemented here.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


# Standard motor imagery sub-bands.
MU_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)


def _band_power(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Average PSD power between fmin and fmax (inclusive)."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        raise ValueError(f"No frequencies in [{fmin}, {fmax}] Hz")
    # Trapezoidal integration is more accurate than mean(); use simple mean
    # for robustness across welch settings — both are monotonic in power.
    return psd[..., mask].mean(axis=-1)


def bandpower_features(
    X: np.ndarray,
    sfreq: float,
    bands=(MU_BAND, BETA_BAND),
    nperseg: int | None = None,
) -> np.ndarray:
    """Compute log-band-power features for each (epoch, channel, band).

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_samples)
    sfreq : float
        Sampling rate in Hz.
    bands : iterable of (fmin, fmax) tuples
    nperseg : int, optional
        Welch window length. Defaults to min(256, n_samples).

    Returns
    -------
    feats : ndarray, shape (n_epochs, n_channels * len(bands))
        Log-band-power. Per-epoch order is ``[ch0_band0, ch0_band1, ch1_band0, ...]``.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected (n_epochs, n_channels, n_samples), got {X.shape}")
    n_epochs, n_channels, n_samples = X.shape
    if nperseg is None:
        nperseg = min(256, n_samples)

    # welch over the last axis returns (epochs, channels, n_freqs)
    freqs, psd = welch(X, fs=sfreq, nperseg=nperseg, axis=-1)

    band_powers = []
    for fmin, fmax in bands:
        bp = _band_power(psd, freqs, fmin, fmax)  # (n_epochs, n_channels)
        band_powers.append(bp)

    # Stack to (n_epochs, n_channels, n_bands) then flatten last two dims so
    # within an epoch we get [ch0_band0, ch0_band1, ch1_band0, ...].
    stacked = np.stack(band_powers, axis=-1)  # (n_epochs, n_channels, n_bands)
    feats = stacked.reshape(n_epochs, n_channels * len(bands))

    # log-transform stabilises heavy tails. Floor for numerical safety.
    feats = np.log(np.maximum(feats, 1e-20))
    return feats


def feature_names(channel_names, bands=(MU_BAND, BETA_BAND)) -> list[str]:
    """Return human-readable feature names matching :func:`bandpower_features` order."""
    band_labels = [f"{int(fmin)}-{int(fmax)}Hz" for fmin, fmax in bands]
    names: list[str] = []
    for ch in channel_names:
        for b in band_labels:
            names.append(f"{ch}_{b}")
    return names
