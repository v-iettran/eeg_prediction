"""Preprocessing & epoching utilities.

The pipeline is:
    raw EDF → standardised channel names + montage  (in :mod:`src.data_io`)
            → 8–30 Hz IIR bandpass + average reference  (here)
            → epochs at T1/T2 events with peak-to-peak rejection  (here)

Two channel "views" are produced:
    * ``X_3ch``  — only [C3, Cz, C4]   (used by RandomForest)
    * ``X_64ch`` — all 64 EEG channels (used by CSP+LDA and EEGNet)
"""
from __future__ import annotations

import numpy as np
import mne

# 3-channel motor view used by the RF baseline.
MOTOR_CHANNELS = ("C3", "Cz", "C4")

# 8–30 Hz captures both mu (8–13) and beta (13–30). IIR is fast and fine.
BANDPASS_LOW = 8.0
BANDPASS_HIGH = 30.0

# Peak-to-peak amplitude threshold for gross-artifact rejection.
PTP_REJECT_UV = 300e-6

# Expected sampling frequency for eegmmidb after dropping the bad subjects.
EXPECTED_SFREQ = 160.0


def preprocess_raw(
    raw: mne.io.Raw,
    l_freq: float = BANDPASS_LOW,
    h_freq: float = BANDPASS_HIGH,
    verbose: str | bool = "ERROR",
) -> mne.io.Raw:
    """In-place IIR bandpass + average reference. Returns the raw."""
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        verbose=verbose,
    )
    raw.set_eeg_reference("average", projection=False, verbose=verbose)
    return raw


def make_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict,
    tmin: float,
    tmax: float,
    reject_uv: float = PTP_REJECT_UV,
    verbose: str | bool = "ERROR",
) -> mne.Epochs:
    """Cut epochs around T1/T2 events with simple peak-to-peak rejection.

    ``baseline=None`` because the bandpass already removed slow drifts.
    """
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        reject=dict(eeg=reject_uv),
        verbose=verbose,
    )
    return epochs


def euclidean_align(X: np.ndarray) -> np.ndarray:
    """Euclidean Alignment: whiten epochs by the subject's mean covariance.

    Removes inter-subject covariance mismatch so spatial filters learned
    on one subject transfer to another.  Zero extra parameters.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_samples)
        All epochs from a single subject.

    Returns
    -------
    X_aligned : ndarray, same shape, float32
    """
    n_epochs, n_ch, n_t = X.shape
    # Mean covariance across epochs: R = (1/n) Σ (X_i X_i^T) / n_t
    X64 = X.astype(np.float64)
    R = np.zeros((n_ch, n_ch), dtype=np.float64)
    for i in range(n_epochs):
        R += X64[i] @ X64[i].T
    R /= n_epochs * n_t

    # R^{-1/2} via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-10)
    R_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))[np.newaxis, :]) @ eigvecs.T

    # Transform all epochs
    X_aligned = np.einsum("ij,ejt->eit", R_inv_sqrt, X64)
    return X_aligned.astype(np.float32)


def select_motor_channels(epochs_data: np.ndarray, ch_names: list[str]) -> np.ndarray:
    """Extract the [C3, Cz, C4] view from a (n_epochs, n_channels, n_samples) array.

    Asserts the channels exist and reorders to exactly ``MOTOR_CHANNELS``.
    """
    name_to_idx = {n: i for i, n in enumerate(ch_names)}
    missing = [c for c in MOTOR_CHANNELS if c not in name_to_idx]
    if missing:
        raise ValueError(
            f"Missing motor channels {missing}; available: {ch_names[:8]}..."
        )
    idx = [name_to_idx[c] for c in MOTOR_CHANNELS]
    return epochs_data[:, idx, :]


def epochs_to_label_array(epochs: mne.Epochs) -> np.ndarray:
    """Map MNE event codes back to {0, 1} labels: T1 → 0 (left), T2 → 1 (right)."""
    codes = epochs.events[:, 2]
    # event_id is {'T1': 2, 'T2': 3}
    y = np.where(codes == 2, 0, 1).astype(np.int64)
    return y
