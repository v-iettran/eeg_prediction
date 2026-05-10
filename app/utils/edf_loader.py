"""EDF file loading, validation, and caching for the Streamlit app."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import mne
import numpy as np
import streamlit as st
from scipy.signal import welch, butter, filtfilt


EXPECTED_N_CHANNELS = 64
EXPECTED_SFREQ = 160.0


def _file_hash(uploaded_file) -> str:
    """Compute a short hash of the uploaded file for caching."""
    uploaded_file.seek(0)
    h = hashlib.md5(uploaded_file.read()).hexdigest()[:12]
    uploaded_file.seek(0)
    return h


@st.cache_data(show_spinner="Loading EDF file...")
def load_edf(_uploaded_file, file_hash: str) -> dict:
    """Load an uploaded EDF file and return extracted arrays + metadata.

    Returns a dict with:
        data: (n_channels, n_samples) float64 array
        times: (n_samples,) array
        ch_names: list of channel names
        sfreq: sampling frequency
        duration: recording duration in seconds
        annotations: list of (onset, duration, description) tuples
        events: (n_events, 3) MNE events array
        event_id: dict mapping annotation names to codes
    """
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        _uploaded_file.seek(0)
        tmp.write(_uploaded_file.read())
        tmp_path = tmp.name

    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)

    try:
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="warn", verbose="ERROR")
    except Exception:
        pass

    data, times = raw.get_data(return_times=True)
    annotations = [
        (float(a["onset"]), float(a["duration"]), str(a["description"]))
        for a in raw.annotations
    ]

    events, full_event_id = mne.events_from_annotations(raw, verbose="ERROR")

    Path(tmp_path).unlink(missing_ok=True)

    return {
        "data": data,
        "times": times,
        "ch_names": raw.ch_names,
        "sfreq": float(raw.info["sfreq"]),
        "duration": float(raw.times[-1]),
        "annotations": annotations,
        "events": events,
        "event_id": full_event_id,
    }


def validate_edf(edf_data: dict) -> list[str]:
    """Return a list of validation warnings/errors."""
    issues = []
    n_ch = len(edf_data["ch_names"])
    if n_ch != EXPECTED_N_CHANNELS:
        issues.append(f"Expected {EXPECTED_N_CHANNELS} channels, found {n_ch}.")

    ann_descs = {a[2] for a in edf_data["annotations"]}
    if "T1" not in ann_descs and "T2" not in ann_descs:
        issues.append(
            "No T1/T2 motor imagery annotations found. "
            "Please upload a file from runs 4, 8, or 12."
        )
    return issues


def has_motor_imagery_events(edf_data: dict) -> bool:
    """Check if the file contains T1/T2 events."""
    ann_descs = {a[2] for a in edf_data["annotations"]}
    return "T1" in ann_descs or "T2" in ann_descs


@st.cache_data(show_spinner=False)
def get_filtered_signal(data: np.ndarray, sfreq: float, l_freq: float = 8.0, h_freq: float = 30.0) -> np.ndarray:
    """Bandpass filter signal data using IIR butterworth."""
    nyq = sfreq / 2.0
    lo = max(l_freq / nyq, 1e-5)
    hi = min(h_freq / nyq, 1.0 - 1e-5)
    b, a = butter(4, [lo, hi], btype="bandpass")
    return filtfilt(b, a, data, axis=-1).astype(np.float64)


@st.cache_data(show_spinner=False)
def compute_psd(signal_1d: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSD via Welch's method for a single channel."""
    nperseg = min(512, len(signal_1d))
    freqs, psd = welch(signal_1d, fs=sfreq, nperseg=nperseg)
    return freqs, psd


def get_event_times(edf_data: dict) -> list[dict]:
    """Extract event onset times with labels from annotations.

    Returns list of {onset: float, duration: float, label: str} dicts.
    """
    events = []
    for onset, duration, desc in edf_data["annotations"]:
        if desc in ("T0", "T1", "T2"):
            events.append({"onset": onset, "duration": duration, "label": desc})
    return events
