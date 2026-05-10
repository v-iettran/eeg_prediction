"""EDF loading + annotation parsing for the PhysioNet eegmmidb dataset.

Files are expected at ``<raw_root>/S{NNN}/S{NNN}R{RR}.edf``. See
PIPELINE_PLAN.md sec. 4.1 for the full data layout.

Only runs 4, 8, 12 (imagined left/right fist) are used by this pipeline.
"""
from __future__ import annotations

from pathlib import Path

import mne


IMAGERY_RUNS = (4, 8, 12)
EVENT_ID = {"T1": 2, "T2": 3}  # 0-indexed: T1 = left fist imagined, T2 = right fist imagined
DEFAULT_RAW_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


def subject_run_path(raw_root: str | Path, subject: int, run: int) -> Path:
    """Return absolute path to ``S{subject:03d}R{run:02d}.edf``."""
    return Path(raw_root) / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf"


def load_subject_raw(
    subject: int,
    runs: tuple[int, ...] = IMAGERY_RUNS,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    verbose: str | bool = "ERROR",
) -> mne.io.Raw:
    """Load and concatenate the requested runs for a single subject.

    Channel names are standardised (``mne.datasets.eegbci.standardize``) and a
    standard_1005 montage is applied. No filtering or referencing is done here
    — that lives in :mod:`src.preprocessing`.
    """
    raws = []
    for run in runs:
        path = subject_run_path(raw_root, subject, run)
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose=verbose)
        raws.append(raw)
    raw = mne.concatenate_raws(raws)
    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="warn", verbose=verbose)
    return raw


def events_from_raw(raw: mne.io.Raw, verbose: str | bool = "ERROR"):
    """Extract MNE-style events array + (filtered) event_id for T1/T2 only.

    The default ``mne.events_from_annotations`` mapping for eegmmidb is
    ``{T0: 1, T1: 2, T2: 3}`` but we only keep T1/T2 here since T0 (rest)
    is not part of the binary classification task.
    """
    events, _ = mne.events_from_annotations(raw, verbose=verbose)
    keep_codes = set(EVENT_ID.values())
    mask = [code in keep_codes for code in events[:, 2]]
    events = events[mask]
    return events, dict(EVENT_ID)
