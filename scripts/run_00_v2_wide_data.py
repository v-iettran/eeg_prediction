"""Generate wide-bandpass (4-40 Hz) V2 epochs for FBCNet.

Produces:
    data/processed/v2_epochs_w0.0-4.0_wide.npz

Uses the same subject split, same rejection threshold (300 µV), but
wider bandpass (4-40 Hz instead of 8-30 Hz) to support FBCNet's
9-band filter bank which spans 4-40 Hz.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import mne

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_io import load_subject_raw, events_from_raw, IMAGERY_RUNS  # noqa: E402
from src.preprocessing import (  # noqa: E402
    preprocess_raw,
    make_epochs,
    select_motor_channels,
    epochs_to_label_array,
    PTP_REJECT_UV,
    EXPECTED_SFREQ,
)
from src.splits import load_split  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
WIDE_L_FREQ = 4.0
WIDE_H_FREQ = 40.0
TMIN, TMAX = 0.0, 4.0
WIN_LABEL = "w0.0-4.0"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_root = ROOT / "data" / "raw"

    print("=" * 70)
    print("V2 Wide-Bandpass Data Processing (4–40 Hz) for FBCNet")
    print("=" * 70)
    print(f"Bandpass       : {WIDE_L_FREQ}–{WIDE_H_FREQ} Hz")
    print(f"PTP threshold  : {PTP_REJECT_UV * 1e6:.0f} µV")

    split = load_split(PROCESSED_DIR / "splits.json")
    pool = sorted(set(split["train"]) | set(split["val"]) | set(split["test"]))
    print(f"Subjects: {len(pool)}")

    print("\nLoading + filtering each subject (4–40 Hz)...")
    cached: dict[int, tuple] = {}
    t0 = time.perf_counter()
    for s in pool:
        try:
            raw = load_subject_raw(s, runs=IMAGERY_RUNS, raw_root=raw_root)
            sfreq = float(raw.info["sfreq"])
            if abs(sfreq - EXPECTED_SFREQ) > 0.5:
                continue
            preprocess_raw(raw, l_freq=WIDE_L_FREQ, h_freq=WIDE_H_FREQ)
            events, event_id = events_from_raw(raw)
            cached[s] = (raw, events, event_id)
        except Exception as e:
            print(f"  S{s:03d}: FAILED ({e!r})")
            continue
        if len(cached) % 20 == 0:
            print(f"  ... {len(cached)} subjects in {time.perf_counter() - t0:.1f}s")
    print(f"Loaded {len(cached)} subjects in {time.perf_counter() - t0:.1f}s")

    print(f"\n--- Epoching window {WIN_LABEL} ---")
    all_X64: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_subj: list[np.ndarray] = []
    ch_names_64: list[str] | None = None
    n_total = 0
    n_kept = 0

    rejection_threshold_used = float(PTP_REJECT_UV)

    for s in sorted(cached):
        raw, events, event_id = cached[s]
        n_total += len(events)
        try:
            ep = make_epochs(raw, events, event_id, tmin=TMIN, tmax=TMAX,
                             reject_uv=rejection_threshold_used)
        except Exception:
            continue
        data = ep.get_data(copy=False)
        if data.shape[0] == 0:
            continue
        y = epochs_to_label_array(ep)
        if ch_names_64 is None:
            ch_names_64 = list(ep.info["ch_names"])
        subj = np.full(data.shape[0], s, dtype=np.int32)
        all_X64.append(data.astype(np.float32))
        all_y.append(y)
        all_subj.append(subj)
        n_kept += data.shape[0]

    X_64ch = np.concatenate(all_X64, axis=0)
    y = np.concatenate(all_y, axis=0).astype(np.int64)
    subject_ids = np.concatenate(all_subj, axis=0).astype(np.int32)

    assert ch_names_64 is not None and len(ch_names_64) == 64
    assert not np.isnan(X_64ch).any()
    assert not np.isinf(X_64ch).any()

    n_left = int((y == 0).sum())
    n_right = int((y == 1).sum())
    print(f"  Epochs kept: {n_kept} / {n_total} ({100 * n_kept / n_total:.1f}%)")
    print(f"  Classes: T1={n_left} T2={n_right}")

    out_path = PROCESSED_DIR / f"v2_epochs_{WIN_LABEL}_wide.npz"
    print(f"  Saving → {out_path}")
    np.savez_compressed(
        out_path,
        X_64ch=X_64ch,
        y=y,
        subject_ids=subject_ids,
        ch_names_64=np.array(ch_names_64),
        sfreq=np.float32(EXPECTED_SFREQ),
        tmin=np.float32(TMIN),
        tmax=np.float32(TMAX),
        bandpass_hz=np.array([WIDE_L_FREQ, WIDE_H_FREQ], dtype=np.float32),
    )
    print(f"  Shape: {X_64ch.shape}")
    print("Wide-bandpass data processing complete.")


if __name__ == "__main__":
    main()
