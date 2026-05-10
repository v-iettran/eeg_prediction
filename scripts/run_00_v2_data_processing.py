"""V2 data processing — re-epoch with relaxed PTP threshold (300 µV).

Produces:
    data/processed/v2_epochs_w0.0-4.0.npz
    data/processed/v2_epochs_w0.5-2.5.npz
    data/processed/v2_data_manifest.json

Uses the same subject split as v1 (reads existing splits.json).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
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
    euclidean_align,
    PTP_REJECT_UV,
    EXPECTED_SFREQ,
    MOTOR_CHANNELS,
)
from src.splits import load_split, split_indices_by_subject  # noqa: E402


PROCESSED_DIR = ROOT / "data" / "processed"
WINDOWS = [(0.0, 4.0), (0.5, 2.5)]


def _process_subject(subject: int, raw_root: Path):
    raw = load_subject_raw(subject, runs=IMAGERY_RUNS, raw_root=raw_root)
    sfreq = float(raw.info["sfreq"])
    if abs(sfreq - EXPECTED_SFREQ) > 0.5:
        return None, None, sfreq
    preprocess_raw(raw)
    events, event_id = events_from_raw(raw)
    return raw, (events, event_id), sfreq


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_root = ROOT / "data" / "raw"

    print("=" * 70)
    print("V2 Data Processing — Relaxed PTP Rejection")
    print("=" * 70)
    print(f"PTP threshold : {PTP_REJECT_UV * 1e6:.0f} µV")
    print(f"Raw root      : {raw_root}")
    print(f"Processed dir : {PROCESSED_DIR}")

    split = load_split(PROCESSED_DIR / "splits.json")
    pool = sorted(set(split["train"]) | set(split["val"]) | set(split["test"]))
    print(f"\nUsing existing split: train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])}")

    # Load v1 manifest for comparison
    v1_manifest_path = PROCESSED_DIR / "data_manifest.json"
    v1_manifest = None
    if v1_manifest_path.exists():
        with open(v1_manifest_path) as f:
            v1_manifest = json.load(f)

    print("\nLoading + filtering each subject...")
    cached: dict[int, tuple[mne.io.Raw, np.ndarray, dict]] = {}
    skipped: list[tuple[int, str]] = []
    t0 = time.perf_counter()
    for s in pool:
        try:
            raw, evs_eid, sfreq = _process_subject(s, raw_root)
        except Exception as e:
            print(f"  S{s:03d}: FAILED ({e!r}) — skipping")
            skipped.append((s, repr(e)))
            continue
        if raw is None:
            print(f"  S{s:03d}: sampling rate {sfreq} ≠ {EXPECTED_SFREQ} — skipping")
            skipped.append((s, f"non-160Hz: {sfreq}"))
            continue
        events, event_id = evs_eid
        cached[s] = (raw, events, event_id)
        if len(cached) % 20 == 0:
            print(f"  ... {len(cached)} subjects loaded in {time.perf_counter() - t0:.1f}s")
    print(f"Loaded {len(cached)} subjects in {time.perf_counter() - t0:.1f}s; skipped {len(skipped)}")

    manifest_per_window = {}
    rejection_threshold_used = float(PTP_REJECT_UV)

    for tmin, tmax in WINDOWS:
        win_label = f"w{tmin:.1f}-{tmax:.1f}"
        print(f"\n--- Epoching window {win_label} ({tmin}–{tmax} s) ---")
        all_X64: list[np.ndarray] = []
        all_X3: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_subj: list[np.ndarray] = []
        ch_names_64: list[str] | None = None

        per_subject_counts: list[dict] = []
        n_total_events = 0
        n_kept = 0

        for s in sorted(cached):
            raw, events, event_id = cached[s]
            n_total_events += len(events)
            try:
                ep = make_epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                                 reject_uv=rejection_threshold_used)
            except Exception as e:
                print(f"  S{s:03d}: epoch creation failed ({e!r})")
                per_subject_counts.append({"subject": s, "n_events": int(len(events)), "n_kept": 0, "error": repr(e)})
                continue
            data = ep.get_data(copy=False)
            if data.shape[0] == 0:
                per_subject_counts.append({"subject": s, "n_events": int(len(events)), "n_kept": 0})
                continue
            y = epochs_to_label_array(ep)
            if ch_names_64 is None:
                ch_names_64 = list(ep.info["ch_names"])

            # Euclidean Alignment per subject (whitens covariance to identity)
            data_ea = euclidean_align(data)

            x3 = select_motor_channels(data_ea, ch_names_64)
            subj = np.full(data_ea.shape[0], s, dtype=np.int32)
            all_X64.append(data_ea)
            all_X3.append(x3)
            all_y.append(y)
            all_subj.append(subj)
            n_kept += data_ea.shape[0]
            per_subject_counts.append({"subject": s, "n_events": int(len(events)), "n_kept": int(data_ea.shape[0])})

        X_64ch = np.concatenate(all_X64, axis=0)
        X_3ch = np.concatenate(all_X3, axis=0)
        y = np.concatenate(all_y, axis=0).astype(np.int64)
        subject_ids = np.concatenate(all_subj, axis=0).astype(np.int32)

        assert ch_names_64 is not None and len(ch_names_64) == 64
        assert X_3ch.shape[1] == 3
        assert X_64ch.shape[0] == X_3ch.shape[0] == y.shape[0] == subject_ids.shape[0]
        assert not np.isnan(X_64ch).any(), "NaN in X_64ch"
        assert not np.isinf(X_64ch).any(), "Inf in X_64ch"
        assert not np.isnan(X_3ch).any(), "NaN in X_3ch"

        n_left = int((y == 0).sum())
        n_right = int((y == 1).sum())
        balance_ratio = min(n_left, n_right) / max(n_left, n_right)
        print(f"  Total epochs kept: {n_kept} / {n_total_events} events ({100 * n_kept / n_total_events:.1f}%)")
        print(f"  Class balance    : T1={n_left} | T2={n_right} | ratio={balance_ratio:.3f}")

        # V1 comparison
        if v1_manifest and win_label in v1_manifest.get("windows", {}):
            v1_kept = v1_manifest["windows"][win_label]["n_kept"]
            print(f"  V1 comparison    : {v1_kept} → {n_kept} epochs (+{n_kept - v1_kept}, +{100*(n_kept - v1_kept)/v1_kept:.1f}%)")

        assert balance_ratio > 0.85, f"class imbalance too large: {n_left} vs {n_right}"

        train_set = set(split["train"]); val_set = set(split["val"]); test_set = set(split["test"])
        n_train = int(np.isin(subject_ids, list(train_set)).sum())
        n_val = int(np.isin(subject_ids, list(val_set)).sum())
        n_test = int(np.isin(subject_ids, list(test_set)).sum())
        print(f"  Per-split counts : train {n_train} | val {n_val} | test {n_test}")

        out_path = PROCESSED_DIR / f"v2_epochs_{win_label}.npz"
        print(f"  Saving → {out_path}")
        np.savez_compressed(
            out_path,
            X_3ch=X_3ch,
            X_64ch=X_64ch,
            y=y,
            subject_ids=subject_ids,
            ch_names_64=np.array(ch_names_64),
            sfreq=np.float32(EXPECTED_SFREQ),
            tmin=np.float32(tmin),
            tmax=np.float32(tmax),
        )

        manifest_per_window[win_label] = {
            "window": [tmin, tmax],
            "n_samples_per_epoch": int(X_64ch.shape[2]),
            "n_total_events": n_total_events,
            "n_kept": n_kept,
            "drop_rate": float(1 - n_kept / max(n_total_events, 1)),
            "class_counts": {"T1_left": n_left, "T2_right": n_right},
            "split_counts": {"train": n_train, "val": n_val, "test": n_test},
            "per_subject_counts": per_subject_counts,
        }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_version": "v2",
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "subjects_used": pool,
        "split": split,
        "preprocessing": {
            "bandpass_hz": [8, 30],
            "filter_method": "iir",
            "reference": "average",
            "ptp_reject_uv": rejection_threshold_used * 1e6,
            "sfreq_hz": EXPECTED_SFREQ,
            "euclidean_alignment": True,
        },
        "channels": {
            "rf_view": list(MOTOR_CHANNELS),
            "n_channels_64ch_view": 64,
        },
        "windows": manifest_per_window,
    }
    manifest_path = PROCESSED_DIR / "v2_data_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {manifest_path}")
    print("V2 data processing complete.")


if __name__ == "__main__":
    main()
