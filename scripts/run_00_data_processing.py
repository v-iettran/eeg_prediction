"""Notebook 00 logic — runnable as a plain script for fast iteration.

Produces:
    data/processed/splits.json
    data/processed/epochs_w0.0-4.0.npz
    data/processed/epochs_w0.5-2.5.npz
    data/processed/data_manifest.json

Re-runnable and deterministic.
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
    PTP_REJECT_UV,
    EXPECTED_SFREQ,
    MOTOR_CHANNELS,
)
from src.splits import (  # noqa: E402
    build_subject_split,
    save_split,
    KNOWN_BAD_SUBJECTS,
)


PROCESSED_DIR = ROOT / "data" / "processed"
WINDOWS = [(0.0, 4.0), (0.5, 2.5)]


def _process_subject(subject: int, raw_root: Path):
    """Load + preprocess one subject, return (events, raw, sfreq)."""
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
    print("Notebook 00 — Data Processing")
    print("=" * 70)
    print(f"Raw root      : {raw_root}")
    print(f"Processed dir : {PROCESSED_DIR}")
    print(f"Windows       : {WINDOWS}")
    print(f"Drop subjects : {KNOWN_BAD_SUBJECTS}")

    # --- subject split -----------------------------------------------------
    split = build_subject_split()
    save_split(split, PROCESSED_DIR / "splits.json")
    pool = sorted(set(split["train"]) | set(split["val"]) | set(split["test"]))
    print(f"\nSplits → train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])}")
    print(f"Total kept    : {len(pool)} subjects (109 - {len(split['dropped'])} dropped)")

    # --- per-subject preprocessing (cache once, then epoch twice) ---------
    print("\nLoading + filtering each subject (8–30 Hz, average reference)...")
    cached: dict[int, tuple[mne.io.Raw, np.ndarray, dict]] = {}
    skipped: list[tuple[int, str]] = []
    t0 = time.perf_counter()
    for s in pool:
        try:
            raw, evs_eid, sfreq = _process_subject(s, raw_root)
        except Exception as e:  # pragma: no cover — defensive
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

    # --- epoching per window ---------------------------------------------
    manifest_per_window = {}
    rejection_threshold_used = float(PTP_REJECT_UV)
    expected_n_samples_per_window: dict[str, int] = {}

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
                ep = make_epochs(
                    raw,
                    events,
                    event_id,
                    tmin=tmin,
                    tmax=tmax,
                    reject_uv=rejection_threshold_used,
                )
            except Exception as e:
                print(f"  S{s:03d}: epoch creation failed ({e!r})")
                per_subject_counts.append({"subject": s, "n_events": int(len(events)), "n_kept": 0, "error": repr(e)})
                continue
            data = ep.get_data(copy=False)  # (n_kept, 64, n_samples)
            if data.shape[0] == 0:
                per_subject_counts.append({"subject": s, "n_events": int(len(events)), "n_kept": 0})
                continue
            y = epochs_to_label_array(ep)
            if ch_names_64 is None:
                ch_names_64 = list(ep.info["ch_names"])
            else:
                # Sanity: channel order is identical across subjects (it should be —
                # same montage). We assert below before saving.
                pass
            x3 = select_motor_channels(data, ch_names_64)
            subj = np.full(data.shape[0], s, dtype=np.int32)
            all_X64.append(data.astype(np.float32))
            all_X3.append(x3.astype(np.float32))
            all_y.append(y)
            all_subj.append(subj)
            n_kept += data.shape[0]
            per_subject_counts.append({"subject": s, "n_events": int(len(events)), "n_kept": int(data.shape[0])})

        X_64ch = np.concatenate(all_X64, axis=0)
        X_3ch = np.concatenate(all_X3, axis=0)
        y = np.concatenate(all_y, axis=0).astype(np.int64)
        subject_ids = np.concatenate(all_subj, axis=0).astype(np.int32)

        # Sanity checks.
        assert ch_names_64 is not None and len(ch_names_64) == 64, ch_names_64
        assert X_3ch.shape[1] == 3, X_3ch.shape
        assert X_64ch.shape[0] == X_3ch.shape[0] == y.shape[0] == subject_ids.shape[0]
        assert not np.isnan(X_64ch).any(), "NaN in X_64ch"
        assert not np.isinf(X_64ch).any(), "Inf in X_64ch"
        assert not np.isnan(X_3ch).any(), "NaN in X_3ch"

        # Class balance check.
        n_left = int((y == 0).sum())
        n_right = int((y == 1).sum())
        print(f"  Total epochs kept: {n_kept} / {n_total_events} events")
        print(f"  Class balance    : T1 (left) {n_left} | T2 (right) {n_right}")
        balance_ratio = min(n_left, n_right) / max(n_left, n_right)
        print(f"  Balance ratio    : {balance_ratio:.3f}")
        assert balance_ratio > 0.85, f"class imbalance too large: {n_left} vs {n_right}"

        # Per-split counts.
        train_set = set(split["train"]); val_set = set(split["val"]); test_set = set(split["test"])
        n_train = int(np.isin(subject_ids, list(train_set)).sum())
        n_val = int(np.isin(subject_ids, list(val_set)).sum())
        n_test = int(np.isin(subject_ids, list(test_set)).sum())
        print(f"  Per-split counts : train {n_train} | val {n_val} | test {n_test}")

        # Channel order assertion for X_3ch.
        idx_C3 = ch_names_64.index("C3")
        idx_Cz = ch_names_64.index("Cz")
        idx_C4 = ch_names_64.index("C4")
        # x3[:, 0, :] should equal X_64ch[:, idx_C3, :], etc.
        np.testing.assert_allclose(X_3ch[:, 0], X_64ch[:, idx_C3], rtol=0, atol=0)
        np.testing.assert_allclose(X_3ch[:, 1], X_64ch[:, idx_Cz], rtol=0, atol=0)
        np.testing.assert_allclose(X_3ch[:, 2], X_64ch[:, idx_C4], rtol=0, atol=0)

        out_path = PROCESSED_DIR / f"epochs_{win_label}.npz"
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
        expected_n_samples_per_window[win_label] = int(X_64ch.shape[2])

    # --- manifest --------------------------------------------------------
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "subjects_total": 109,
        "subjects_dropped": split["dropped"],
        "subjects_skipped_at_load": [{"subject": s, "reason": r} for s, r in skipped],
        "subjects_used": pool,
        "split": split,
        "preprocessing": {
            "bandpass_hz": [8, 30],
            "filter_method": "iir",
            "reference": "average",
            "ptp_reject_uv": rejection_threshold_used * 1e6,
            "sfreq_hz": EXPECTED_SFREQ,
        },
        "channels": {
            "rf_view": list(MOTOR_CHANNELS),
            "n_channels_64ch_view": 64,
        },
        "windows": manifest_per_window,
        "expected_n_samples_per_window": expected_n_samples_per_window,
    }
    with open(PROCESSED_DIR / "data_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nWrote", PROCESSED_DIR / "data_manifest.json")
    print("Done.")


if __name__ == "__main__":
    main()
