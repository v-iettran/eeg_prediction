"""Builds notebooks/*.ipynb from the runnable scripts under scripts/.

Each notebook is a thin orchestrator: it imports from src/ and calls into
the same logic the scripts use. Generated notebooks are committed to the
repo so users can open them directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._nb_builder import build_notebook  # noqa: E402


NOTEBOOK_DIR = ROOT / "notebooks"


# ---------------------------------------------------------------------------
# 00 — Data Processing
# ---------------------------------------------------------------------------
NB00_CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# 00 — Data Processing

Run this notebook **first**. Every other notebook in this project loads from
`data/processed/`, so re-run only when raw data, preprocessing, or splits change.

This notebook produces:
- `data/processed/splits.json` — subject-level train/val/test split (seed=42).
- `data/processed/epochs_w0.0-4.0.npz` and `epochs_w0.5-2.5.npz` — epoch tensors.
- `data/processed/data_manifest.json` — full audit trail.

See `PIPELINE_PLAN.md` §4 for the design and rationale.
"""),
    ("code", """\
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print('Project root:', ROOT)
"""),
    ("markdown", "## 1. Build the deterministic subject split"),
    ("code", """\
from src.splits import build_subject_split, save_split, KNOWN_BAD_SUBJECTS

split = build_subject_split()
save_split(split, ROOT / 'data' / 'processed' / 'splits.json')

print('Dropped (known-bad):', split['dropped'])
print(f"Train: {len(split['train'])}  Val: {len(split['val'])}  Test: {len(split['test'])}")
print('First few train subjects:', split['train'][:10])
"""),
    ("markdown", """\
## 2. Run preprocessing + epoching for both windows

Implemented in `scripts/run_00_data_processing.py` so the same pipeline is
re-runnable from a terminal. We invoke it here for visibility — re-running
is idempotent.
"""),
    ("code", """\
from scripts.run_00_data_processing import main as run_data_processing
run_data_processing()
"""),
    ("markdown", "## 3. Quick verification: load each window file and inspect shapes"),
    ("code", """\
import numpy as np

for win in ['w0.0-4.0', 'w0.5-2.5']:
    arr = np.load(ROOT / 'data' / 'processed' / f'epochs_{win}.npz', allow_pickle=False)
    print(f"{win}: X_3ch={arr['X_3ch'].shape} | X_64ch={arr['X_64ch'].shape} | y={arr['y'].shape} | sfreq={arr['sfreq']}")
    print(f"   class balance: T1={(arr['y']==0).sum()} T2={(arr['y']==1).sum()}")
    print(f"   3-ch order asserted = {list(arr['ch_names_64'][[arr['ch_names_64'].tolist().index(c) for c in ['C3','Cz','C4']]])}")
"""),
    ("markdown", """\
## 4. Sanity checks against the plan

- Sampling rate must be 160 Hz.
- No NaN/Inf.
- Class balance within ±15%.
- 3-channel view ordered exactly as `[C3, Cz, C4]`.
- Manifest written.
"""),
    ("code", """\
import json
manifest = json.loads((ROOT / 'data' / 'processed' / 'data_manifest.json').read_text())
print('MNE version :', manifest['mne_version'])
print('Subjects used:', len(manifest['subjects_used']))
for win, info in manifest['windows'].items():
    cb = info['class_counts']
    print(f"  {win}: kept {info['n_kept']}/{info['n_total_events']} epochs ({info['drop_rate']*100:.1f}% drop) "
          f"| T1={cb['T1_left']} T2={cb['T2_right']} | "
          f"train={info['split_counts']['train']} val={info['split_counts']['val']} test={info['split_counts']['test']}")
"""),
    ("markdown", "All artifacts written. Continue with notebooks 01, 02, 03 in any order."),
]


def main() -> None:
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    build_notebook(NB00_CELLS, NOTEBOOK_DIR / "00_data_processing.ipynb")
    print('Wrote', NOTEBOOK_DIR / "00_data_processing.ipynb")


if __name__ == "__main__":
    main()
