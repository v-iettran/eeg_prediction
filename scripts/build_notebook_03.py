"""Build notebooks/03_eegnet.ipynb."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._nb_builder import build_notebook  # noqa: E402


CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# 03 — EEGNet on raw 64-channel epochs

End-to-end CNN (Lawhern 2018) — no hand-crafted features. Block 2 learns
CSP-like spatial filters across all 64 channels per temporal-frequency band.
See `PIPELINE_PLAN.md` §7 for the architecture.

GPU is recommended; the script auto-selects CUDA → MPS → CPU.
"""),
    ("code", """\
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
print('CUDA:', torch.cuda.is_available(), '| MPS:', torch.backends.mps.is_available())

from scripts.run_03_eegnet import main as run_eegnet
"""),
    ("markdown", "## Train + evaluate (both windows, then pick winner)"),
    ("code", "run_eegnet()"),
    ("markdown", "## Inspect saved results"),
    ("code", """\
import json
result = json.loads((ROOT / 'reports' / 'eegnet_results.json').read_text())
print('Architecture:', result['model_config']['architecture'])
print('# params    :', result['model_config']['n_params'])
print('Device      :', result['model_config']['training']['device'])
print('Winning window:', result['winning_window'])
print('Test metrics:')
for k, v in result['test_metrics'].items():
    print(f'  {k}: {v}')
print('Per-window val:')
for w in result['windows_evaluated']:
    print(f"  window {w['window']}: val macroF1 {w['val_macro_f1']:.4f}  acc {w['val_accuracy']:.4f}  ({w['training_time_sec']:.1f}s)")
"""),
    ("markdown", "## Training curves and confusion matrix"),
    ("code", """\
from IPython.display import Image, display
display(Image(str(ROOT / 'reports' / 'figures' / 'eegnet_training_curves.png')))
display(Image(str(ROOT / 'reports' / 'figures' / 'eegnet_confusion.png')))
"""),
]


def main() -> None:
    NOTEBOOK_DIR = ROOT / "notebooks"
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    build_notebook(CELLS, NOTEBOOK_DIR / "03_eegnet.ipynb")
    print('Wrote', NOTEBOOK_DIR / "03_eegnet.ipynb")


if __name__ == "__main__":
    main()
