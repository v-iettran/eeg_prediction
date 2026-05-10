"""Build notebook 07 — EEG-TCNet + FBCNet."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _nb_builder import build_notebook  # noqa: E402

cells = [
    ("markdown", "# 07 — EEG-TCNet & FBCNet\n\n"
     "Train two purpose-built motor imagery architectures:\n"
     "- **EEG-TCNet**: EEGNet frontend + Temporal Convolutional Network (dilated causal convolutions)\n"
     "- **FBCNet**: Filter Bank CSP Network (9-band filter bank + per-band spatial filters + log-variance)\n\n"
     "Uses V2 data. EEG-TCNet uses 8-30 Hz epochs; FBCNet uses 4-40 Hz wide-bandpass epochs."),
    ("code", "import subprocess, sys\n"
     "result = subprocess.run(\n"
     "    [sys.executable, '../scripts/run_07_tcnet_fbcnet.py'],\n"
     "    capture_output=False,\n"
     ")\n"
     "result.returncode"),
]

if __name__ == "__main__":
    out = ROOT / "notebooks" / "07_tcnet_fbcnet.ipynb"
    build_notebook(cells, out)
    print(f"Built {out}")
