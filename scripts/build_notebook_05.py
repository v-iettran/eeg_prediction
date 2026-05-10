"""Build notebook 05 — CSP+LDA v2 + EEGNet v2."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _nb_builder import build_notebook  # noqa: E402

cells = [
    ("markdown", "# 05 — CSP+LDA v2 & EEGNet v2\n\n"
     "Retrain CSP+LDA and EEGNet on V2 data (relaxed PTP rejection → more epochs).\n"
     "Same architecture and hyperparameters as V1 — only the data changes."),
    ("code", "import subprocess, sys\n"
     "result = subprocess.run(\n"
     "    [sys.executable, '../scripts/run_05_v2_csp_eegnet.py'],\n"
     "    capture_output=False,\n"
     ")\n"
     "result.returncode"),
]

if __name__ == "__main__":
    out = ROOT / "notebooks" / "05_v2_csp_eegnet.ipynb"
    build_notebook(cells, out)
    print(f"Built {out}")
