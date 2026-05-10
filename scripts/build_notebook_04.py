"""Build notebook 04 — Classical Ensemble (V2 enriched features)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _nb_builder import build_notebook  # noqa: E402

cells = [
    ("markdown", "# 04 — Classical Ensemble (V2 Enriched Features)\n\n"
     "Train RF v2, XGBoost, SVM, and Logistic Regression on 31 enriched "
     "features extracted from the 3-channel motor view (C3, Cz, C4).\n\n"
     "Uses V2 data (300 µV PTP threshold → ~4000 epochs vs ~2500 in V1)."),
    ("code", "import subprocess, sys\n"
     "result = subprocess.run(\n"
     "    [sys.executable, '../scripts/run_04_classical_ensemble.py'],\n"
     "    capture_output=False,\n"
     ")\n"
     "result.returncode"),
]

if __name__ == "__main__":
    out = ROOT / "notebooks" / "04_classical_ensemble.ipynb"
    build_notebook(cells, out)
    print(f"Built {out}")
