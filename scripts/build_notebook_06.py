"""Build notebook 06 — Stacking Ensemble."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _nb_builder import build_notebook  # noqa: E402

cells = [
    ("markdown", "# 06 — Stacking Ensemble\n\n"
     "Combine the best classical model, V2 CSP+LDA, and V2 EEGNet via "
     "a logistic regression meta-learner trained on val-set predictions.\n\n"
     "Evaluates on the held-out test set and compares all V1 + V2 models."),
    ("code", "import subprocess, sys\n"
     "result = subprocess.run(\n"
     "    [sys.executable, '../scripts/run_06_ensemble.py'],\n"
     "    capture_output=False,\n"
     ")\n"
     "result.returncode"),
]

if __name__ == "__main__":
    out = ROOT / "notebooks" / "06_ensemble.ipynb"
    build_notebook(cells, out)
    print(f"Built {out}")
