"""Build notebooks/01_random_forest.ipynb."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._nb_builder import build_notebook  # noqa: E402


CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# 01 — Random Forest on Band-Power Features

Trains a Random Forest classifier on log mu (8–13 Hz) + log beta (13–30 Hz)
band-power features extracted from 3 motor channels (C3, Cz, C4).

Evaluates both candidate windows ([0.0, 4.0] and [0.5, 2.5] seconds), picks
the winner by val macro-F1, and computes test metrics on the winning window
only. See `PIPELINE_PLAN.md` §5.
"""),
    ("code", """\
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_01_random_forest import main as run_rf
"""),
    ("markdown", "## Train + evaluate"),
    ("code", "run_rf()"),
    ("markdown", "## Inspect saved results"),
    ("code", """\
import json
result = json.loads((ROOT / 'reports' / 'rf_results.json').read_text())
print('Winning window:', result['winning_window'])
print('Test metrics:')
for k, v in result['test_metrics'].items():
    print(f'  {k}: {v}')
print('Top features:')
for name, imp in sorted(result['feature_importances'].items(), key=lambda kv: -kv[1])[:6]:
    print(f'  {name}: {imp:.4f}')
"""),
    ("markdown", """\
## Confusion matrix and feature importance

Saved alongside the JSON in `reports/figures/`:
- `rf_confusion.png` — 2x2 normalised confusion matrix on test
- `rf_feature_importance.png` — top RF feature importances
"""),
    ("code", """\
from IPython.display import Image, display
display(Image(str(ROOT / 'reports' / 'figures' / 'rf_confusion.png')))
display(Image(str(ROOT / 'reports' / 'figures' / 'rf_feature_importance.png')))
"""),
]


def main() -> None:
    NOTEBOOK_DIR = ROOT / "notebooks"
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    build_notebook(CELLS, NOTEBOOK_DIR / "01_random_forest.ipynb")
    print('Wrote', NOTEBOOK_DIR / "01_random_forest.ipynb")


if __name__ == "__main__":
    main()
