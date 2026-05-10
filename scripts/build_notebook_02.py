"""Build notebooks/02_csp_lda.ipynb."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._nb_builder import build_notebook  # noqa: E402


CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# 02 — CSP + LDA on 64-channel epochs

Common Spatial Patterns + Linear Discriminant Analysis. CSP learns spatial
filters that maximise variance ratio between T1 and T2; LDA classifies on
log-variance features. See `PIPELINE_PLAN.md` §6.
"""),
    ("code", """\
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_02_csp_lda import main as run_csp_lda
"""),
    ("markdown", "## Train + evaluate"),
    ("code", "run_csp_lda()"),
    ("markdown", "## Inspect saved results"),
    ("code", """\
import json
result = json.loads((ROOT / 'reports' / 'csp_lda_results.json').read_text())
print('Winning window:', result['winning_window'])
print('Winning n_components:', result['winning_n_components'])
print('Test metrics:')
for k, v in result['test_metrics'].items():
    print(f'  {k}: {v}')
"""),
    ("markdown", """\
## CSP scalp patterns

The killer interpretability plot: each topomap is one CSP spatial pattern.
For successful left/right motor imagery decoding, expect contralateral
red/blue blobs near C3 / C4 in the top patterns.
"""),
    ("code", """\
from IPython.display import Image, display
display(Image(str(ROOT / 'reports' / 'figures' / 'csp_topomaps.png')))
display(Image(str(ROOT / 'reports' / 'figures' / 'csp_lda_confusion.png')))
"""),
]


def main() -> None:
    NOTEBOOK_DIR = ROOT / "notebooks"
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    build_notebook(CELLS, NOTEBOOK_DIR / "02_csp_lda.ipynb")
    print('Wrote', NOTEBOOK_DIR / "02_csp_lda.ipynb")


if __name__ == "__main__":
    main()
