"""Build a notebook .ipynb from a list of (kind, source) cell tuples.

Used by ``scripts/build_notebooks.py`` so each notebook's content is defined
in plain Python and we don't have to hand-roll JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _cell(kind: str, source: str) -> dict:
    if kind == "code":
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }
    if kind == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
    raise ValueError(f"Unknown cell kind: {kind}")


def build_notebook(cells: Iterable[tuple[str, str]], path: str | Path) -> None:
    nb = {
        "cells": [_cell(k, s) for k, s in cells],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
