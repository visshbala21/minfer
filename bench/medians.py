"""Combine repeated runs of a benchmark into medians with their min and max.

    .venv/bin/python -m bench.medians bench/results/jct_7b_r?.json --out bench/results/jct_7b.json

The repetitions must have the same structure (one script, same arguments). The output has
that structure with every number replaced by its median over the repetitions; under
``"repeats"`` it lists the input files and holds two more trees of the same shape with the
minimum and the maximum. Strings and other non-numeric values are taken from the first
file; a list is kept if every repetition has the same one (flags, a command line) and
dropped otherwise (per-session records, error strings) — those are in the input files. A
number missing from some repetition (a run that failed) is the median of the repetitions
that have it, and ``"repeats"["counts"]`` says how many that was wherever it was not all
of them.

The README's tables are generated from these median files: never from a best-of-n.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def combine(trees: list[Any], path: str = "") -> tuple[Any, Any, Any, dict[str, int]]:
    """(median, min, max, counts) of the same position in several result trees."""
    present = [tree for tree in trees if tree is not None]
    if not present:
        return None, None, None, {}
    first = present[0]
    if isinstance(first, dict):
        median, low, high, counts = {}, {}, {}, {}
        keys = list(
            dict.fromkeys(key for tree in present if isinstance(tree, dict) for key in tree)
        )
        for key in keys:
            values = [tree.get(key) if isinstance(tree, dict) else None for tree in trees]
            m, lo, hi, c = combine(values, f"{path}/{key}")
            if m is None and lo is None:
                continue
            median[key], low[key], high[key] = m, lo, hi
            counts.update(c)
        return median, low, high, counts
    if all(_is_number(tree) for tree in present):
        array = np.asarray(present, dtype=float)
        counts = {} if len(present) == len(trees) else {path: len(present)}
        return float(np.median(array)), float(array.min()), float(array.max()), counts
    if isinstance(first, list):
        # Kept when every repetition has the same list (flags, a command line), else dropped.
        same = len(present) == len(trees) and all(tree == first for tree in present)
        return (first, None, None, {}) if same else (None, None, None, {})
    return first, None, None, {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    trees = [json.loads(path.read_text()) for path in args.inputs]
    median, low, high, counts = combine(trees)
    median["repeats"] = {
        "files": [str(path) for path in args.inputs],
        "statistic": "median over the files; min and max below",
        "counts": counts,
        "min": low,
        "max": high,
    }
    args.out.write_text(json.dumps(median, indent=1) + "\n")


if __name__ == "__main__":
    main()
