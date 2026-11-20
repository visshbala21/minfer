"""The models the benchmarks run on, and where each one's results live.

The headline model is Qwen2.5-7B-Instruct; Qwen2.5-1.5B-Instruct, the model the engine was
developed on, is kept as a secondary set of results. Result files of the 1.5B model keep
their original names (``bench/results/matrix/``, ``ablations.json``, ...); the 7B model's
carry a ``_7b`` suffix.
"""

from __future__ import annotations

from pathlib import Path

PRIMARY = "Qwen/Qwen2.5-7B-Instruct"
SECONDARY = "Qwen/Qwen2.5-1.5B-Instruct"
MODELS = {"7b": PRIMARY, "1.5b": SECONDARY}


def resolve(name: str) -> str:
    """A model id from a short name (``7b``, ``1.5b``) or a full id."""
    return MODELS.get(name.lower(), name)


def tag(model: str) -> str:
    """Short name of ``model`` (``7b``, ``1.5b``), derived from its id if it is not listed."""
    for short, full in MODELS.items():
        if full == model:
            return short
    return model.rsplit("/", 1)[-1].lower()


def results_path(path: str | Path, model: str) -> Path:
    """``path`` for the 1.5B model; ``<stem>_<tag><suffix>`` for any other model."""
    path = Path(path)
    if model == SECONDARY:
        return path
    return path.with_name(f"{path.stem}_{tag(model)}{path.suffix}")
