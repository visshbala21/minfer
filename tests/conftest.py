"""Shared pytest configuration.

Tests marked ``gpu`` need a CUDA device and the real model weights; they are skipped
automatically elsewhere (CI runs ``-m "not gpu"``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

PRIMARY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip GPU tests when no CUDA device is present."""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tokenizer():
    """Tokenizer of the primary model."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(PRIMARY_MODEL)
