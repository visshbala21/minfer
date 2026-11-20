"""Model hyperparameters, parsed from a Hugging Face ``config.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of a Qwen2-family decoder-only transformer.

    Only the fields the forward pass needs are kept. ``head_dim`` is derived, not read:
    Qwen2 always uses ``hidden_size // num_heads``.
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    eos_token_id: int
    torch_dtype: torch.dtype

    @property
    def head_dim(self) -> int:
        """Per-head dimension of the query/key/value vectors."""
        return self.hidden_size // self.num_heads

    @property
    def gqa_group_size(self) -> int:
        """Number of query heads that share one key/value head."""
        return self.num_heads // self.num_kv_heads

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        """Build a config from the contents of a Hugging Face ``config.json``.

        Both config layouts in the wild are accepted: the one the Qwen2.5 checkpoints ship
        with (``torch_dtype``, top-level ``rope_theta``) and the one current ``transformers``
        writes (``dtype``, ``rope_parameters.rope_theta``).
        """
        if raw.get("model_type") != "qwen2":
            raise ValueError(f"unsupported model_type {raw.get('model_type')!r}; expected 'qwen2'")
        if raw.get("use_sliding_window"):
            raise ValueError("sliding-window attention is not supported")
        if raw["hidden_size"] % raw["num_attention_heads"]:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if raw["num_attention_heads"] % raw["num_key_value_heads"]:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        rope = raw.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"unsupported rope_type {rope['rope_type']!r}")
        rope_theta = raw["rope_theta"] if "rope_theta" in raw else rope["rope_theta"]
        dtype_name = raw.get("torch_dtype") or raw.get("dtype") or "float32"
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_layers=raw["num_hidden_layers"],
            num_heads=raw["num_attention_heads"],
            num_kv_heads=raw["num_key_value_heads"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=rope_theta,
            max_position_embeddings=raw["max_position_embeddings"],
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            eos_token_id=raw["eos_token_id"],
            torch_dtype=_DTYPES[dtype_name],
        )

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> ModelConfig:
        """Read ``config.json`` from a local model directory."""
        return cls.from_dict(json.loads((Path(model_dir) / "config.json").read_text()))
