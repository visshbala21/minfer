"""A tiny randomly initialized Qwen2 for CPU tests."""

from __future__ import annotations

import torch

from minfer.attention import DenseCausalAttention
from minfer.engine import Engine, EngineConfig
from minfer.model import ModelConfig, Qwen2Model


def make_tiny_model(seed: int = 0) -> Qwen2Model:
    """A small randomly initialized Qwen2 (float32, CPU)."""
    config = ModelConfig(
        vocab_size=199,
        hidden_size=64,
        intermediate_size=160,
        num_layers=3,
        num_heads=4,
        num_kv_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        eos_token_id=3,
        torch_dtype=torch.float32,
    )
    model = Qwen2Model(config)
    generator = torch.Generator().manual_seed(seed)
    for name, param in model.named_parameters():
        noise = torch.randn(param.shape, generator=generator)
        if "norm" in name:
            param.copy_(1.0 + 0.1 * noise)
        elif "embed" in name:
            param.copy_(noise)
        else:
            param.copy_(noise * 0.15)
    return model.eval()


def dense_logits(model: Qwen2Model, tokens: list[int]) -> torch.Tensor:
    """Next-token logits after ``tokens`` from one cache-free pass."""
    with torch.inference_mode():
        ids = torch.tensor(tokens, device=model.device)
        positions = torch.arange(len(tokens), device=model.device)
        hidden = model(ids, positions, DenseCausalAttention(1, len(tokens)))
        return model.logits(hidden[-1:])[0]


def dense_greedy(
    model: Qwen2Model, prompt: list[int], max_tokens: int, eos: int | None = None
) -> list[int]:
    """Greedy generation by recomputing the whole sequence each step (the oracle)."""
    tokens = list(prompt)
    out: list[int] = []
    for _ in range(max_tokens):
        token = int(dense_logits(model, tokens).argmax())
        tokens.append(token)
        out.append(token)
        if token == eos:
            break
    return out


def make_tiny_engine(**config: object) -> Engine:
    """A CPU engine over the tiny model (top-level so it can be built in a child process)."""
    config.setdefault("num_blocks", 400)
    config.setdefault("block_size", 4)
    return Engine(make_tiny_model(), EngineConfig(attention_backend="reference", **config))
