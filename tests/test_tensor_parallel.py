"""Tensor parallelism on the CPU: shard algebra in one process, then two real processes.

This machine has one GPU, so the multi-GPU path (NCCL) cannot run here. What is tested is
everything except the transport: the weight slicing, the lockstep execution protocol and
the sharded KV cache, with two processes exchanging all-reduces over gloo.
"""

from __future__ import annotations

import random

import pytest
import torch
from tiny import dense_greedy, make_tiny_model, make_tiny_shard

from minfer.attention import DenseCausalAttention
from minfer.engine import EngineConfig
from minfer.model import Qwen2Model
from minfer.model.loader import shard_state_dict
from minfer.model.qwen2 import Shard
from minfer.parallel import TensorParallelGroup
from minfer.sampling import SamplingParams


def identity(tensor):
    return tensor


@pytest.mark.parametrize("world_size", [1, 2])
def test_shard_partial_sums_reconstruct_the_full_layers(world_size):
    """Σ over ranks of each row-parallel output equals the unsharded layer's output."""
    full = make_tiny_model()
    shards = [
        make_tiny_shard(Shard(r, world_size, identity), torch.device("cpu"))
        for r in range(world_size)
    ]
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(11, full.config.hidden_size, generator=generator)
    positions = torch.arange(11)
    cos, sin = full.rope(positions)
    dense = DenseCausalAttention(1, 11)
    with torch.inference_mode():
        for layer in range(full.config.num_layers):
            reference = full.layers[layer]
            mlp = sum(s.layers[layer].mlp(x) for s in shards)
            assert torch.allclose(mlp, reference.mlp(x), atol=1e-5)
            attention = sum(s.layers[layer].self_attn(x, cos, sin, layer, dense) for s in shards)
            assert torch.allclose(
                attention, reference.self_attn(x, cos, sin, layer, dense), atol=1e-5
            )


def test_shard_shapes_and_validation():
    full = make_tiny_model()
    config = full.config
    state = shard_state_dict(full.state_dict(), config, rank=1, world_size=2)
    prefix = "layers.0."
    head_dim = config.head_dim
    assert state[prefix + "self_attn.q_proj.weight"].shape == (2 * head_dim, config.hidden_size)
    assert state[prefix + "self_attn.q_proj.bias"].shape == (2 * head_dim,)
    assert state[prefix + "self_attn.k_proj.weight"].shape == (head_dim, config.hidden_size)
    assert state[prefix + "self_attn.o_proj.weight"].shape == (config.hidden_size, 2 * head_dim)
    assert state[prefix + "mlp.gate_proj.weight"].shape == (80, config.hidden_size)
    assert state[prefix + "mlp.down_proj.weight"].shape == (config.hidden_size, 80)
    assert state["embed_tokens.weight"].shape == full.embed_tokens.weight.shape  # replicated
    assert torch.equal(
        state[prefix + "self_attn.q_proj.weight"],
        full.layers[0].self_attn.q_proj.weight[2 * head_dim :],
    )
    model = Qwen2Model(config, Shard(0, 2, identity))
    assert (model.num_heads, model.num_kv_heads) == (2, 1)
    with pytest.raises(ValueError, match="evenly"):
        Qwen2Model(config, Shard(0, 3, identity))


@pytest.fixture(scope="module")
def group():
    group = TensorParallelGroup(
        make_tiny_shard, ["cpu", "cpu"], dist_backend="gloo", attention_backend="reference"
    )
    yield group
    group.close()


def tp_config(**overrides) -> EngineConfig:
    settings = {
        "num_blocks": 400,
        "block_size": 4,
        "attention_backend": "reference",
        "cuda_graphs": False,
    }
    return EngineConfig(**{**settings, **overrides})


def test_two_process_engine_matches_the_single_process_oracle(group):
    model = make_tiny_model()
    rng = random.Random(3)
    prompts = [[rng.randrange(4, 199) for _ in range(rng.randint(1, 30))] for _ in range(12)]
    expected = [dense_greedy(model, p, 16) for p in prompts]
    engine = group.engine(tp_config())
    # The driver's KV store holds only its own kv heads: the cache is sharded.
    assert engine.store.data.shape[4] == model.config.num_kv_heads // 2
    assert engine.generate(prompts, SamplingParams(max_tokens=16)) == expected
    blocks = engine.blocks
    assert blocks.num_free + blocks.num_evictable == blocks.num_usable


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_num_seqs": 2},
        {"prefill_chunk_tokens": 5, "max_batched_tokens": 9},
        {"num_blocks": 30, "watermark": 0.0},
        {"prefix_cache": False},
        {"speculation": "ngram", "ngram_size": 1},
    ],
    ids=["two-at-a-time", "chunked", "memory-pressure", "no-prefix-cache", "ngram-speculation"],
)
def test_two_process_engine_under_scheduling_variants(group, overrides):
    model = make_tiny_model()
    rng = random.Random(4)
    motif = [rng.randrange(4, 199) for _ in range(6)]
    prompts = [motif * rng.randint(1, 3) + [rng.randrange(4, 199)] for _ in range(8)]
    expected = [dense_greedy(model, p, 12) for p in prompts]
    engine = group.engine(tp_config(**overrides))
    params = SamplingParams(max_tokens=12)
    assert engine.generate(prompts, params) == expected
    assert engine.generate(prompts, params) == expected  # again, through the prefix cache


def test_unsupported_configurations_are_rejected(group):
    with pytest.raises(ValueError, match="cuda_graphs"):
        group.engine(tp_config(cuda_graphs=True))
