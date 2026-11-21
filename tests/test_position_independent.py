"""Position-independent KV reuse: rotation, the content index, and the engine (CPU only)."""

from __future__ import annotations

import random

import pytest
import torch
from test_engine import assert_all_released
from tiny import dense_greedy, make_tiny_model

from minfer.cache.block_manager import BlockManager
from minfer.cache.kv_store import PagedKVStore
from minfer.cache.radix_tree import PrefixCache
from minfer.engine import Engine, EngineConfig
from minfer.model import ModelConfig, Qwen2Model
from minfer.model.qwen2 import RotaryEmbedding, apply_rope
from minfer.sampling import SamplingParams

SIZE = 4


def rotation(
    shifts: torch.Tensor, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    angles = shifts.float()[:, None] * (1.0 / theta**exponents)[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


def test_copy_rotated_moves_keys_to_their_new_positions():
    """A key rotated for position p, copied with shift Δ, is the key rotated for p + Δ."""
    head_dim, theta = 16, 10000.0
    rope = RotaryEmbedding(head_dim, theta, 512)
    store = PagedKVStore(2, 12, SIZE, 2, head_dim, "cpu", torch.float32)
    generator = torch.Generator().manual_seed(0)
    raw_keys = torch.randn(2, 6, 2, head_dim, generator=generator)  # [layers, tokens, heads, dim]
    values = torch.randn(2, 6, 2, head_dim, generator=generator)
    old = torch.tensor([3, 4, 5, 40, 41, 42])
    shift = torch.tensor([200, 200, 200, -37, -37, -37])
    source = torch.tensor([8, 9, 10, 24, 25, 26])  # blocks 1 and 3, offsets 0-2
    target = torch.tensor([48, 49, 50, 65, 66, 67])  # block 6 offsets 0-2, block 8 offsets 1-3
    for layer in range(2):
        cos, sin = rope(old)
        store.write(layer, source, apply_rope(raw_keys[layer], cos, sin), values[layer])
    store.copy_rotated(source, target, *rotation(shift, head_dim, theta))
    for layer in range(2):
        cos, sin = rope(old + shift)
        keys, copied_values = store.read(layer, target)
        assert torch.allclose(keys, apply_rope(raw_keys[layer], cos, sin), atol=1e-5)
        assert torch.equal(copied_values, values[layer])


def tokens_of(seed: int, count: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1, 1000) for _ in range(count)]


def test_find_reuse_locates_cached_tokens_at_other_positions():
    blocks = BlockManager(64, SIZE)
    cache = PrefixCache(blocks, content_index=True)
    known = tokens_of(1, 40)
    blocks.ensure_capacity("a", 40)
    table = list(blocks.table("a"))
    cache.sync_sequence("a", [*known, 0], 40)
    blocks.free("a")
    # Blocks 2-6 of the cached sequence (positions 8-27) reappear behind 13 new tokens,
    # not aligned to the block grid of the new prompt.
    prompt = [*tokens_of(2, 13), *known[8:28], *tokens_of(3, 9)]
    assert cache.find_reuse(prompt, 0, len(prompt) - 1, 8) == [(13, 8, table[2:7])]
    # A run must reach the minimum length, and nothing outside [start, stop) is used.
    assert cache.find_reuse(prompt, 0, len(prompt) - 1, 24) == []
    assert cache.find_reuse(prompt, 14, len(prompt) - 1, 8) == [(17, 12, table[3:7])]
    assert cache.find_reuse(prompt, 0, 29, 8) == [(13, 8, table[2:6])]
    # Tokens that only look alike for part of a block are not a match.
    partial = [*tokens_of(2, 13), *known[9:28], 0]
    assert cache.find_reuse(partial, 0, 32, 8) == [(16, 12, table[3:7])]
    # Evicted KV is no longer offered.
    cache.evict(10)
    assert cache.find_reuse(prompt, 0, len(prompt) - 1, 8) == []
    cache.check_invariants()


def one_layer_model() -> Qwen2Model:
    """One transformer layer: its keys and values depend on nothing but the token."""
    config = ModelConfig(
        vocab_size=199,
        hidden_size=64,
        intermediate_size=160,
        num_layers=1,
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
    generator = torch.Generator().manual_seed(4)
    for name, param in model.named_parameters():
        noise = torch.randn(param.shape, generator=generator)
        param.copy_(
            1.0 + 0.1 * noise if "norm" in name else noise * (1.0 if "embed" in name else 0.15)
        )
    return model.eval()


def make_engine(model, **overrides) -> Engine:
    config = EngineConfig(
        attention_backend="reference",
        num_blocks=200,
        block_size=SIZE,
        position_independent_reuse=True,
        reuse_min_tokens=8,
        **overrides,
    )
    return Engine(model, config, eos_token_ids=())


def run(engine: Engine, prompt: list[int], max_tokens: int = 8) -> tuple[list[int], int]:
    engine.add_request(prompt, SamplingParams(max_tokens=max_tokens))
    tokens, cached = [], 0
    while engine.has_work:
        for out in engine.step():
            tokens.append(out.token_id)
            cached = out.cached_tokens
    return tokens, cached


@pytest.mark.parametrize("ratio", [0.0, 0.25])
def test_reuse_is_exact_for_a_one_layer_model(ratio):
    """With one layer, cached KV does not depend on context: shifted reuse changes nothing."""
    model = one_layer_model()
    rng = random.Random(8)

    def fresh(count: int) -> list[int]:
        return [rng.randrange(4, 199) for _ in range(count)]

    document, other = fresh(48), fresh(40)
    engine = make_engine(model, reuse_recompute_ratio=ratio)
    for prompt in ([*fresh(7), *document, *fresh(5)], [*fresh(3), *other]):
        assert run(engine, prompt)[0] == dense_greedy(model, prompt, 8)
    # The same document and the other text again, behind different openings and in the
    # opposite order: no common prefix, two shifted duplicates.
    prompt = [*fresh(18), *other[8:], *fresh(6), *document, *fresh(9)]
    output, cached = run(engine, prompt)
    assert output == dense_greedy(model, prompt, 8)
    stats = engine.prefix_cache.stats
    assert stats.reuse_spans == 2 and stats.reused_tokens >= 40
    assert cached == stats.reused_tokens  # reported to the client as tokens not prefilled
    if ratio:
        assert stats.reuse_recomputed_tokens > 0
    assert_all_released(engine)


def test_reuse_saves_prefill_and_releases_its_pins_on_a_deeper_model():
    """Three layers: the reuse is approximate, so only the accounting is asserted."""
    model = make_tiny_model()
    rng = random.Random(9)

    def fresh(count: int) -> list[int]:
        return [rng.randrange(4, 199) for _ in range(count)]

    document = fresh(64)
    engine = make_engine(model, reuse_recompute_ratio=0.25, max_batched_tokens=24)
    run(engine, [*fresh(5), *document])
    before = engine.stats.prefill_tokens
    prompt = [*fresh(11), *document, *fresh(6)]
    output, cached = run(engine, prompt)
    assert len(output) == 8
    assert cached >= 40 and engine.stats.prefill_tokens - before == len(prompt) - cached
    assert_all_released(engine)
    # An aborted request with reuse still planned gives its pinned sources back.
    seq_id = engine.add_request([*fresh(9), *document, *fresh(30)], SamplingParams(max_tokens=4))
    engine.step()
    engine.abort(seq_id)
    assert_all_released(engine)


def test_reuse_is_off_by_default():
    model = make_tiny_model()
    engine = Engine(
        model,
        EngineConfig(attention_backend="reference", num_blocks=200, block_size=SIZE),
        eos_token_ids=(),
    )
    rng = random.Random(10)
    document = [rng.randrange(4, 199) for _ in range(64)]
    run(engine, [5, 6, 7, *document])
    prompt = [9, 9, 9, 9, 9, *document]
    assert run(engine, prompt) == (dense_greedy(model, prompt, 8), 0)
