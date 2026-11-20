"""Paged KV storage and the reference paged-attention backend (CPU only).

The key property: running a sequence through the paged cache in arbitrary pieces — prefill
chunks, single-token decode steps, ragged batches mixed with other sequences, blocks shared
with another sequence — produces the same hidden states as one cache-free pass.
"""

from __future__ import annotations

import pytest
import torch

from minfer.attention import (
    DenseCausalAttention,
    ReferencePagedBackend,
    SequenceSpan,
    build_paged_batch,
)
from minfer.cache import BlockManager
from minfer.cache.kv_store import PagedKVStore
from minfer.model import ModelConfig, Qwen2Model

BLOCK = 4


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
        tie_word_embeddings=True,
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


def make_store(model: Qwen2Model, num_blocks: int) -> PagedKVStore:
    config = model.config
    return PagedKVStore(
        config.num_layers,
        num_blocks,
        BLOCK,
        config.num_kv_heads,
        config.head_dim,
        "cpu",
        model.dtype,
    )


def dense_hidden(model: Qwen2Model, tokens: list[int]) -> torch.Tensor:
    with torch.inference_mode():
        ids = torch.tensor(tokens)
        return model(ids, torch.arange(len(tokens)), DenseCausalAttention(1, len(tokens)))


# ------------------------------------------------------------------------ batch layout


def test_build_paged_batch_layout():
    spans = [
        SequenceSpan(token_ids=[10, 11, 12], start=6, block_table=[5, 2, 9]),  # positions 6..8
        SequenceSpan(token_ids=[20], start=0, block_table=[7]),  # position 0
        SequenceSpan(token_ids=[30, 31], start=3, block_table=[4, 8, 1]),  # 3..4, 3rd unused
    ]
    batch = build_paged_batch(spans, BLOCK, "cpu")
    assert batch.num_sequences == 3 and batch.num_tokens == 6
    assert batch.token_ids.tolist() == [10, 11, 12, 20, 30, 31]
    assert batch.positions.tolist() == [6, 7, 8, 0, 3, 4]
    assert batch.qo_indptr.tolist() == [0, 3, 4, 6]
    # pages needed: ceil(9/4)=3, ceil(1/4)=1, ceil(5/4)=2
    assert batch.kv_indptr.tolist() == [0, 3, 4, 6]
    assert batch.kv_indices.tolist() == [5, 2, 9, 7, 4, 8]
    assert batch.kv_last_page_len.tolist() == [1, 1, 1]
    # slot = block * 2 * BLOCK + offset
    assert batch.slots.tolist() == [
        2 * 8 + 2,
        2 * 8 + 3,
        9 * 8 + 0,
        7 * 8 + 0,
        4 * 8 + 3,
        8 * 8 + 0,
    ]
    assert batch.qo_indptr.dtype == torch.int32 and batch.kv_indices.dtype == torch.int32


def test_build_paged_batch_full_last_page():
    batch = build_paged_batch([SequenceSpan([1, 2, 3, 4], 4, [3, 6])], BLOCK, "cpu")
    assert batch.kv_last_page_len.tolist() == [BLOCK]
    assert batch.kv_indices.tolist() == [3, 6]


def test_build_paged_batch_validates_spans():
    with pytest.raises(ValueError, match="at least one token"):
        build_paged_batch([SequenceSpan([], 0, [1])], BLOCK, "cpu")
    with pytest.raises(ValueError, match="at least one token"):
        build_paged_batch([], BLOCK, "cpu")
    with pytest.raises(ValueError, match="block table"):
        build_paged_batch([SequenceSpan([1, 2], 7, [1, 2])], BLOCK, "cpu")


# ---------------------------------------------------------------------------- KV store


def test_kv_store_write_read_and_block_copy():
    store = PagedKVStore(2, 6, BLOCK, 2, 8, "cpu", torch.float32)
    assert store.nbytes == 2 * 6 * 2 * BLOCK * 2 * 8 * 4
    generator = torch.Generator().manual_seed(0)
    k, v = torch.randn(3, 2, 8, generator=generator), torch.randn(3, 2, 8, generator=generator)
    slots = torch.tensor([3 * 8 + 1, 5 * 8 + 0, 3 * 8 + 3])  # (block 3, off 1), (5, 0), (3, 3)
    store.write(1, slots, k, v)

    layer = store.layer(1)
    assert torch.equal(layer[3, 0, 1], k[0]) and torch.equal(layer[3, 1, 1], v[0])
    assert torch.equal(layer[5, 0, 0], k[1]) and torch.equal(layer[5, 1, 0], v[1])
    assert torch.equal(layer[3, 0, 3], k[2]) and torch.equal(layer[3, 1, 3], v[2])
    assert store.layer(0).abs().sum() == 0  # other layers untouched
    read_k, read_v = store.read(1, slots)
    assert torch.equal(read_k, k) and torch.equal(read_v, v)

    store.copy_block_prefix(src_block=3, dst_block=2, num_tokens=2)
    assert torch.equal(store.layer(1)[2, :, :2], store.layer(1)[3, :, :2])
    assert store.layer(1)[2, :, 2:].abs().sum() == 0  # positions past the prefix not copied


# ------------------------------------------------------- paged reference vs dense oracle


def run_schedule(model, backend, manager, sequences, schedule):
    """Execute steps of ``{name: num_new_tokens}`` and collect per-position hidden states."""
    hidden = {name: {} for name in sequences}
    progress = {name: sequences[name]["start"] for name in sequences}
    for step in schedule:
        spans, order = [], []
        for name, count in step.items():
            start = progress[name]
            tokens = sequences[name]["tokens"][start : start + count]
            assert len(tokens) == count
            manager.ensure_capacity(name, start + count)
            spans.append(SequenceSpan(tokens, start, manager.table(name)))
            order.append((name, start, count))
        batch = build_paged_batch(spans, BLOCK, "cpu")
        is_decode = all(count == 1 for _, _, count in order)
        context = backend.decode(batch) if is_decode else backend.prefill(batch)
        with torch.inference_mode():
            out = model(batch.token_ids, batch.positions, context)
        row = 0
        for name, start, count in order:
            for offset in range(count):
                hidden[name][start + offset] = out[row]
                row += 1
            progress[name] = start + count
    return hidden


def test_paged_reference_matches_dense_oracle():
    model = make_tiny_model()
    manager = BlockManager(num_blocks=64, block_size=BLOCK)
    store = make_store(model, manager.num_blocks)
    backend = ReferencePagedBackend(store)

    generator = torch.Generator().manual_seed(1)
    vocab = model.config.vocab_size
    a = torch.randint(0, vocab, (45,), generator=generator).tolist()
    b = torch.randint(0, vocab, (30,), generator=generator).tolist()
    c = torch.randint(0, vocab, (7,), generator=generator).tolist()
    sequences = {name: {"tokens": t, "start": 0} for name, t in (("a", a), ("b", b), ("c", c))}

    schedule = [
        {"a": 10, "b": 30},  # ragged prefill; block tables interleave as both grow
        {"a": 13, "c": 7},  # second chunk of a (ends mid-block), all of c
        {"a": 1},  # decode, batch of one
        {"a": 1},
    ]
    hidden = run_schedule(model, backend, manager, sequences, schedule)
    # a and b allocated in the same step, so a's later blocks are not contiguous with its first
    table = manager.table("a")
    assert table != list(range(table[0], table[0] + len(table)))

    for name, tokens in (("a", a[:25]), ("b", b), ("c", c)):
        reference = dense_hidden(model, tokens)
        got = torch.stack([hidden[name][p] for p in range(len(tokens))])
        torch.testing.assert_close(got, reference, atol=2e-5, rtol=0)


def test_decode_batches_match_dense_oracle():
    model = make_tiny_model(seed=3)
    manager = BlockManager(num_blocks=64, block_size=BLOCK)
    backend = ReferencePagedBackend(make_store(model, manager.num_blocks))
    generator = torch.Generator().manual_seed(4)
    tokens = {
        name: torch.randint(0, model.config.vocab_size, (n,), generator=generator).tolist()
        for name, n in (("x", 19), ("y", 12), ("z", 9))
    }
    sequences = {name: {"tokens": t, "start": 0} for name, t in tokens.items()}
    # Prefill different amounts, then decode all three together until the shortest ends.
    schedule = [{"x": 12, "y": 5, "z": 2}] + [{"x": 1, "y": 1, "z": 1}] * 7
    hidden = run_schedule(model, backend, manager, sequences, schedule)
    for name, total in (("x", 19), ("y", 12), ("z", 9)):
        reference = dense_hidden(model, tokens[name][:total])
        got = torch.stack([hidden[name][p] for p in range(total)])
        torch.testing.assert_close(got, reference, atol=2e-5, rtol=0)


def test_shared_prefix_blocks_give_identical_results():
    """A sequence that adopts another's blocks for a common prefix needs no recomputation."""
    model = make_tiny_model(seed=5)
    manager = BlockManager(num_blocks=64, block_size=BLOCK)
    backend = ReferencePagedBackend(make_store(model, manager.num_blocks))
    generator = torch.Generator().manual_seed(6)
    vocab = model.config.vocab_size
    first = torch.randint(0, vocab, (22,), generator=generator).tolist()
    shared = 2 * BLOCK  # two full blocks
    second = first[:shared] + torch.randint(0, vocab, (15,), generator=generator).tolist()

    sequences = {"first": {"tokens": first, "start": 0}}
    run_schedule(model, backend, manager, sequences, [{"first": 22}])

    manager.adopt("second", manager.table("first")[: shared // BLOCK])
    sequences = {"second": {"tokens": second, "start": shared}}
    hidden = run_schedule(model, backend, manager, sequences, [{"second": 9}, {"second": 6}])

    reference = dense_hidden(model, second)
    got = torch.stack([hidden["second"][p] for p in range(shared, len(second))])
    torch.testing.assert_close(got, reference[shared:], atol=2e-5, rtol=0)
    # The donor's blocks are shared, not copied, and survive the donor being freed.
    assert manager.table("second")[:2] == manager.table("first")[:2]
    manager.free("first")
    assert all(manager.ref_count(b) == 1 for b in manager.table("second"))
    manager.check_invariants(external_refs={})
