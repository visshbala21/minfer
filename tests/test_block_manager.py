"""Unit tests for the KV block manager (CPU only)."""

from __future__ import annotations

import random

import pytest

from minfer.cache.block_manager import (
    BlockManager,
    OutOfBlocks,
    kv_bytes_per_block,
    plan_num_blocks,
)


def test_budget_math_for_qwen25_1p5b():
    # 28 layers, 2 KV heads, head_dim 128, 16-token blocks, bf16 (2 bytes), keys + values.
    per_block = kv_bytes_per_block(28, 2, 128, 16, 2)
    assert per_block == 28 * 2 * 16 * 2 * 128 * 2 == 458_752
    assert plan_num_blocks(10 * per_block + 5, per_block, 1.0) == 10
    assert plan_num_blocks(10 * per_block, per_block, 0.5) == 5
    with pytest.raises(ValueError):
        plan_num_blocks(1, 1, 0.0)


def test_reserved_blocks_are_never_allocated():
    manager = BlockManager(num_blocks=8, block_size=16, num_reserved=1)
    assert manager.num_usable == 7 and manager.num_free == 7
    blocks = manager.allocate(7)
    assert 0 not in blocks and sorted(blocks) == list(range(1, 8))
    assert manager.num_free == 0 and manager.num_used == 7
    with pytest.raises(OutOfBlocks):
        manager.allocate(1)


def test_failed_allocation_leaves_pool_untouched():
    manager = BlockManager(num_blocks=5)
    manager.allocate(2)
    with pytest.raises(OutOfBlocks):
        manager.allocate(3)
    assert manager.num_free == 2
    manager.check_invariants()


def test_reference_counting_frees_only_at_zero():
    manager = BlockManager(num_blocks=6)
    blocks = manager.allocate(2)
    manager.retain(blocks)
    assert [manager.ref_count(b) for b in blocks] == [2, 2]
    assert manager.release(blocks) == 0 and manager.num_free == 3
    assert manager.release(blocks) == 2 and manager.num_free == 5
    with pytest.raises(ValueError):
        manager.release(blocks[:1])
    with pytest.raises(ValueError):
        manager.retain(blocks[:1])


def test_freed_blocks_are_reused():
    manager = BlockManager(num_blocks=4)
    first = manager.allocate(3)
    manager.release(first[1:2])
    assert manager.allocate(1) == first[1:2]


def test_tables_grow_by_whole_blocks():
    manager = BlockManager(num_blocks=64, block_size=16)
    assert manager.blocks_for(0) == 0 and manager.blocks_for(1) == 1
    assert manager.blocks_for(16) == 1 and manager.blocks_for(17) == 2

    assert len(manager.ensure_capacity("a", 40)) == 3
    assert manager.capacity("a") == 48 and len(manager.table("a")) == 3
    assert manager.ensure_capacity("a", 48) == []
    assert manager.blocks_missing("a", 49) == 1
    assert len(manager.ensure_capacity("a", 49)) == 1
    assert manager.free("a") == 4 and manager.table("a") == []
    assert manager.free("a") == 0


def test_ensure_capacity_is_atomic_on_failure():
    manager = BlockManager(num_blocks=4, block_size=4)
    manager.ensure_capacity("a", 8)
    with pytest.raises(OutOfBlocks):
        manager.ensure_capacity("a", 100)
    assert len(manager.table("a")) == 2 and manager.num_free == 1
    manager.check_invariants(external_refs={})


def test_adopt_shares_blocks_and_free_keeps_other_owner():
    manager = BlockManager(num_blocks=16, block_size=4)
    manager.ensure_capacity("a", 12)
    shared = manager.table("a")[:2]
    manager.adopt("b", shared)
    assert [manager.ref_count(b) for b in shared] == [2, 2]
    manager.ensure_capacity("b", 12)
    assert manager.table("b")[:2] == shared and manager.table("b")[2] not in manager.table("a")
    assert manager.free("a") == 1  # only a's private third block is returned
    assert [manager.ref_count(b) for b in shared] == [1, 1]
    manager.check_invariants(external_refs={})
    assert manager.free("b") == 3
    assert manager.num_free == manager.num_usable


def test_adopt_rebinds_a_prefix_and_releases_displaced_blocks():
    manager = BlockManager(num_blocks=12, block_size=4)
    manager.ensure_capacity("a", 16)  # 4 blocks
    manager.ensure_capacity("b", 10)  # 3 private blocks
    own = list(manager.table("b"))
    shared = manager.table("a")[:2]
    manager.adopt("b", shared)
    assert manager.table("b") == [*shared, own[2]]
    assert [manager.ref_count(b) for b in shared] == [2, 2]
    assert [manager.ref_count(b) for b in own[:2]] == [0, 0]  # displaced and freed
    manager.adopt("b", manager.table("a"))  # longer than b's table: replaces and appends
    assert manager.table("b") == manager.table("a")
    assert manager.ref_count(own[2]) == 0
    manager.adopt("b", manager.table("a"))  # idempotent
    assert all(manager.ref_count(b) == 2 for b in manager.table("a"))
    manager.check_invariants(external_refs={})


def test_cached_blocks_and_evictable_accounting():
    manager = BlockManager(num_blocks=8, block_size=4)
    a, b = manager.ensure_capacity("seq", 8)
    assert manager.num_evictable == 0
    manager.cache_block(a)
    manager.cache_block(b)
    assert manager.is_cached(a) and manager.num_evictable == 0  # still used by "seq"
    with pytest.raises(ValueError):
        manager.cache_block(a)
    manager.free("seq")
    assert manager.num_evictable == 2 and manager.num_free == 5  # held by the cache alone
    manager.adopt("other", [a])
    assert manager.num_evictable == 1
    manager.check_invariants(external_refs={a: 1, b: 1})
    manager.uncache_block(b)
    assert manager.num_evictable == 0 and manager.num_free == 6
    manager.uncache_block(a)  # still referenced by "other": not freed
    assert manager.ref_count(a) == 1 and not manager.is_cached(a)
    with pytest.raises(ValueError):
        manager.uncache_block(a)
    manager.free("other")
    assert manager.num_free == manager.num_usable
    manager.check_invariants(external_refs={})


def test_randomized_operations_keep_invariants():
    rng = random.Random(0)
    manager = BlockManager(num_blocks=97, block_size=8, num_reserved=1)
    lengths: dict[int, int] = {}
    pinned: dict[int, int] = {}  # references held "externally", like a prefix cache would
    next_id = 0
    for _ in range(4000):
        action = rng.random()
        if action < 0.35:
            seq, tokens = next_id, rng.randint(1, 60)
            next_id += 1
            if lengths and rng.random() < 0.4:  # share a prefix with an existing sequence
                donor = rng.choice(list(lengths))
                manager.adopt(
                    seq, manager.table(donor)[: rng.randint(0, len(manager.table(donor)))]
                )
            try:
                manager.ensure_capacity(seq, tokens)
                lengths[seq] = max(tokens, manager.capacity(seq))
            except OutOfBlocks:
                manager.free(seq)
        elif action < 0.6 and lengths:
            seq = rng.choice(list(lengths))
            try:
                manager.ensure_capacity(seq, lengths[seq] + rng.randint(1, 20))
                lengths[seq] = manager.capacity(seq)
            except OutOfBlocks:
                pass
        elif action < 0.8 and lengths:
            seq = rng.choice(list(lengths))
            manager.free(seq)
            del lengths[seq]
        elif action < 0.9 and lengths:
            block = rng.choice(manager.table(rng.choice(list(lengths))))
            manager.retain([block])
            pinned[block] = pinned.get(block, 0) + 1
        elif pinned:
            block = rng.choice(list(pinned))
            manager.release([block])
            pinned[block] -= 1
            if not pinned[block]:
                del pinned[block]
        manager.check_invariants(external_refs=pinned)
    for seq in list(lengths):
        manager.free(seq)
    manager.release([b for b, n in pinned.items() for _ in range(n)])
    assert manager.num_free == manager.num_usable
