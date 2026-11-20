"""Radix-tree prefix cache: insert, match, split, evict, and reference counting."""

from __future__ import annotations

import random

from minfer.cache import BlockManager, PrefixCache

BLOCK = 4


def make(num_blocks: int = 64) -> tuple[PrefixCache, BlockManager]:
    blocks = BlockManager(num_blocks, BLOCK)
    return PrefixCache(blocks), blocks


def toks(*runs: tuple[int, int]) -> list[int]:
    """Tokens made of runs ``(value, blocks)``: each run is ``blocks`` full blocks of ``value``."""
    return [value for value, blocks in runs for _ in range(blocks * BLOCK)]


def compute(cache: PrefixCache, seq_id, tokens: list[int]) -> int:
    """Admit a sequence, "compute" everything it still needs, and share it via the tree."""
    blocks = cache.blocks
    cached = cache.sync_sequence(seq_id, tokens, 0)
    blocks.ensure_capacity(seq_id, len(tokens))
    cache.sync_sequence(seq_id, [*tokens, -1], len(tokens))  # one more token pending
    return cached


def check(cache: PrefixCache) -> None:
    cache.check_invariants()
    cache.blocks.check_invariants(external_refs=None)


# ---------------------------------------------------------------------------- insert / match


def test_empty_cache_matches_nothing():
    cache, _ = make()
    assert cache.match_length(toks((1, 3))) == 0
    assert cache.sync(toks((1, 3)), [], 2) == []
    assert cache.num_nodes == 0


def test_insert_then_match_shares_the_same_blocks():
    cache, blocks = make()
    tokens = toks((1, 3))
    assert compute(cache, "a", tokens) == 0
    table = blocks.table("a")
    assert cache.num_nodes == 1 and cache.num_blocks == 3
    assert all(blocks.ref_count(b) == 2 for b in table)  # sequence + cache

    longer = [*tokens, 7, 7]
    assert cache.match_length(longer) == 12
    assert cache.sync_sequence("b", longer, 0) == 12
    assert blocks.table("b") == table
    assert all(blocks.ref_count(b) == 3 for b in table)
    check(cache)


def test_match_always_leaves_a_pending_token():
    cache, blocks = make()
    tokens = toks((1, 3))
    compute(cache, "a", tokens)
    # An identical request may only take two blocks: its 12th token must be recomputed.
    assert cache.match_length(tokens) == 8
    assert cache.sync_sequence("b", tokens, 0) == 8
    assert blocks.table("b") == blocks.table("a")[:2]
    assert cache.match_length(tokens[:4]) == 0


def test_match_is_block_aligned_and_stops_at_first_difference():
    cache, _ = make()
    tokens = toks((1, 4))
    compute(cache, "a", tokens)
    differs_mid_block = [*tokens[:6], 99, *tokens[7:], 5]
    assert cache.match_length(differs_mid_block) == 4  # only the first block is identical
    differs_at_start = [99, *tokens[1:], 5]
    assert cache.match_length(differs_at_start) == 0


def test_match_length_does_not_touch_lru():
    cache, _ = make()
    compute(cache, "a", toks((1, 2)))
    (node,) = cache.nodes()
    before = node.last_access
    cache.match_length([*toks((1, 2)), 0])
    assert node.last_access == before


def test_extending_a_cached_sequence_lengthens_its_leaf():
    cache, blocks = make()
    compute(cache, "turn1", toks((1, 2)))
    turn2 = toks((1, 2), (2, 2))
    assert compute(cache, "turn2", turn2) == 8
    assert cache.num_nodes == 1 and cache.num_blocks == 4
    (leaf,) = cache.root.children.values()
    assert leaf.blocks == blocks.table("turn2") and leaf.tokens == tuple(turn2)
    assert leaf.blocks[:2] == blocks.table("turn1")
    check(cache)


def test_extending_an_interior_node_adds_a_child():
    cache, _ = make()
    compute(cache, "a", toks((1, 2), (2, 1)))
    compute(cache, "b", toks((1, 2), (3, 1)))  # splits: head (1,1) with children (2) and (3)
    assert cache.num_nodes == 3
    compute(cache, "c", toks((1, 2), (4, 2)))
    assert cache.num_nodes == 4 and cache.num_blocks == 6
    (head,) = cache.root.children.values()
    assert len(head.children) == 3
    check(cache)


def test_growing_sequence_inserts_incrementally():
    cache, blocks = make()
    tokens = list(range(100, 100 + 6 * BLOCK))
    cached = cache.sync_sequence("s", tokens[:9], 0)
    for length in range(9, len(tokens) + 1):  # one token at a time, like decoding
        blocks.ensure_capacity("s", length)
        cached = cache.sync_sequence("s", [*tokens[:length], -1], length)
        assert cached == length
        assert cache.num_blocks == length // BLOCK
    # The leaf's edge was lengthened in place: one node, however long the sequence grows.
    assert cache.num_nodes == 1
    assert cache.match_length([*tokens, -1]) == len(tokens)
    check(cache)


# ----------------------------------------------------------------------------------- split


def test_divergence_inside_an_edge_splits_it():
    cache, blocks = make()
    a = toks((1, 2), (2, 2))
    b = toks((1, 2), (3, 2))
    compute(cache, "a", a)
    assert compute(cache, "b", b) == 8
    assert cache.num_nodes == 3 and cache.num_blocks == 6
    (head,) = cache.root.children.values()
    assert head.tokens == tuple(toks((1, 2))) and head.blocks == blocks.table("a")[:2]
    tails = sorted(head.children.values(), key=lambda n: n.tokens)
    assert [t.tokens for t in tails] == [tuple(toks((2, 2))), tuple(toks((3, 2)))]
    assert tails[0].blocks == blocks.table("a")[2:] and tails[1].blocks == blocks.table("b")[2:]
    assert blocks.table("b")[:2] == blocks.table("a")[:2]
    check(cache)


def test_split_keeps_grandchildren_reachable():
    cache, _ = make()
    compute(cache, "a", toks((1, 3), (2, 1)))
    compute(cache, "b", toks((1, 3), (4, 1)))  # head of three blocks with two children
    assert cache.num_nodes == 3
    compute(cache, "c", toks((1, 1), (9, 1)))  # splits the head after its first block
    assert cache.num_nodes == 5
    assert cache.match_length([*toks((1, 3), (2, 1)), 0]) == 16
    assert cache.match_length([*toks((1, 3), (4, 1)), 0]) == 16
    assert cache.match_length([*toks((1, 1), (9, 1)), 0]) == 8
    check(cache)


def test_lookup_that_diverges_does_not_split():
    cache, _ = make()
    compute(cache, "a", toks((1, 4)))
    assert cache.sync_sequence("b", toks((1, 2), (5, 2)), 0) == 8
    assert cache.num_nodes == 1  # b had nothing to insert, so the edge stays whole
    check(cache)


# ------------------------------------------------------------------------------ duplicates


def test_duplicate_computation_collapses_onto_the_tree_copy():
    """Two sequences that computed the same prefix independently end up sharing one copy."""
    cache, blocks = make()
    tokens = toks((1, 3))
    blocks.ensure_capacity("a", 12)
    blocks.ensure_capacity("b", 12)
    own_b = list(blocks.table("b"))
    free_before = blocks.num_free
    cache.sync_sequence("a", [*tokens, -1], 12)
    cache.sync_sequence("b", [*tokens, -1], 12)
    assert blocks.table("b") == blocks.table("a")
    assert all(blocks.ref_count(b) == 0 for b in own_b)  # b's private copies were freed
    assert blocks.num_free == free_before + 3
    assert cache.num_blocks == 3
    check(cache)


def test_catch_up_replaces_a_partial_block_and_extends():
    cache, blocks = make()
    tokens = toks((1, 4))
    compute(cache, "leader", tokens)
    # The follower has computed 6 tokens on its own: one full block and half of the next.
    blocks.ensure_capacity("follower", 6)
    own = list(blocks.table("follower"))
    cached = cache.sync_sequence("follower", [*tokens, 0], 6)
    assert cached == 16
    assert blocks.table("follower") == blocks.table("leader")
    assert all(blocks.ref_count(b) == 0 for b in own)
    check(cache)


# --------------------------------------------------------------------- eviction / ref counts


def test_blocks_in_use_are_not_evicted():
    cache, blocks = make()
    compute(cache, "a", toks((1, 3)))
    assert blocks.num_evictable == 0
    assert cache.evict(10) == 0
    assert cache.num_blocks == 3
    blocks.free("a")
    assert blocks.num_evictable == 3
    assert all(blocks.ref_count(b) == 1 for node in cache.nodes() for b in node.blocks)


def test_evict_trims_the_tail_of_the_lru_leaf():
    cache, blocks = make()
    tokens = toks((1, 2), (2, 2))
    compute(cache, "a", tokens)
    table = list(blocks.table("a"))
    blocks.free("a")
    free_before = blocks.num_free
    assert cache.evict(1) == 1
    assert blocks.num_free == free_before + 1 and blocks.ref_count(table[3]) == 0
    assert cache.match_length([*tokens, 0]) == 12  # the head of the edge is still usable
    assert cache.num_blocks == 3 and cache.stats.evicted_blocks == 1
    check(cache)
    assert cache.evict(100) == 3
    assert cache.num_nodes == 0 and blocks.num_free == blocks.num_usable
    check(cache)


def test_eviction_order_is_least_recently_used_leaf_first():
    cache, blocks = make()
    old, mid, new = toks((1, 2)), toks((2, 2)), toks((3, 2))
    for name, tokens in (("old", old), ("mid", mid), ("new", new)):
        compute(cache, name, tokens)
        blocks.free(name)
    cache.sync_sequence("reader", [*old, 0], 0)  # "old" becomes the most recently used
    blocks.free("reader")
    assert cache.evict(2) == 2
    assert cache.match_length([*mid, 0]) == 0
    assert cache.match_length([*old, 0]) == 8 and cache.match_length([*new, 0]) == 8
    assert cache.evict(2) == 2
    assert cache.match_length([*new, 0]) == 0 and cache.match_length([*old, 0]) == 8
    check(cache)


def test_shared_prefix_outlives_its_evicted_branches():
    cache, blocks = make()
    for name, tail in (("a", 2), ("b", 3)):
        compute(cache, name, toks((1, 2), (tail, 2)))
        blocks.free(name)
    assert cache.num_nodes == 3
    assert cache.evict(4) == 4  # both branches go before the shared head
    (head,) = cache.nodes()
    assert head.tokens == tuple(toks((1, 2)))
    assert cache.evict(2) == 2  # the head became a leaf and is now evictable
    assert cache.num_nodes == 0
    check(cache)


def test_pinned_leaf_is_skipped_and_stays_evictable_later():
    cache, blocks = make()
    compute(cache, "pinned", toks((1, 2)))
    compute(cache, "loose", toks((2, 2)))
    blocks.free("loose")
    assert cache.evict(4) == 2  # only the unpinned leaf
    assert cache.match_length([*toks((1, 2)), 0]) == 8
    blocks.free("pinned")
    assert cache.evict(4) == 2
    assert blocks.num_free == blocks.num_usable


def test_partially_pinned_edge_is_trimmed_to_the_pinned_part():
    cache, blocks = make()
    tokens = toks((1, 4))
    compute(cache, "a", tokens)
    blocks.free("a")
    cache.sync_sequence("b", [*tokens[:8], 9], 0)  # pins the first two blocks only
    assert blocks.num_evictable == 2
    assert cache.evict(4) == 2
    assert cache.match_length([*tokens, 0]) == 8
    check(cache)


def test_reference_counts_through_a_full_lifecycle():
    cache, blocks = make()
    tokens = toks((1, 2))
    compute(cache, "a", tokens)
    table = list(blocks.table("a"))
    cache.sync_sequence("b", [*tokens, 0], 0)
    cache.sync_sequence("c", [*tokens, 0], 0)
    assert [blocks.ref_count(b) for b in table] == [4, 4]  # a, b, c and the cache
    for name in "abc":
        blocks.free(name)
    assert [blocks.ref_count(b) for b in table] == [1, 1]
    assert blocks.num_evictable == 2 and blocks.num_free == blocks.num_usable - 2
    blocks.check_invariants(external_refs=cache.block_refs())
    assert cache.clear() == 2
    assert blocks.num_free == blocks.num_usable


def test_stats_and_heap_compaction():
    cache, blocks = make()
    tokens = toks((1, 2))
    compute(cache, "a", tokens)
    for i in range(2000):  # every lookup leaves a stale heap entry behind
        cache.sync_sequence(("r", i), [*tokens, 0], 0)
        blocks.free(("r", i))
    assert len(cache._heap) <= 64 + 8 * cache.num_nodes + 1
    assert cache.stats.inserted_blocks == 2


# ------------------------------------------------------------------------------ randomized


def test_randomized_against_a_brute_force_model():
    """Random sessions with shared prefixes; matches are checked against plain lists."""
    rng = random.Random(0)
    cache, blocks = make(num_blocks=160)
    stems = [[rng.randrange(50) for _ in range(rng.randint(1, 6) * BLOCK)] for _ in range(4)]
    known: list[list[int]] = []  # block-aligned token prefixes that have been inserted
    live: dict[int, list[int]] = {}

    def expected_match(tokens: list[int]) -> int:
        limit = (len(tokens) - 1) // BLOCK
        best = 0
        for prefix in known:
            n = 0
            while (
                n < limit
                and (n + 1) * BLOCK <= len(prefix)
                and prefix[n * BLOCK : (n + 1) * BLOCK] == tokens[n * BLOCK : (n + 1) * BLOCK]
            ):
                n += 1
            best = max(best, n)
        return best * BLOCK

    evictions = 0
    for step in range(600):
        roll = rng.random()
        if roll < 0.5:
            base = rng.choice(stems + list(live.values())) if live else rng.choice(stems)
            tokens = base[: rng.randint(1, len(base))] + [
                rng.randrange(50) for _ in range(rng.randint(0, 3 * BLOCK))
            ]
            needed = blocks.blocks_for(len(tokens))
            if blocks.num_free < needed:
                freed = cache.evict(needed - blocks.num_free)
                evictions += freed
                if freed:
                    known.clear()  # the model is rebuilt from what is still in the tree
                    for node in cache.nodes():
                        path, cursor = [], node
                        while cursor is not cache.root:
                            path = list(cursor.tokens) + path
                            cursor = cursor.parent
                        known.append(path)
                if blocks.num_free < needed:
                    continue
            assert cache.match_length(tokens) == expected_match(tokens)
            cached = cache.sync_sequence(step, tokens, 0)
            assert cached == expected_match(tokens)
            blocks.ensure_capacity(step, len(tokens))
            cache.sync_sequence(step, [*tokens, -1], len(tokens))
            known.append(tokens[: len(tokens) // BLOCK * BLOCK])
            live[step] = tokens
        elif live:
            victim = rng.choice(list(live))
            blocks.free(victim)
            del live[victim]
        cache.check_invariants()
        blocks.check_invariants(external_refs=cache.block_refs())
    assert evictions > 0 and cache.num_nodes > 4
    for seq_id in live:
        blocks.free(seq_id)
    cache.clear()
    assert blocks.num_free == blocks.num_usable and cache.num_nodes == 0
