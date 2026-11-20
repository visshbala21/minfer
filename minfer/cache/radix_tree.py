"""Radix-tree prefix cache: token prefixes mapped to shared, reference-counted KV blocks.

Agent workloads resend the same tokens over and over: every request of an agent type starts
with the same system prompt and tool schemas, and every turn of a session starts with the
whole previous turn. The keys and values of a token depend only on the tokens up to it, so
they can be computed once and shared by every request with the same prefix.

The tree is keyed on token sequences. Each node owns one edge: a run of tokens and the KV
blocks holding them. Edges are *block aligned* — an edge is a whole number of blocks —
because a block is the unit the page tables can share; a prefix is therefore reused up to
its last full block, and at most ``block_size - 1`` matching tokens are recomputed.
Children are indexed by the tokens of their first block, so descending the tree costs one
dictionary lookup per node plus a comparison along the edge.

The cache holds one reference to every block in the tree (see
:meth:`BlockManager.cache_block`); sequences that use a cached prefix hold their own
references through their block tables. A block referenced only by the cache is evictable.
Eviction removes least-recently-used leaves, from the tail of the edge backwards — the
tokens furthest from the root are the least likely to be shared.
"""

from __future__ import annotations

import heapq
from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass
from itertools import count

from minfer.cache.block_manager import BlockManager


class RadixNode:
    """A tree node and the edge leading to it.

    Attributes:
        tokens: The edge's tokens; ``len(tokens) == len(blocks) * block_size``.
        blocks: KV blocks holding the edge's tokens, in position order.
        children: Child nodes, keyed by the tokens of their edge's first block.
        parent: The parent node; ``None`` for the root and for nodes removed from the tree.
        last_access: Logical time the node was last matched or inserted.
    """

    __slots__ = ("blocks", "children", "last_access", "parent", "tokens")

    def __init__(self, tokens: tuple[int, ...], blocks: list[int], parent: RadixNode | None):
        self.tokens = tokens
        self.blocks = blocks
        self.children: dict[tuple[int, ...], RadixNode] = {}
        self.parent = parent
        self.last_access = 0


@dataclass
class PrefixCacheStats:
    """Monotonic counters of prefix-cache activity.

    ``lookup_tokens``/``hit_tokens`` are counted when a sequence is admitted: how many
    tokens it needed in the cache, and how many it found there. ``catchup_tokens`` are
    tokens a sequence adopted later, while still prefilling, because another sequence
    computed them first. Both kinds are prefill work that was not done.
    """

    lookups: int = 0
    hits: int = 0
    lookup_tokens: int = 0
    hit_tokens: int = 0
    catchup_tokens: int = 0
    inserted_blocks: int = 0
    evicted_blocks: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of looked-up tokens served from the cache."""
        return self.hit_tokens / self.lookup_tokens if self.lookup_tokens else 0.0

    @property
    def prefill_tokens_saved(self) -> int:
        """Tokens whose KV was reused instead of computed."""
        return self.hit_tokens + self.catchup_tokens


class PrefixCache:
    """Radix tree over token sequences whose nodes hold shared KV blocks."""

    def __init__(self, blocks: BlockManager) -> None:
        self.blocks = blocks
        self.block_size = blocks.block_size
        self.root = RadixNode((), [], None)
        self.num_nodes = 0
        self.num_blocks = 0
        self.stats = PrefixCacheStats()
        self._clock = 0
        # Min-heap of (last_access, tiebreak, node) with lazy invalidation: an entry is
        # current only if the node is still a leaf of the tree with that access time.
        self._heap: list[tuple[int, int, RadixNode]] = []
        self._tiebreak = count()

    # ------------------------------------------------------------------ walk

    def _matching_blocks(
        self, node: RadixNode, tokens: Sequence[int], start: int, limit: int
    ) -> int:
        """Leading blocks of ``node``'s edge equal to ``tokens`` from block ``start``.

        At most ``limit`` blocks are compared. The first block is known to match (it is the
        key the node was found under).
        """
        size = self.block_size
        span = min(len(node.blocks), limit)
        begin = start * size
        if tuple(tokens[begin : begin + span * size]) == node.tokens[: span * size]:
            return span
        edge = node.tokens
        matched = 1
        while matched < span:
            lo = matched * size
            if tuple(tokens[begin + lo : begin + lo + size]) != edge[lo : lo + size]:
                break
            matched += 1
        return matched

    def match_length(self, tokens: Sequence[int]) -> int:
        """Tokens of ``tokens`` a new sequence could take from the cache right now.

        Read-only (does not count as a use for LRU). At least one token is always left
        uncached, because a sequence needs a pending token to produce logits from.
        """
        size = self.block_size
        limit = (len(tokens) - 1) // size
        node, index = self.root, 0
        while index < limit:
            child = node.children.get(tuple(tokens[index * size : (index + 1) * size]))
            if child is None:
                break
            matched = self._matching_blocks(child, tokens, index, limit - index)
            index += matched
            if matched < len(child.blocks):
                break
            node = child
        return index * size

    def sync(self, tokens: Sequence[int], own_blocks: Sequence[int], limit: int) -> list[int]:
        """Reconcile a sequence's leading blocks with the tree, in both directions.

        Walks the tree along ``tokens``. Where the tree already holds a block for a
        position, that block is the canonical one; where it does not and the sequence has
        computed one (``own_blocks``), the sequence's block is inserted. The walk continues
        past the sequence's own blocks for as long as the tree has matching ones.

        Args:
            tokens: The sequence's tokens.
            own_blocks: Blocks the sequence has completely filled, in position order.
            limit: Maximum number of leading blocks to return.

        Returns:
            The canonical blocks of the longest prefix (≤ ``limit`` blocks) that is in the
            tree after the call. It covers at least ``own_blocks[:limit]``. The caller must
            point its table at these blocks (:meth:`BlockManager.adopt`).
        """
        size = self.block_size
        num_own = min(len(own_blocks), limit)
        self._clock += 1
        canonical: list[int] = []
        node, index = self.root, 0
        while index < limit:
            key = tuple(tokens[index * size : (index + 1) * size])
            child = node.children.get(key)
            if child is None:
                if index < num_own:
                    grow = self._extend_leaf if self._is_leaf(node) else self._add_child
                    canonical.extend(grow(node, tokens, own_blocks, index, num_own))
                break
            matched = self._matching_blocks(child, tokens, index, limit - index)
            if matched < len(child.blocks) and index + matched < num_own:
                # The sequence diverges inside the edge and has blocks of its own to hang
                # below the divergence point.
                self._split(child, matched)
            self._touch(child)
            canonical.extend(child.blocks[:matched])
            index += matched
            if matched < len(child.blocks):
                break
            node = child
        return canonical

    def _is_leaf(self, node: RadixNode) -> bool:
        return node is not self.root and not node.children

    def _add_child(
        self,
        parent: RadixNode,
        tokens: Sequence[int],
        own_blocks: Sequence[int],
        start: int,
        stop: int,
    ) -> list[int]:
        """Hang ``own_blocks[start:stop]`` below ``parent`` as a new leaf."""
        size = self.block_size
        new_blocks = list(own_blocks[start:stop])
        child = RadixNode(tuple(tokens[start * size : stop * size]), new_blocks, parent)
        parent.children[child.tokens[:size]] = child
        self.num_nodes += 1
        self._account_insert(child, new_blocks)
        return new_blocks

    def _extend_leaf(
        self,
        leaf: RadixNode,
        tokens: Sequence[int],
        own_blocks: Sequence[int],
        start: int,
        stop: int,
    ) -> list[int]:
        """Append ``own_blocks[start:stop]`` to a leaf's edge.

        A sequence that keeps generating would otherwise grow a chain of one-block nodes;
        lengthening the leaf keeps the depth of the tree proportional to the number of
        *divergences*, not to the number of tokens.
        """
        size = self.block_size
        new_blocks = list(own_blocks[start:stop])
        leaf.tokens += tuple(tokens[start * size : stop * size])
        leaf.blocks.extend(new_blocks)
        self._account_insert(leaf, new_blocks)
        return new_blocks

    def _account_insert(self, node: RadixNode, new_blocks: list[int]) -> None:
        for block in new_blocks:
            self.blocks.cache_block(block)
        self.num_blocks += len(new_blocks)
        self.stats.inserted_blocks += len(new_blocks)
        self._touch(node)

    def _split(self, node: RadixNode, num_blocks: int) -> None:
        """Cut ``node``'s edge after ``num_blocks`` blocks.

        ``node`` keeps the head of the edge (and its place under its parent); a new node
        takes the tail and all of ``node``'s children.
        """
        cut = num_blocks * self.block_size
        tail = RadixNode(node.tokens[cut:], node.blocks[num_blocks:], node)
        tail.children = node.children
        for grandchild in tail.children.values():
            grandchild.parent = tail
        tail.last_access = node.last_access
        node.tokens = node.tokens[:cut]
        node.blocks = node.blocks[:num_blocks]
        node.children = {tail.tokens[: self.block_size]: tail}
        self.num_nodes += 1
        if not tail.children:
            heapq.heappush(self._heap, (tail.last_access, next(self._tiebreak), tail))

    def _touch(self, node: RadixNode) -> None:
        """Record a use of ``node``."""
        node.last_access = self._clock
        if not node.children:
            heapq.heappush(self._heap, (self._clock, next(self._tiebreak), node))
            if len(self._heap) > 64 + 8 * self.num_nodes:
                self._rebuild_heap()

    def _rebuild_heap(self) -> None:
        """Drop stale heap entries (every touch of a leaf leaves its older entry behind)."""
        self._heap = [
            (node.last_access, next(self._tiebreak), node)
            for node in self.nodes()
            if not node.children
        ]
        heapq.heapify(self._heap)

    # ------------------------------------------------------------- sequences

    def sync_sequence(self, seq_id: Hashable, tokens: Sequence[int], num_cached: int) -> int:
        """Share a sequence's full blocks through the tree and adopt what the tree has.

        Inserts the blocks the sequence has filled, replaces any of its blocks the tree
        already had (so one physical copy survives) and extends its table with further
        cached blocks that match its tokens.

        Returns:
            The sequence's new ``num_cached`` (never less than the value passed in, and
            always leaving at least one token pending).
        """
        size = self.block_size
        limit = (len(tokens) - 1) // size
        if limit == 0:
            return num_cached
        own = self.blocks.table(seq_id)[: num_cached // size]
        canonical = self.sync(tokens, own, limit)
        self.blocks.adopt(seq_id, canonical)
        return max(num_cached, len(canonical) * size)

    # -------------------------------------------------------------- eviction

    def evict(self, num_blocks: int) -> int:
        """Free up to ``num_blocks`` blocks held only by the cache, least recently used first.

        Returns the number of blocks freed, which is less than requested when everything
        left in the tree is in use by running sequences.
        """
        blocks = self.blocks
        freed = 0
        in_use = []
        while freed < num_blocks and self._heap:
            entry = heapq.heappop(self._heap)
            access, _, node = entry
            if node.parent is None or node.children or access != node.last_access:
                continue  # stale: removed, no longer a leaf, or touched since
            key = node.tokens[: self.block_size]
            before = freed
            while node.blocks and freed < num_blocks and blocks.ref_count(node.blocks[-1]) == 1:
                blocks.uncache_block(node.blocks.pop())
                freed += 1
            if freed > before:
                node.tokens = node.tokens[: len(node.blocks) * self.block_size]
            if node.blocks:
                # Either enough was freed, or the rest is pinned by a running sequence.
                in_use.append(entry)
                continue
            parent = node.parent
            del parent.children[key]
            node.parent = None
            self.num_nodes -= 1
            if parent is not self.root and not parent.children:
                heapq.heappush(self._heap, (parent.last_access, next(self._tiebreak), parent))
        for entry in in_use:
            heapq.heappush(self._heap, entry)
        self.num_blocks -= freed
        self.stats.evicted_blocks += freed
        return freed

    def clear(self) -> int:
        """Evict everything that is not in use; return the number of blocks freed."""
        return self.evict(self.num_blocks)

    # ------------------------------------------------------------ inspection

    def nodes(self) -> Iterator[RadixNode]:
        """Every node except the root, parents before children."""
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def block_refs(self) -> dict[int, int]:
        """References the cache holds, as ``block -> 1`` (for invariant checks)."""
        return {block: 1 for node in self.nodes() for block in node.blocks}

    def check_invariants(self) -> None:
        """Assert the tree's structural invariants."""
        size = self.block_size
        seen: set[int] = set()
        num_nodes = 0
        for node in self.nodes():
            num_nodes += 1
            assert node.blocks, "empty node left in the tree"
            assert len(node.tokens) == len(node.blocks) * size, "edge is not block aligned"
            assert node.parent is not None
            assert node.parent.children[node.tokens[:size]] is node, "node filed under wrong key"
            assert node.last_access <= node.parent.last_access or node.parent is self.root, (
                "a child was used more recently than its parent"
            )
            for block in node.blocks:
                assert block not in seen, f"block {block} appears twice in the tree"
                assert self.blocks.is_cached(block), f"tree block {block} is not flagged cached"
                seen.add(block)
        assert num_nodes == self.num_nodes, f"{num_nodes} nodes, counter says {self.num_nodes}"
        assert len(seen) == self.num_blocks, f"{len(seen)} blocks, counter says {self.num_blocks}"
