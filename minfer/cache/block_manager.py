"""Fixed-size KV blocks: allocation, reference counting, and per-sequence block tables.

A *block* is an index into every layer's KV tensor and holds the keys and values of
``block_size`` consecutive token positions. A sequence's KV is described by its *block
table*: the list of blocks backing positions ``[0, block_size)``, ``[block_size,
2·block_size)``, and so on. Blocks are reference counted because they can be shared: by the
prefix cache, and by several sequences that start with the same tokens.

Everything here is plain Python over integers; the tensors themselves live in
:class:`minfer.cache.kv_store.PagedKVStore`.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable


class OutOfBlocks(RuntimeError):
    """Raised when an allocation asks for more blocks than are free."""


def kv_bytes_per_block(
    num_layers: int, num_kv_heads: int, head_dim: int, block_size: int, itemsize: int
) -> int:
    """Bytes of KV storage one block occupies across all layers (keys and values)."""
    return num_layers * 2 * block_size * num_kv_heads * head_dim * itemsize


def plan_num_blocks(free_bytes: int, bytes_per_block: int, fraction: float) -> int:
    """Number of blocks that fit in ``fraction`` of the currently free device memory.

    ``fraction`` leaves the remainder for activations, attention workspaces and CUDA-graph
    memory pools, none of which exist yet when the cache is sized.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    return int(free_bytes * fraction) // bytes_per_block


class BlockManager:
    """Reference-counted pool of KV blocks plus the block table of every live sequence.

    The lowest ``num_reserved`` block ids are never handed out. Block 0 is reserved so that
    fixed-shape (padded) batches have a harmless block to read from and write to.

    A block can additionally be *cached*: the prefix cache holds one reference to it and
    flags it here. A cached block with exactly that one reference is *evictable* — nothing
    running uses it, so the cache can give it back. The manager keeps the evictable count up
    to date on every retain/release, so admission can treat evictable blocks as available
    memory without scanning the cache.

    Invariants (checked by :meth:`check_invariants`, exercised in the tests):
      * a block is on the free list iff its reference count is zero;
      * every block in a sequence's table holds at least one reference on that table's behalf;
      * ``num_evictable`` equals the number of cached blocks whose reference count is one.
    """

    def __init__(self, num_blocks: int, block_size: int = 16, num_reserved: int = 1) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_blocks <= num_reserved:
            raise ValueError(f"need more than {num_reserved} blocks, got {num_blocks}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_reserved = num_reserved
        self._ref_counts = [0] * num_blocks
        self._cached = [False] * num_blocks
        self._num_evictable = 0
        # LIFO free list; popping from the end hands out low ids first.
        self._free = list(range(num_blocks - 1, num_reserved - 1, -1))
        self._tables: dict[Hashable, list[int]] = {}

    # ------------------------------------------------------------------ pool

    @property
    def num_free(self) -> int:
        """Blocks available for allocation right now."""
        return len(self._free)

    @property
    def num_usable(self) -> int:
        """Blocks that can ever be allocated (total minus reserved)."""
        return self.num_blocks - self.num_reserved

    @property
    def num_used(self) -> int:
        """Blocks currently referenced by at least one owner."""
        return self.num_usable - len(self._free)

    @property
    def num_evictable(self) -> int:
        """Cached blocks no sequence references: memory the prefix cache can give back."""
        return self._num_evictable

    def blocks_for(self, num_tokens: int) -> int:
        """Number of blocks needed to hold ``num_tokens`` positions."""
        return -(-num_tokens // self.block_size)

    def allocate(self, count: int) -> list[int]:
        """Take ``count`` free blocks, each with a reference count of one."""
        if count > len(self._free):
            raise OutOfBlocks(f"requested {count} blocks, {len(self._free)} free")
        blocks = [self._free.pop() for _ in range(count)]
        for block in blocks:
            self._ref_counts[block] = 1
        return blocks

    def retain(self, blocks: Iterable[int]) -> None:
        """Add one reference to each block (it must already be in use)."""
        counts, cached = self._ref_counts, self._cached
        for block in blocks:
            count = counts[block]
            if count <= 0:
                raise ValueError(f"cannot retain free block {block}")
            counts[block] = count + 1
            if count == 1 and cached[block]:
                self._num_evictable -= 1

    def release(self, blocks: Iterable[int]) -> int:
        """Drop one reference from each block; return how many became free."""
        counts, cached = self._ref_counts, self._cached
        freed = 0
        for block in blocks:
            count = counts[block]
            if count <= 0:
                raise ValueError(f"block {block} released more times than retained")
            counts[block] = count - 1
            if count == 1:
                if cached[block]:
                    raise ValueError(f"cached block {block} lost the cache's reference")
                self._free.append(block)
                freed += 1
            elif count == 2 and cached[block]:
                self._num_evictable += 1
        return freed

    def cache_block(self, block: int) -> None:
        """Give the prefix cache a reference to an in-use block and flag it as cached."""
        if self._cached[block]:
            raise ValueError(f"block {block} is already cached")
        self.retain((block,))
        self._cached[block] = True

    def uncache_block(self, block: int) -> None:
        """Drop the prefix cache's reference; the block is freed if nothing else holds it."""
        if not self._cached[block]:
            raise ValueError(f"block {block} is not cached")
        if self._ref_counts[block] == 1:
            self._num_evictable -= 1
        self._cached[block] = False
        self.release((block,))

    def is_cached(self, block: int) -> bool:
        """Whether the prefix cache holds a reference to ``block``."""
        return self._cached[block]

    def ref_count(self, block: int) -> int:
        """Current reference count of ``block``."""
        return self._ref_counts[block]

    # ---------------------------------------------------------------- tables

    def table(self, seq_id: Hashable) -> list[int]:
        """The block table of a sequence (empty if it holds no blocks)."""
        return self._tables.get(seq_id, [])

    def capacity(self, seq_id: Hashable) -> int:
        """Token positions the sequence's current table can hold."""
        return len(self.table(seq_id)) * self.block_size

    def adopt(self, seq_id: Hashable, blocks: Iterable[int]) -> None:
        """Make ``blocks`` the first entries of the sequence's table, sharing them.

        Used when the prefix cache already holds the keys/values of the sequence's leading
        positions. The table takes its own reference on every block it newly points to and
        drops its reference on any block of its own that is displaced, so a sequence can
        adopt a longer cached prefix at any time, not only when its table is empty.
        """
        table = self._tables.setdefault(seq_id, [])
        displaced = []
        for index, block in enumerate(blocks):
            if index < len(table):
                if table[index] == block:
                    continue
                displaced.append(table[index])
                self.retain((block,))
                table[index] = block
            else:
                self.retain((block,))
                table.append(block)
        self.release(displaced)

    def blocks_missing(self, seq_id: Hashable, num_tokens: int) -> int:
        """How many more blocks the sequence needs to hold ``num_tokens`` positions."""
        return max(0, self.blocks_for(num_tokens) - len(self.table(seq_id)))

    def ensure_capacity(self, seq_id: Hashable, num_tokens: int) -> list[int]:
        """Grow the sequence's table to hold ``num_tokens`` positions.

        Returns the newly allocated blocks. Raises :class:`OutOfBlocks` without changing
        anything if the pool cannot cover the growth.
        """
        missing = self.blocks_missing(seq_id, num_tokens)
        if missing == 0:
            return []
        new_blocks = self.allocate(missing)
        self._tables.setdefault(seq_id, []).extend(new_blocks)
        return new_blocks

    def free(self, seq_id: Hashable) -> int:
        """Drop the sequence's table; return how many blocks returned to the pool."""
        blocks = self._tables.pop(seq_id, None)
        return self.release(blocks) if blocks else 0

    # ------------------------------------------------------------ validation

    def check_invariants(self, external_refs: dict[int, int] | None = None) -> None:
        """Assert internal consistency.

        Args:
            external_refs: references held outside the tables (e.g. by the prefix cache),
                as ``block -> count``. With it, reference counts are checked exactly.
        """
        free = set(self._free)
        assert len(free) == len(self._free), "free list has duplicates"
        assert all(b >= self.num_reserved for b in free), "reserved block on free list"
        expected = [0] * self.num_blocks
        for table in self._tables.values():
            for block in table:
                expected[block] += 1
        for block, count in (external_refs or {}).items():
            expected[block] += count
        for block in range(self.num_reserved, self.num_blocks):
            count = self._ref_counts[block]
            assert (block in free) == (count == 0), f"block {block}: count {count} vs free list"
            if external_refs is not None:
                assert count == expected[block], f"block {block}: {count} != {expected[block]}"
            else:
                assert count >= expected[block], f"block {block}: {count} < {expected[block]}"
            assert not self._cached[block] or count >= 1, f"cached block {block} is free"
        evictable = sum(
            1
            for block in range(self.num_blocks)
            if self._cached[block] and self._ref_counts[block] == 1
        )
        assert evictable == self._num_evictable, f"{evictable} != {self._num_evictable}"
