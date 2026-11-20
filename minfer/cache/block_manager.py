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

    Invariants (checked by :meth:`check_invariants`, exercised in the tests):
      * a block is on the free list iff its reference count is zero;
      * every block in a sequence's table holds at least one reference on that table's behalf.
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
        for block in blocks:
            if self._ref_counts[block] <= 0:
                raise ValueError(f"cannot retain free block {block}")
            self._ref_counts[block] += 1

    def release(self, blocks: Iterable[int]) -> int:
        """Drop one reference from each block; return how many became free."""
        freed = 0
        for block in blocks:
            count = self._ref_counts[block]
            if count <= 0:
                raise ValueError(f"block {block} released more times than retained")
            self._ref_counts[block] = count - 1
            if count == 1:
                self._free.append(block)
                freed += 1
        return freed

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
        """Start a sequence's table with already-populated blocks, sharing them.

        Used on a prefix-cache hit: the table takes its own reference on each block.
        """
        if self._tables.get(seq_id):
            raise ValueError(f"sequence {seq_id!r} already has a block table")
        blocks = list(blocks)
        self.retain(blocks)
        self._tables[seq_id] = blocks

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

    def replace_block(self, seq_id: Hashable, index: int, block: int) -> int:
        """Swap entry ``index`` of the table for ``block``; return the block it replaced.

        The caller transfers its reference on ``block`` to the table and receives the
        table's reference on the old block (copy-on-write of a shared block).
        """
        table = self._tables[seq_id]
        old = table[index]
        table[index] = block
        return old

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
