"""Host-memory tier for the KV cache: pinned CPU memory behind the GPU block pool.

Recomputing a paused session's history costs a prefill pass over tens of thousands of
tokens; copying the same keys and values back from host memory costs a PCIe transfer
(about a megabyte per 16-token block for a 7B model). This module is the mechanism: a pool
of pinned host memory laid out in the same blocks as the GPU store, and asynchronous copies
in both directions. Policy — what to offload, what to bring back, when recomputing is
cheaper — lives with the prefix cache and the scheduler.

**Layout.** The GPU store is layer-major (``[layers, blocks, ...]``, what the attention
kernels index). The host pool is block-major (``[host_blocks, layers, ...]``) so that a run
of consecutive host blocks is one contiguous range of pinned memory, which is what an
asynchronous device copy needs; host blocks are therefore allocated as extents.

**Asynchrony.** Transfers run on their own CUDA stream. An offload gathers the blocks into a
staging tensor on the engine's stream (ordered after every write to them), then copies the
staging tensor to the host on the transfer stream; the GPU blocks may be reused as soon as
the gather is queued. A restore copies host blocks into a staging tensor on the transfer
stream — queued behind any offload of the same blocks still in flight — and the engine
scatters it into the store once the copy's event has completed. The engine does not wait on
a transfer — it polls — except to keep the GPU staging memory of the copies in flight under
``max_staging_bytes``: a burst of offloads (many sessions pausing at once, each with
gigabytes of KV) would otherwise take the memory the engine's own passes need (DESIGN.md
B-013). Then the oldest copies are waited for and finalized first.

On a CPU store (tests) the same interface copies synchronously.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch
from torch import Tensor

from minfer.cache.kv_store import FP8, PagedKVStore


class HostPoolFull(RuntimeError):
    """Raised when the host pool cannot hold an offload."""


class ExtentAllocator:
    """First-fit allocator of runs of consecutive slots, with coalescing on free."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.num_free = size
        self._free: list[tuple[int, int]] = [(0, size)] if size else []  # (start, length), sorted

    def allocate(self, count: int) -> list[tuple[int, int]]:
        """Take ``count`` slots as few extents as first-fit finds; all or nothing."""
        if count > self.num_free:
            raise HostPoolFull(f"requested {count} host blocks, {self.num_free} free")
        taken: list[tuple[int, int]] = []
        remaining = count
        kept: list[tuple[int, int]] = []
        for start, length in self._free:
            if remaining == 0:
                kept.append((start, length))
                continue
            use = min(length, remaining)
            taken.append((start, use))
            remaining -= use
            if use < length:
                kept.append((start + use, length - use))
        self._free = kept
        self.num_free -= count
        return taken

    def free(self, slots: list[int]) -> None:
        """Return individual slots (in any order) to the pool."""
        if not slots:
            return
        runs: list[tuple[int, int]] = []
        for slot in sorted(slots):
            if runs and runs[-1][0] + runs[-1][1] == slot:
                runs[-1] = (runs[-1][0], runs[-1][1] + 1)
            else:
                runs.append((slot, 1))
        merged: list[tuple[int, int]] = []
        for start, length in sorted([*self._free, *runs]):
            if merged and merged[-1][0] + merged[-1][1] >= start:
                if merged[-1][0] + merged[-1][1] > start:
                    raise ValueError(f"host slot {start} freed twice")
                merged[-1] = (merged[-1][0], merged[-1][1] + length)
            else:
                merged.append((start, length))
        self._free = merged
        self.num_free += len(slots)


@dataclass
class _Copy:
    """One chunked device copy: its staging tensors live until it is finalized."""

    event: torch.cuda.Event | None
    staged: list[Tensor] | None
    gpu_blocks: list[int] | None  # restore target; None for an offload
    nbytes: int
    finalized: bool = False


@dataclass
class Transfer:
    """An offload or restore in flight.

    Attributes:
        num_blocks: Blocks being copied.
        pending: Copies not yet popped by :meth:`HostTier.poll` (some may already have
            been finalized early to free staging memory).
        done: Set by :meth:`HostTier.poll` once every copy has completed (and, for a
            restore, been written into the GPU store).
    """

    num_blocks: int
    pending: deque = field(default_factory=deque)
    done: bool = False


@dataclass
class HostTierStats:
    """Monotonic counters of host-tier activity."""

    offloaded_blocks: int = 0
    restored_blocks: int = 0
    dropped_blocks: int = 0
    staging_waits: int = 0  # copies waited for because staging memory hit its cap
    peak_staging_bytes: int = 0


class HostTier:
    """Pinned host memory for KV blocks, with asynchronous copies to and from the GPU.

    Args:
        store: The GPU KV store this tier backs.
        num_blocks: Capacity in blocks.
        chunk_blocks: Largest number of blocks copied by one device copy; bounds the size
            of a staging tensor on the GPU.
        max_staging_bytes: Cap on the GPU staging memory of all copies in flight. One
            chunk may exceed it alone; past it, the oldest copies are waited for.
    """

    def __init__(
        self,
        store: PagedKVStore,
        num_blocks: int,
        chunk_blocks: int = 256,
        max_staging_bytes: int = 1 << 30,
    ) -> None:
        self.store = store
        self.num_blocks = num_blocks
        self.chunk_blocks = chunk_blocks
        self.max_staging_bytes = max_staging_bytes
        self.cuda = store.device.type == "cuda"
        self.allocator = ExtentAllocator(num_blocks)
        self.stats = HostTierStats()
        # One pool per storage dtype of the store; FP8 bytes are handled as uint8, which is
        # what index operations support.
        self._gpu: list[Tensor] = []
        self._host: list[Tensor] = []
        for dtype, tensor in store.tensors.items():
            raw = tensor.view(torch.uint8) if dtype == FP8 else tensor
            shape = (num_blocks, raw.shape[0], *raw.shape[2:])
            self._gpu.append(raw)
            self._host.append(torch.empty(shape, dtype=raw.dtype, pin_memory=self.cuda))
        self.bytes_per_block = sum(t[0].numel() * t.element_size() for t in self._host)
        self._stream = torch.cuda.Stream(store.device) if self.cuda else None
        self._transfers: deque[Transfer] = deque()
        self._copies: deque[_Copy] = deque()  # every unfinalized copy, oldest first
        self.staging_bytes = 0

    @property
    def num_free(self) -> int:
        """Host blocks available."""
        return self.allocator.num_free

    @property
    def busy(self) -> bool:
        """Whether any copy is still in flight (or completed and not yet polled)."""
        return bool(self._transfers)

    # ------------------------------------------------------------- transfers

    def offload(self, gpu_blocks: list[int]) -> tuple[list[int], Transfer]:
        """Start copying ``gpu_blocks`` to the host; return their host slots and the transfer.

        The slots are usable as a restore source immediately (a restore queues behind this
        copy), and the GPU blocks may be freed or overwritten as soon as this returns.

        Raises:
            HostPoolFull: If the pool cannot hold them; nothing is changed.
        """
        extents = self.allocator.allocate(len(gpu_blocks))
        transfer = Transfer(len(gpu_blocks))
        slots: list[int] = []
        cursor = 0
        for start, length in extents:
            slots.extend(range(start, start + length))
            for offset in range(0, length, self.chunk_blocks):
                count = min(self.chunk_blocks, length - offset)
                chunk = gpu_blocks[cursor : cursor + count]
                cursor += count
                self._make_room(count)
                index = torch.tensor(chunk, device=self.store.device)
                staged = [
                    gpu.index_select(1, index).transpose(0, 1).contiguous() for gpu in self._gpu
                ]
                event = self._copy(staged, start + offset, to_host=True)
                transfer.pending.append(self._track(event, staged, None, count))
        self.stats.offloaded_blocks += len(gpu_blocks)
        self._transfers.append(transfer)
        return slots, transfer

    def restore(self, host_slots: list[int], gpu_blocks: list[int]) -> Transfer:
        """Start copying ``host_slots`` back into ``gpu_blocks`` (same length, same order).

        The GPU blocks hold valid KV only once the returned transfer is ``done``.
        """
        transfer = Transfer(len(host_slots))
        position = 0
        while position < len(host_slots):
            start = host_slots[position]
            count = 1
            while (
                count < self.chunk_blocks
                and position + count < len(host_slots)
                and host_slots[position + count] == start + count
            ):
                count += 1
            self._make_room(count)
            staged, event = self._copy_from_host(start, count)
            targets = gpu_blocks[position : position + count]
            transfer.pending.append(self._track(event, staged, targets, count))
            position += count
        self.stats.restored_blocks += len(host_slots)
        self._transfers.append(transfer)
        return transfer

    def _track(
        self,
        event: torch.cuda.Event | None,
        staged: list[Tensor],
        gpu_blocks: list[int] | None,
        count: int,
    ) -> _Copy:
        copy = _Copy(event, staged, gpu_blocks, count * self.bytes_per_block)
        self._copies.append(copy)
        self.staging_bytes += copy.nbytes
        self.stats.peak_staging_bytes = max(self.stats.peak_staging_bytes, self.staging_bytes)
        return copy

    def _finalize(self, copy: _Copy) -> None:
        """Scatter a completed restore into the store; release the staging memory."""
        if copy.gpu_blocks is not None:
            index = torch.tensor(copy.gpu_blocks, device=self.store.device)
            for gpu, source in zip(self._gpu, copy.staged, strict=True):
                gpu.index_copy_(1, index, source.transpose(0, 1))
        copy.staged = None
        copy.finalized = True
        self.staging_bytes -= copy.nbytes

    def _make_room(self, count: int) -> None:
        """Wait for the oldest copies until a ``count``-block chunk fits under the cap."""
        needed = count * self.bytes_per_block
        while self._copies and self.staging_bytes + needed > self.max_staging_bytes:
            copy = self._copies.popleft()
            if copy.finalized:
                continue
            if copy.event is not None:
                copy.event.synchronize()
            self.stats.staging_waits += 1
            self._finalize(copy)

    def _copy(self, staged: list[Tensor], start: int, to_host: bool) -> torch.cuda.Event | None:
        count = staged[0].shape[0]
        if not self.cuda:
            for host, source in zip(self._host, staged, strict=True):
                host[start : start + count].copy_(source)
            return None
        stream = self._stream
        # The staging tensors are produced on the engine's stream: do not read them early.
        stream.wait_stream(torch.cuda.current_stream(self.store.device))
        with torch.cuda.stream(stream):
            for host, source in zip(self._host, staged, strict=True):
                host[start : start + count].copy_(source, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        return event

    def _copy_from_host(
        self, start: int, count: int
    ) -> tuple[list[Tensor], torch.cuda.Event | None]:
        if not self.cuda:
            return [host[start : start + count].clone() for host in self._host], None
        with torch.cuda.stream(self._stream):
            staged = [
                host[start : start + count].to(self.store.device, non_blocking=True)
                for host in self._host
            ]
            event = torch.cuda.Event()
            event.record(self._stream)
        return staged, event

    def poll(self) -> None:
        """Finalize completed copies; mark transfers whose copies are all complete.

        A restored chunk is scattered into the GPU store here, on the caller's stream.
        Never blocks.
        """
        while self._transfers:
            transfer = self._transfers[0]
            pending = transfer.pending
            while pending:
                copy = pending[0]
                if not copy.finalized:
                    if copy.event is not None and not copy.event.query():
                        return  # copies complete in order on the transfer stream
                    self._finalize(copy)
                pending.popleft()
            transfer.done = True
            self._transfers.popleft()
        while self._copies and self._copies[0].finalized:
            self._copies.popleft()

    def wait(self) -> None:
        """Block until every transfer in flight has completed (tests, shutdown)."""
        if self.cuda:
            self._stream.synchronize()
        self.poll()

    def free(self, host_slots: list[int]) -> None:
        """Give host slots back to the pool."""
        self.allocator.free(host_slots)
        self.stats.dropped_blocks += len(host_slots)
