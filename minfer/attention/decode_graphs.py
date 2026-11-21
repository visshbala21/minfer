"""CUDA graphs of the decode pass at bucketed batch sizes.

A decode step for the 1.5B model launches several hundred small CUDA kernels; at small and
medium batch sizes the time PyTorch spends dispatching them from Python exceeds the time
the GPU spends executing them. A CUDA graph records the kernel launches of one forward
pass once and replays them with a single call.

A graph freezes shapes and tensor addresses, so each bucket owns static input buffers
(token ids, positions, KV slots), a FlashInfer decode wrapper created in CUDA-graph mode
(its page-table buffers are static too, and are refilled by ``plan()`` before each
replay), and a static logits output. A batch of ``n`` sequences runs in the smallest bucket
that holds it; unused rows are padding that reads and writes the reserved block 0.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from minfer.attention.base import AttentionContext
from minfer.attention.flashinfer_backend import (
    KernelContext,
    TritonDecodeContext,
    flashinfer,
    needs_tensor_cores,
)
from minfer.attention.paged import (
    PagedBatch,
    SequenceSpan,
    build_paged_batch,
    dense_block_table,
    paged_layout,
)
from minfer.cache.kv_store import PagedKVStore
from minfer.model.qwen2 import Qwen2Model

DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# A padding row: one token at position 0 whose K/V lands in slot 0 of the reserved block.
_PAD = SequenceSpan(token_ids=(0,), start=0, block_table=(0,))


@dataclass
class _Bucket:
    """Static state of one captured batch size."""

    size: int
    token_ids: Tensor
    positions: Tensor
    slots: Tensor
    wrappers: dict[torch.dtype, flashinfer.BatchDecodeWithPagedKVCacheWrapper] | None
    block_table: Tensor | None = None
    seq_lens: Tensor | None = None
    graph: torch.cuda.CUDAGraph | None = None
    logits: Tensor | None = None


@dataclass
class _Staging:
    """Pinned host buffers for one pass's inputs, and their device-side copies."""

    per_token: Tensor
    tables: Tensor
    indices: Tensor
    device_indices: Tensor
    device_per_token: Tensor
    copied: torch.cuda.Event


class DecodeGraphs:
    """Captured decode passes (model forward + unembedding) for a set of batch sizes.

    Args:
        model: The model whose decode pass is captured.
        store: The paged KV store the pass reads and writes.
        workspace: FlashInfer float workspace buffer (shared with the eager wrappers).
        batch_sizes: Bucket sizes to capture.
        max_pages_per_sequence: Longest block table a sequence can have; sizes the static
            page-index buffers.
    """

    def __init__(
        self,
        model: Qwen2Model,
        store: PagedKVStore,
        workspace: Tensor,
        batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
        max_pages_per_sequence: int = 2048,
        decode_kernel: str = "flashinfer",
    ) -> None:
        self.model = model
        self.store = store
        self.sizes = sorted(set(batch_sizes))
        self.decode_kernel = decode_kernel
        device = model.device
        self._buckets: dict[int, _Bucket] = {}
        for size in self.sizes:
            if decode_kernel == "triton":
                # The Triton kernel reads the block table itself: no planning, just two
                # more static buffers.
                self._buckets[size] = _Bucket(
                    size=size,
                    token_ids=torch.zeros(size, dtype=torch.int64, device=device),
                    positions=torch.zeros(size, dtype=torch.int64, device=device),
                    slots=torch.zeros(size, dtype=torch.int64, device=device),
                    wrappers=None,
                    block_table=torch.zeros(
                        (size, max_pages_per_sequence), dtype=torch.int32, device=device
                    ),
                    seq_lens=torch.ones(size, dtype=torch.int32, device=device),
                )
                continue
            # One graph-mode wrapper per KV storage dtype (two for a mixed-precision store),
            # each with its own static page-table buffers.
            wrappers = {
                kv_dtype: flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    workspace,
                    "NHD",
                    use_cuda_graph=True,
                    use_tensor_cores=needs_tensor_cores(model.num_heads, model.num_kv_heads),
                    paged_kv_indptr_buffer=torch.zeros(size + 1, dtype=torch.int32, device=device),
                    paged_kv_indices_buffer=torch.zeros(
                        size * max_pages_per_sequence, dtype=torch.int32, device=device
                    ),
                    paged_kv_last_page_len_buffer=torch.zeros(
                        size, dtype=torch.int32, device=device
                    ),
                )
                for kv_dtype in store.tensors
            }
            self._buckets[size] = _Bucket(
                size=size,
                token_ids=torch.zeros(size, dtype=torch.int64, device=device),
                positions=torch.zeros(size, dtype=torch.int64, device=device),
                slots=torch.zeros(size, dtype=torch.int64, device=device),
                wrappers=wrappers,
            )
        self._capture_all()
        # Recorded after each pass's inputs and plan are enqueued. FlashInfer's plan()
        # stages its metadata in a pinned host buffer and copies it to the device
        # asynchronously, so a pass may not be planned before the previous pass's copies
        # have executed (only possible when passes are launched ahead: ``run_async``).
        self._inputs_enqueued = torch.cuda.Event()
        # ``run_async``'s inputs are staged in two alternating sets of pinned buffers,
        # allocated once: allocating pinned memory per pass stalled the host for tens of
        # milliseconds whenever the caching host allocator had to grow (DESIGN.md B-010).
        largest = self.sizes[-1]
        self._staging = [
            _Staging(
                per_token=torch.empty(3 * largest, dtype=torch.int64, pin_memory=True),
                tables=torch.empty(2 * largest + 1, dtype=torch.int32, pin_memory=True),
                indices=torch.empty(
                    largest * max_pages_per_sequence, dtype=torch.int32, pin_memory=True
                ),
                device_indices=torch.empty(
                    largest * max_pages_per_sequence, dtype=torch.int32, device=device
                ),
                device_per_token=torch.empty(3 * largest, dtype=torch.int64, device=device),
                copied=torch.cuda.Event(),
            )
            for _ in range(2)
        ]
        self._next_staging = 0

    @property
    def supports_async(self) -> bool:
        """Whether :meth:`run_async` is available (FlashInfer's decode kernel only)."""
        return self.decode_kernel != "triton"

    @property
    def max_batch_size(self) -> int:
        """Largest batch a captured graph can run."""
        return self.sizes[-1]

    def bucket_size(self, batch_size: int) -> int | None:
        """Smallest captured size that holds ``batch_size`` rows, if any."""
        index = bisect.bisect_left(self.sizes, batch_size)
        return self.sizes[index] if index < len(self.sizes) else None

    def _plan(self, bucket: _Bucket, batch: PagedBatch) -> None:
        """Point the bucket's attention at ``batch``'s page tables (host-side work)."""
        if bucket.wrappers is None:
            table, lengths = dense_block_table(batch)
            bucket.block_table[:, : table.shape[1]].copy_(table)
            bucket.seq_lens.copy_(lengths)
            return
        for kv_dtype, wrapper in bucket.wrappers.items():
            wrapper.plan(
                batch.kv_indptr,
                batch.kv_indices,
                batch.kv_last_page_len,
                self.model.num_heads,
                self.model.num_kv_heads,
                self.model.config.head_dim,
                batch.block_size,
                pos_encoding_mode="NONE",
                q_data_type=self.model.dtype,
                kv_data_type=kv_dtype,
            )

    def _forward(self, bucket: _Bucket) -> Tensor:
        context: AttentionContext
        if bucket.wrappers is None:
            context = TritonDecodeContext(
                self.store, bucket.slots, bucket.block_table, bucket.seq_lens
            )
        else:
            context = KernelContext(self.store, bucket.slots, bucket.wrappers)
        hidden = self.model(bucket.token_ids, bucket.positions, context)
        return self.model.logits(hidden)

    @torch.inference_mode()
    def _capture_all(self) -> None:
        """Capture every bucket, largest first, into one shared memory pool.

        Graphs are replayed one at a time and each one's output is consumed before the next
        replay, so they can share a pool; capturing the largest first sizes it once.
        """
        pool = torch.cuda.graph_pool_handle()
        block_size = self.store.block_size
        for size in reversed(self.sizes):
            bucket = self._buckets[size]
            batch = build_paged_batch([_PAD] * size, block_size, self.model.device)
            self._plan(bucket, batch)
            # One eager pass first: lazy initialization (kernel JIT, cuBLAS handles) must
            # not happen while the launches are being recorded.
            self._forward(bucket)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                bucket.logits = self._forward(bucket)
            bucket.graph = graph

    def run(self, spans: Sequence[SequenceSpan]) -> Tensor:
        """Run one decode pass and return ``[len(spans), vocab]`` logits.

        The result is a view of a static buffer: it is valid until the next call.
        """
        count = len(spans)
        size = self.bucket_size(count)
        if size is None:
            raise ValueError(f"no captured graph holds a batch of {count}")
        bucket = self._buckets[size]
        padded = [*spans, *([_PAD] * (size - count))]
        batch = build_paged_batch(padded, self.store.block_size, self.model.device)
        bucket.token_ids.copy_(batch.token_ids)
        bucket.positions.copy_(batch.positions)
        bucket.slots.copy_(batch.slots)
        self._inputs_enqueued.synchronize()
        self._plan(bucket, batch)
        self._inputs_enqueued.record()
        bucket.graph.replay()
        return bucket.logits[:count]

    def run_async(self, spans: Sequence[SequenceSpan], token_ids: Tensor | None) -> Tensor:
        """Enqueue one decode pass without waiting for the GPU; return its logits.

        Unlike :meth:`run`, nothing here synchronizes with the device, so the pass can be
        enqueued while the previous one is still executing: every input is staged in
        pinned memory and copied asynchronously, and the page indices reach the planner
        as a device tensor (FlashInfer copies host-side indices synchronously).

        Args:
            spans: One span per sequence; their token ids are used only if ``token_ids``
                is ``None``.
            token_ids: ``[len(spans)]`` int64 device tensor of the input tokens — the
                previous pass's samples, which the host has not seen yet.
        """
        count = len(spans)
        size = self.bucket_size(count)
        if size is None or not self.supports_async:
            raise ValueError(f"no captured graph runs a batch of {count} asynchronously")
        bucket = self._buckets[size]
        layout = paged_layout([*spans, *([_PAD] * (size - count))], self.store.block_size)
        staging = self._staging[self._next_staging]
        self._next_staging ^= 1
        staging.copied.synchronize()  # its previous contents have reached the device
        pages = len(layout.kv_indices)
        per_token = staging.per_token.numpy()
        per_token[:size] = layout.positions
        per_token[size : 2 * size] = layout.slots
        per_token[2 * size : 3 * size] = layout.token_ids
        tables = staging.tables.numpy()
        tables[: size + 1] = layout.kv_indptr
        tables[size + 1 : 2 * size + 1] = layout.kv_last_page_len
        staging.indices.numpy()[:pages] = layout.kv_indices
        device_per_token = staging.device_per_token[: 3 * size]
        device_per_token.copy_(staging.per_token[: 3 * size], non_blocking=True)
        indices = staging.device_indices[:pages]
        indices.copy_(staging.indices[:pages], non_blocking=True)
        bucket.positions.copy_(device_per_token[:size])
        bucket.slots.copy_(device_per_token[size : 2 * size])
        if token_ids is None:
            bucket.token_ids.copy_(device_per_token[2 * size :])
        else:
            bucket.token_ids[:count].copy_(token_ids)
        self._inputs_enqueued.synchronize()
        for kv_dtype, wrapper in bucket.wrappers.items():
            wrapper.plan(
                staging.tables[: size + 1],
                indices,
                staging.tables[size + 1 : 2 * size + 1],
                self.model.num_heads,
                self.model.num_kv_heads,
                self.model.config.head_dim,
                self.store.block_size,
                pos_encoding_mode="NONE",
                q_data_type=self.model.dtype,
                kv_data_type=kv_dtype,
            )
        self._inputs_enqueued.record()
        staging.copied.record()
        bucket.graph.replay()
        return bucket.logits[:count]
