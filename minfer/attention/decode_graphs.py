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
    wrapper: flashinfer.BatchDecodeWithPagedKVCacheWrapper | None
    block_table: Tensor | None = None
    seq_lens: Tensor | None = None
    graph: torch.cuda.CUDAGraph | None = None
    logits: Tensor | None = None


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
                    wrapper=None,
                    block_table=torch.zeros(
                        (size, max_pages_per_sequence), dtype=torch.int32, device=device
                    ),
                    seq_lens=torch.ones(size, dtype=torch.int32, device=device),
                )
                continue
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                use_cuda_graph=True,
                use_tensor_cores=needs_tensor_cores(model.num_heads, model.num_kv_heads),
                paged_kv_indptr_buffer=torch.zeros(size + 1, dtype=torch.int32, device=device),
                paged_kv_indices_buffer=torch.zeros(
                    size * max_pages_per_sequence, dtype=torch.int32, device=device
                ),
                paged_kv_last_page_len_buffer=torch.zeros(size, dtype=torch.int32, device=device),
            )
            self._buckets[size] = _Bucket(
                size=size,
                token_ids=torch.zeros(size, dtype=torch.int64, device=device),
                positions=torch.zeros(size, dtype=torch.int64, device=device),
                slots=torch.zeros(size, dtype=torch.int64, device=device),
                wrapper=wrapper,
            )
        self._capture_all()

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
        if bucket.wrapper is None:
            table, lengths = dense_block_table(batch)
            bucket.block_table[:, : table.shape[1]].copy_(table)
            bucket.seq_lens.copy_(lengths)
            return
        bucket.wrapper.plan(
            batch.kv_indptr,
            batch.kv_indices,
            batch.kv_last_page_len,
            self.model.num_heads,
            self.model.num_kv_heads,
            self.model.config.head_dim,
            batch.block_size,
            pos_encoding_mode="NONE",
            q_data_type=self.model.dtype,
            kv_data_type=self.store.data.dtype,
        )

    def _forward(self, bucket: _Bucket) -> Tensor:
        context: AttentionContext
        if bucket.wrapper is None:
            context = TritonDecodeContext(
                self.store, bucket.slots, bucket.block_table, bucket.seq_lens
            )
        else:
            context = KernelContext(self.store, bucket.slots, bucket.wrapper)
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
        self._plan(bucket, batch)
        bucket.graph.replay()
        return bucket.logits[:count]
