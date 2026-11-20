"""CUDA graphs of the prefill pass at bucketed (sequences, tokens) shapes.

The decode graphs (:mod:`minfer.attention.decode_graphs`) remove Python dispatch from
single-token steps; this does the same for the ragged pass that prefills prompt chunks.
Even with fused kernels an eager pass costs about 7 ms of dispatch for 28 layers, whatever
its size — more than the GPU needs for a few hundred tokens — and under load almost every
engine step contains one.

A prefill batch has two free dimensions, the number of spans and the total number of
tokens, so buckets are pairs ``(S, T)``. A real batch is padded up to its bucket with
spans that live entirely in the reserved block 0: one-token spans to reach ``S`` spans, and
one longer span that absorbs the remaining tokens. FlashInfer's prefill wrapper in
CUDA-graph mode keeps the kernel's launch shape fixed for a given ``(S, T)`` while
``plan()`` rewrites its static page-table buffers every pass.

The unembedding is applied to ``S`` selected rows only (the last row of each span that
completes its sequence), gathered through a static index buffer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from minfer.attention.flashinfer_backend import KernelContext, flashinfer
from minfer.attention.paged import PagedBatch, SequenceSpan, build_paged_batch
from minfer.cache.kv_store import PagedKVStore
from minfer.model.qwen2 import Qwen2Model

DEFAULT_SEQUENCE_BUCKETS = (2, 4, 8, 16, 32)
# Up to 1,024 tokens only. An eager pass overlaps its Python dispatch with GPU work, so
# its cost is max(dispatch ≈ 7 ms, compute); a graph removes the dispatch but pays for its
# padding. Measured (bench/prefill_latency.py): graphs win clearly below ~512 tokens and
# lose above ~1,000 once padded, so larger passes stay eager. Steps of ≤ 25% bound the waste.
DEFAULT_TOKEN_BUCKETS = (32, 64, 128, 192, 256, 384, 512, 640, 768, 896, 1024)


@dataclass
class _Bucket:
    """Static state of one captured ``(sequences, tokens)`` shape."""

    token_ids: Tensor
    positions: Tensor
    slots: Tensor
    rows: Tensor
    wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper
    graph: torch.cuda.CUDAGraph | None = None
    logits: Tensor | None = None


def _padding(num_spans: int, num_tokens: int, block_size: int) -> list[SequenceSpan]:
    """``num_spans`` spans totalling ``num_tokens`` tokens, all inside block 0."""
    if num_spans == 0:
        return []
    rest = num_tokens - (num_spans - 1)
    pages = -(-rest // block_size)
    single = SequenceSpan(token_ids=(0,), start=0, block_table=(0,))
    return [*([single] * (num_spans - 1)), SequenceSpan((0,) * rest, 0, (0,) * pages)]


class PrefillGraphs:
    """Captured prefill passes (model forward + unembedding of selected rows).

    Args:
        model: The model whose prefill pass is captured.
        store: The paged KV store (single storage dtype).
        workspace: FlashInfer float workspace buffer.
        sequence_buckets: Span counts to capture.
        token_buckets: Token counts to capture.
        max_pages_per_sequence: Longest block table a sequence can have.
    """

    def __init__(
        self,
        model: Qwen2Model,
        store: PagedKVStore,
        workspace: Tensor,
        sequence_buckets: Sequence[int] = DEFAULT_SEQUENCE_BUCKETS,
        token_buckets: Sequence[int] = DEFAULT_TOKEN_BUCKETS,
        max_pages_per_sequence: int = 2048,
    ) -> None:
        if len(store.tensors) != 1:
            raise ValueError("prefill graphs need a single-dtype KV store")
        self.model = model
        self.store = store
        self.sequence_buckets = sorted(set(sequence_buckets))
        self.token_buckets = sorted(set(token_buckets))
        self._kv_dtype = next(iter(store.tensors))
        device = model.device
        self._buckets: dict[tuple[int, int], _Bucket] = {}
        for seqs in self.sequence_buckets:
            for tokens in self.token_buckets:
                if tokens < seqs:
                    continue
                wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    workspace,
                    "NHD",
                    use_cuda_graph=True,
                    qo_indptr_buf=torch.zeros(seqs + 1, dtype=torch.int32, device=device),
                    paged_kv_indptr_buf=torch.zeros(seqs + 1, dtype=torch.int32, device=device),
                    paged_kv_indices_buf=torch.zeros(
                        seqs * max_pages_per_sequence, dtype=torch.int32, device=device
                    ),
                    paged_kv_last_page_len_buf=torch.zeros(seqs, dtype=torch.int32, device=device),
                    max_total_num_rows=tokens,
                )
                self._buckets[seqs, tokens] = _Bucket(
                    token_ids=torch.zeros(tokens, dtype=torch.int64, device=device),
                    positions=torch.zeros(tokens, dtype=torch.int64, device=device),
                    slots=torch.zeros(tokens, dtype=torch.int64, device=device),
                    rows=torch.zeros(seqs, dtype=torch.int64, device=device),
                    wrapper=wrapper,
                )
        self._capture_all()

    def bucket(self, num_spans: int, num_tokens: int) -> tuple[int, int] | None:
        """The smallest captured shape that can hold a batch, or ``None``.

        Padding needs at least one token per padding span, and a bucket with spare tokens
        needs at least one padding span to put them in.
        """
        for seqs in self.sequence_buckets:
            if seqs < num_spans:
                continue
            for tokens in self.token_buckets:
                spare_spans, spare_tokens = seqs - num_spans, tokens - num_tokens
                if (seqs, tokens) not in self._buckets or spare_tokens < spare_spans:
                    continue
                if spare_spans > 0 or spare_tokens == 0:
                    return seqs, tokens
        return None

    def _plan(self, bucket: _Bucket, batch: PagedBatch) -> None:
        bucket.wrapper.plan(
            batch.qo_indptr,
            batch.kv_indptr,
            batch.kv_indices,
            batch.kv_last_page_len,
            self.model.num_heads,
            self.model.num_kv_heads,
            self.model.config.head_dim,
            batch.block_size,
            causal=True,
            pos_encoding_mode="NONE",
            q_data_type=self.model.dtype,
            kv_data_type=self._kv_dtype,
        )

    def _forward(self, bucket: _Bucket) -> Tensor:
        context = KernelContext(self.store, bucket.slots, {self._kv_dtype: bucket.wrapper})
        hidden = self.model(bucket.token_ids, bucket.positions, context)
        return self.model.logits(hidden.index_select(0, bucket.rows))

    @torch.inference_mode()
    def _capture_all(self) -> None:
        """Capture every bucket, largest first, into one shared memory pool."""
        pool = torch.cuda.graph_pool_handle()
        size = self.store.block_size
        for (seqs, tokens), bucket in sorted(self._buckets.items(), key=lambda kv: -kv[0][1]):
            batch = build_paged_batch(_padding(seqs, tokens, size), size, self.model.device)
            self._plan(bucket, batch)
            self._forward(bucket)  # lazy initialization must not be recorded
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                bucket.logits = self._forward(bucket)
            bucket.graph = graph

    def run(
        self, key: tuple[int, int], spans: Sequence[SequenceSpan], rows: Sequence[int]
    ) -> Tensor:
        """Run one prefill pass in bucket ``key``; return logits of the given token rows.

        ``rows`` index the flattened real tokens (at most one per span). The result is a
        view of a static buffer, valid until the next call.
        """
        seqs, tokens = key
        bucket = self._buckets[key]
        size = self.store.block_size
        real_tokens = sum(len(span.token_ids) for span in spans)
        padded = [*spans, *_padding(seqs - len(spans), tokens - real_tokens, size)]
        batch = build_paged_batch(padded, size, self.model.device)
        bucket.token_ids.copy_(batch.token_ids)
        bucket.positions.copy_(batch.positions)
        bucket.slots.copy_(batch.slots)
        if rows:
            index = torch.tensor(rows, dtype=torch.int64)
            bucket.rows[: len(rows)].copy_(index)
        self._plan(bucket, batch)
        bucket.graph.replay()
        return bucket.logits[: len(rows)]
