"""CUDA graphs of the joint pass: prefill chunks and decoding sequences in one forward.

A step that has both prefill and decode work would otherwise run two passes, and each pass
reads every weight of the model: at 7B that is about 5 ms of memory traffic before any
arithmetic. The joint pass runs the linear layers once over ``[prefill tokens | decode
rows]`` and gives each group its own attention kernel
(:class:`minfer.attention.flashinfer_backend.JointContext`).

Without a graph the pass would pay eager dispatch for every layer, which is what a decode
graph exists to avoid, so joint passes are captured at bucketed shapes ``(prefill spans,
prefill tokens, decode rows)``. The prefill part is padded like a prefill graph, the decode
part like a decode graph; padding rows read and write the reserved block 0. Passes that fit
no shape (more than 1,024 prefill tokens, mostly) run eagerly: there the prefill arithmetic
hides the dispatch.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from minfer.attention.flashinfer_backend import JointContext, flashinfer, needs_tensor_cores
from minfer.attention.paged import PagedBatch, SequenceSpan, build_paged_batch, split_batch
from minfer.attention.prefill_graphs import fit_shape, padding_spans
from minfer.cache.kv_store import PagedKVStore
from minfer.model.qwen2 import Qwen2Model

# Coarser than the prefill graphs' buckets: every prefill shape is captured once per
# decode size.
DEFAULT_SEQUENCE_BUCKETS = (2, 8, 32)
DEFAULT_TOKEN_BUCKETS = (64, 128, 256, 384, 512, 768, 1024)
DEFAULT_DECODE_SIZES = (2, 4, 8, 16, 32, 64, 128, 256)

_PAD = SequenceSpan(token_ids=(0,), start=0, block_table=(0,))


@dataclass
class _Bucket:
    """Static state of one captured ``(prefill spans, prefill tokens, decode rows)`` shape."""

    token_ids: Tensor
    positions: Tensor
    slots: Tensor
    rows: Tensor
    prefill: flashinfer.BatchPrefillWithPagedKVCacheWrapper
    decode: flashinfer.BatchDecodeWithPagedKVCacheWrapper
    graph: torch.cuda.CUDAGraph | None = None
    logits: Tensor | None = None


class JointGraphs:
    """Captured joint passes (model forward + unembedding of selected rows).

    Args:
        model: The model whose pass is captured.
        store: The paged KV store (single storage dtype).
        workspace: FlashInfer float workspace buffer.
        sequence_buckets: Prefill span counts to capture.
        token_buckets: Prefill token counts to capture.
        decode_sizes: Decode row counts to capture.
        max_pages_per_sequence: Longest block table a sequence can have.
    """

    def __init__(
        self,
        model: Qwen2Model,
        store: PagedKVStore,
        workspace: Tensor,
        sequence_buckets: Sequence[int] = DEFAULT_SEQUENCE_BUCKETS,
        token_buckets: Sequence[int] = DEFAULT_TOKEN_BUCKETS,
        decode_sizes: Sequence[int] = DEFAULT_DECODE_SIZES,
        max_pages_per_sequence: int = 2048,
    ) -> None:
        if len(store.tensors) != 1:
            raise ValueError("joint graphs need a single-dtype KV store")
        self.model = model
        self.store = store
        self.sequence_buckets = sorted(set(sequence_buckets))
        self.token_buckets = sorted(set(token_buckets))
        self.decode_sizes = sorted(set(decode_sizes))
        self._prefill_shapes = {
            (s, t) for s in self.sequence_buckets for t in self.token_buckets if t >= s
        }
        self._kv_dtype = next(iter(store.tensors))
        device = model.device
        tensor_cores = needs_tensor_cores(model.num_heads, model.num_kv_heads)

        def int_buffer(length: int) -> Tensor:
            return torch.zeros(length, dtype=torch.int32, device=device)

        self._buckets: dict[tuple[int, int, int], _Bucket] = {}
        for seqs, tokens in sorted(self._prefill_shapes):
            for size in self.decode_sizes:
                total = tokens + size
                self._buckets[seqs, tokens, size] = _Bucket(
                    token_ids=torch.zeros(total, dtype=torch.int64, device=device),
                    positions=torch.zeros(total, dtype=torch.int64, device=device),
                    slots=torch.zeros(total, dtype=torch.int64, device=device),
                    rows=torch.zeros(seqs + size, dtype=torch.int64, device=device),
                    prefill=flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                        workspace,
                        "NHD",
                        use_cuda_graph=True,
                        qo_indptr_buf=int_buffer(seqs + 1),
                        paged_kv_indptr_buf=int_buffer(seqs + 1),
                        paged_kv_indices_buf=int_buffer(seqs * max_pages_per_sequence),
                        paged_kv_last_page_len_buf=int_buffer(seqs),
                        max_total_num_rows=tokens,
                    ),
                    decode=flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                        workspace,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=tensor_cores,
                        paged_kv_indptr_buffer=int_buffer(size + 1),
                        paged_kv_indices_buffer=int_buffer(size * max_pages_per_sequence),
                        paged_kv_last_page_len_buffer=int_buffer(size),
                    ),
                )
        self._capture_all()

    def bucket(
        self, num_prefill_spans: int, num_prefill_tokens: int, num_decode: int
    ) -> tuple[int, int, int] | None:
        """The smallest captured shape that can hold a pass, or ``None``."""
        shape = fit_shape(
            self._prefill_shapes,
            self.sequence_buckets,
            self.token_buckets,
            num_prefill_spans,
            num_prefill_tokens,
        )
        index = bisect.bisect_left(self.decode_sizes, num_decode)
        if shape is None or index == len(self.decode_sizes):
            return None
        return (*shape, self.decode_sizes[index])

    def _plan(self, bucket: _Bucket, batch: PagedBatch, num_prefill_spans: int) -> None:
        """Point the bucket's two kernels at their parts of ``batch`` (host-side work)."""
        model = self.model
        shape = (model.num_heads, model.num_kv_heads, model.config.head_dim, batch.block_size)
        dtypes = {"q_data_type": model.dtype, "kv_data_type": self._kv_dtype}
        head, tail = split_batch(batch, num_prefill_spans)
        bucket.prefill.plan(
            head.qo_indptr,
            head.kv_indptr,
            head.kv_indices,
            head.kv_last_page_len,
            *shape,
            causal=True,
            pos_encoding_mode="NONE",
            **dtypes,
        )
        bucket.decode.plan(
            tail.kv_indptr,
            tail.kv_indices,
            tail.kv_last_page_len,
            *shape,
            pos_encoding_mode="NONE",
            **dtypes,
        )

    def _forward(self, bucket: _Bucket, num_prefill_tokens: int) -> Tensor:
        context = JointContext(
            self.store,
            bucket.slots,
            {self._kv_dtype: bucket.prefill},
            {self._kv_dtype: bucket.decode},
            num_prefill_tokens,
        )
        hidden = self.model(bucket.token_ids, bucket.positions, context)
        return self.model.logits(hidden.index_select(0, bucket.rows))

    @torch.inference_mode()
    def _capture_all(self) -> None:
        """Capture every bucket, largest first, into one shared memory pool."""
        pool = torch.cuda.graph_pool_handle()
        block_size = self.store.block_size
        for key in sorted(self._buckets, key=lambda k: -(k[1] + k[2])):
            seqs, tokens, size = key
            bucket = self._buckets[key]
            spans = [*padding_spans(seqs, tokens, block_size), *([_PAD] * size)]
            batch = build_paged_batch(spans, block_size, self.model.device)
            self._plan(bucket, batch, seqs)
            self._forward(bucket, tokens)  # lazy initialization must not be recorded
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                bucket.logits = self._forward(bucket, tokens)
            bucket.graph = graph

    def run(
        self,
        key: tuple[int, int, int],
        prefill: Sequence[SequenceSpan],
        rows: Sequence[int],
        decode: Sequence[SequenceSpan],
    ) -> Tensor:
        """Run one joint pass in bucket ``key``.

        Args:
            key: A shape returned by :meth:`bucket` for this pass.
            prefill: The prefill spans.
            rows: Rows of the flattened prefill tokens whose logits are wanted (at most
                one per span).
            decode: The decoding sequences' one-token spans.

        Returns:
            ``[len(rows) + len(decode), vocab]`` logits: the requested prefill rows, then
            every decode row. A view of a static buffer, valid until the next call.
        """
        seqs, tokens, size = key
        bucket = self._buckets[key]
        block_size = self.store.block_size
        real_tokens = sum(len(span.token_ids) for span in prefill)
        spans = [
            *prefill,
            *padding_spans(seqs - len(prefill), tokens - real_tokens, block_size),
            *decode,
            *([_PAD] * (size - len(decode))),
        ]
        batch = build_paged_batch(spans, block_size, self.model.device)
        bucket.token_ids.copy_(batch.token_ids)
        bucket.positions.copy_(batch.positions)
        bucket.slots.copy_(batch.slots)
        wanted = [*rows, *range(tokens, tokens + len(decode))]
        bucket.rows[: len(wanted)].copy_(torch.tensor(wanted, dtype=torch.int64))
        self._plan(bucket, batch, seqs)
        bucket.graph.replay()
        return bucket.logits[: len(wanted)]
