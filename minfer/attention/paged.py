"""Batches over the paged KV cache, and a plain-PyTorch reference implementation.

A forward pass processes, for each sequence in the batch, a run of *new* tokens at
positions ``[start, start + len)``; everything before ``start`` is already in the cache.
:class:`PagedBatch` is the flattened description of such a pass: which tokens, at which
positions, where their keys/values get written, and each sequence's page table in the CSR
form FlashInfer uses. Prefill chunks, speculative verification and single-token decode are
all the same shape of batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain, islice
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from minfer.attention.base import AttentionContext
from minfer.cache.kv_store import PagedKVStore


@dataclass(frozen=True)
class SequenceSpan:
    """One sequence's share of a forward pass.

    Attributes:
        token_ids: The new tokens to process.
        start: Position of the first new token (number of tokens already cached).
        block_table: The sequence's blocks; must cover ``start + len(token_ids)`` positions.
    """

    token_ids: Sequence[int]
    start: int
    block_table: Sequence[int]


@dataclass(frozen=True)
class PagedBatch:
    """Flattened layout of a forward pass over ``num_sequences`` spans.

    Per-token tensors live on the model's device. The page-table tensors stay on the CPU:
    attention planning reads them there, and keeping them off the device avoids a
    device-to-host synchronization every step.

    Attributes:
        token_ids: ``[n]`` int64, device.
        positions: ``[n]`` int64, device. Absolute position of each token in its sequence.
        slots: ``[n]`` int64, device. KV slot each token's key is written to.
        qo_indptr: ``[s + 1]`` int32, CPU. Span boundaries in the flattened token order.
        kv_indptr: ``[s + 1]`` int32, CPU. CSR offsets into ``kv_indices``.
        kv_indices: ``[pages]`` int32, CPU. Blocks of each sequence, in position order.
        kv_last_page_len: ``[s]`` int32, CPU. Valid positions in each sequence's last block.
        block_size: Positions per block.
    """

    token_ids: Tensor
    positions: Tensor
    slots: Tensor
    qo_indptr: Tensor
    kv_indptr: Tensor
    kv_indices: Tensor
    kv_last_page_len: Tensor
    block_size: int

    @property
    def num_sequences(self) -> int:
        """Number of sequences in the batch."""
        return self.kv_last_page_len.shape[0]

    @property
    def num_tokens(self) -> int:
        """Number of tokens processed by the pass."""
        return self.token_ids.shape[0]


def build_paged_batch(
    spans: Sequence[SequenceSpan], block_size: int, device: torch.device | str
) -> PagedBatch:
    """Assemble the batch layout for ``spans`` (vectorized with numpy)."""
    count = len(spans)
    lengths = np.fromiter((len(s.token_ids) for s in spans), dtype=np.int64, count=count)
    starts = np.fromiter((s.start for s in spans), dtype=np.int64, count=count)
    if count == 0 or lengths.min() <= 0:
        raise ValueError("every span must contain at least one token")
    ends = starts + lengths
    num_pages = -(-ends // block_size)
    for span, pages in zip(spans, num_pages, strict=True):
        if len(span.block_table) < pages:
            raise ValueError(
                f"block table holds {len(span.block_table)} blocks, span needs {pages}"
            )

    total = int(lengths.sum())
    token_ids = np.fromiter(
        chain.from_iterable(s.token_ids for s in spans), dtype=np.int64, count=total
    )
    qo_indptr = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(lengths, out=qo_indptr[1:])
    span_of_token = np.repeat(np.arange(count), lengths)
    positions = np.arange(total) - (qo_indptr[:-1] - starts)[span_of_token]

    kv_indptr = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(num_pages, out=kv_indptr[1:])
    kv_indices = np.fromiter(
        chain.from_iterable(
            islice(s.block_table, int(p)) for s, p in zip(spans, num_pages, strict=True)
        ),
        dtype=np.int64,
        count=int(kv_indptr[-1]),
    )
    blocks = kv_indices[kv_indptr[span_of_token] + positions // block_size]
    slots = blocks * (2 * block_size) + positions % block_size
    last_page_len = (ends - 1) % block_size + 1

    return PagedBatch(
        token_ids=torch.from_numpy(token_ids).to(device),
        positions=torch.from_numpy(positions).to(device),
        slots=torch.from_numpy(slots).to(device),
        qo_indptr=torch.from_numpy(qo_indptr.astype(np.int32)),
        kv_indptr=torch.from_numpy(kv_indptr.astype(np.int32)),
        kv_indices=torch.from_numpy(kv_indices.astype(np.int32)),
        kv_last_page_len=torch.from_numpy(last_page_len.astype(np.int32)),
        block_size=block_size,
    )


def split_batch(batch: PagedBatch, num_spans: int) -> tuple[PagedBatch, PagedBatch]:
    """``batch``'s first ``num_spans`` spans and the rest, as two batches.

    The parts' per-token tensors are views of ``batch``'s; their page tables are rebased
    so that each part describes itself. A joint pass uses this to plan one attention
    kernel for its prefill spans and another for its decode rows.
    """
    rows = int(batch.qo_indptr[num_spans])
    pages = int(batch.kv_indptr[num_spans])
    head = PagedBatch(
        token_ids=batch.token_ids[:rows],
        positions=batch.positions[:rows],
        slots=batch.slots[:rows],
        qo_indptr=batch.qo_indptr[: num_spans + 1],
        kv_indptr=batch.kv_indptr[: num_spans + 1],
        kv_indices=batch.kv_indices[:pages],
        kv_last_page_len=batch.kv_last_page_len[:num_spans],
        block_size=batch.block_size,
    )
    tail = PagedBatch(
        token_ids=batch.token_ids[rows:],
        positions=batch.positions[rows:],
        slots=batch.slots[rows:],
        qo_indptr=batch.qo_indptr[num_spans:] - rows,
        kv_indptr=batch.kv_indptr[num_spans:] - pages,
        kv_indices=batch.kv_indices[pages:],
        kv_last_page_len=batch.kv_last_page_len[num_spans:],
        block_size=batch.block_size,
    )
    return head, tail


def dense_block_table(batch: PagedBatch) -> tuple[Tensor, Tensor]:
    """The batch's page tables as a padded matrix, for kernels that index them directly.

    Returns CPU tensors: ``[s, max_pages]`` int32 block ids (zero padded) and ``[s]`` int32
    cached lengths (positions each sequence attends over, including this pass's tokens).
    """
    counts = np.diff(batch.kv_indptr.numpy())
    table = np.zeros((len(counts), int(counts.max())), dtype=np.int32)
    table[np.arange(table.shape[1])[None, :] < counts[:, None]] = batch.kv_indices.numpy()
    lengths = (counts - 1) * batch.block_size + batch.kv_last_page_len.numpy()
    return torch.from_numpy(table), torch.from_numpy(lengths.astype(np.int32))


class PagedAttentionBackend(Protocol):
    """Factory for per-pass attention contexts over a :class:`PagedKVStore`."""

    def prefill(self, batch: PagedBatch) -> AttentionContext:
        """Context for a pass in which spans may hold several tokens."""
        ...

    def decode(self, batch: PagedBatch) -> AttentionContext:
        """Context for a pass in which every span holds exactly one token."""
        ...

    def joint(self, batch: PagedBatch, num_prefill_spans: int) -> AttentionContext:
        """Context for a pass whose spans after the first ``num_prefill_spans`` hold one token.

        Those are decoding sequences sharing a pass with prefill chunks; a backend may
        serve the two groups with different kernels.
        """
        ...


class _ReferenceContext:
    """Writes K/V into the store, then attends sequence by sequence with a position mask."""

    def __init__(self, store: PagedKVStore, batch: PagedBatch) -> None:
        self.store = store
        self.batch = batch
        device = batch.token_ids.device
        size = batch.block_size
        self.bounds = batch.qo_indptr.tolist()
        self.kv_slots: list[Tensor] = []
        self.masks: list[Tensor] = []
        kv_indptr = batch.kv_indptr.tolist()
        for i in range(batch.num_sequences):
            pages = batch.kv_indices[kv_indptr[i] : kv_indptr[i + 1]].to(torch.int64)
            kv_len = (len(pages) - 1) * size + int(batch.kv_last_page_len[i])
            offsets = torch.arange(size, dtype=torch.int64)
            slots = (pages[:, None] * (2 * size) + offsets[None, :]).reshape(-1)[:kv_len]
            self.kv_slots.append(slots.to(device))
            query_positions = batch.positions[self.bounds[i] : self.bounds[i + 1]]
            kv_positions = torch.arange(kv_len, device=device)
            self.masks.append(kv_positions[None, :] <= query_positions[:, None])

    def attend(self, layer: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """See :meth:`minfer.attention.base.AttentionContext.attend`."""
        self.store.write(layer, self.batch.slots, k, v)
        outputs = []
        for i, (slots, mask) in enumerate(zip(self.kv_slots, self.masks, strict=True)):
            keys, values = self.store.read(layer, slots)
            queries = q[self.bounds[i] : self.bounds[i + 1]]
            out = F.scaled_dot_product_attention(
                queries.transpose(0, 1).unsqueeze(0),
                keys.transpose(0, 1).unsqueeze(0),
                values.transpose(0, 1).unsqueeze(0),
                attn_mask=mask,
                enable_gqa=True,
            )
            outputs.append(out.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)


class ReferencePagedBackend:
    """Paged attention in plain PyTorch.

    Slow by construction (a Python loop over sequences, a gather per sequence and layer), but
    it runs on any device and in any dtype, including float32 and the CPU, which FlashInfer
    does not. It is the oracle the accelerated backends are tested against and the backend
    the CPU test-suite drives the engine with.
    """

    def __init__(self, store: PagedKVStore) -> None:
        self.store = store

    def prefill(self, batch: PagedBatch) -> AttentionContext:
        """See :meth:`PagedAttentionBackend.prefill`."""
        return _ReferenceContext(self.store, batch)

    def decode(self, batch: PagedBatch) -> AttentionContext:
        """See :meth:`PagedAttentionBackend.decode`."""
        return _ReferenceContext(self.store, batch)

    def joint(self, batch: PagedBatch, num_prefill_spans: int) -> AttentionContext:
        """See :meth:`PagedAttentionBackend.joint`."""
        return _ReferenceContext(self.store, batch)
