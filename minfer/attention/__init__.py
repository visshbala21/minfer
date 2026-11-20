"""Attention implementations behind the :class:`AttentionContext` seam."""

from minfer.attention.base import AttentionContext
from minfer.attention.dense import DenseCausalAttention
from minfer.attention.paged import (
    PagedAttentionBackend,
    PagedBatch,
    ReferencePagedBackend,
    SequenceSpan,
    build_paged_batch,
)

__all__ = [
    "AttentionContext",
    "DenseCausalAttention",
    "PagedAttentionBackend",
    "PagedBatch",
    "ReferencePagedBackend",
    "SequenceSpan",
    "build_paged_batch",
]
