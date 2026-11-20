"""Attention implementations behind the :class:`AttentionContext` seam."""

from minfer.attention.base import AttentionContext
from minfer.attention.dense import DenseCausalAttention

__all__ = ["AttentionContext", "DenseCausalAttention"]
