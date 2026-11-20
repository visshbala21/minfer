"""The seam between the model and the attention implementation.

The transformer layers in :mod:`minfer.model.qwen2` compute rotated queries/keys and values
and then hand them to an :class:`AttentionContext`. The context decides where keys and values
live (nowhere, for the dense reference; a paged KV cache, for the engine) and which kernel
computes the attention output. One forward-pass implementation therefore serves the dense
reference used in tests, FlashInfer paged attention, and the custom Triton kernel.
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor


class AttentionContext(Protocol):
    """Attention strategy for a single forward pass over a flattened batch of ``n`` tokens."""

    def attend(self, layer: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Compute causal self-attention for one layer.

        Args:
            layer: Index of the transformer layer (selects that layer's KV storage).
            q: Rotated queries, ``[n, num_heads, head_dim]``.
            k: Rotated keys for the same tokens, ``[n, num_kv_heads, head_dim]``.
            v: Values for the same tokens, ``[n, num_kv_heads, head_dim]``.

        Returns:
            Attention output, ``[n, num_heads, head_dim]``.
        """
        ...
