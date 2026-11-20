"""Cache-free causal attention over full sequences: the numerical reference."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


class DenseCausalAttention:
    """Causal attention over ``batch`` equal-length sequences, with no KV cache.

    The flattened token batch is interpreted as ``batch`` sequences of ``seq_len`` tokens
    laid out back to back. This is the straightforward O(T²) formulation used to check the
    model against Hugging Face and as the oracle for the paged engine paths.
    """

    def __init__(self, batch: int, seq_len: int) -> None:
        self.batch = batch
        self.seq_len = seq_len

    def attend(self, layer: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """See :meth:`minfer.attention.base.AttentionContext.attend`."""
        b, t = self.batch, self.seq_len
        num_heads, head_dim = q.shape[1], q.shape[2]
        num_kv_heads = k.shape[1]
        q = q.view(b, t, num_heads, head_dim).transpose(1, 2)
        k = k.view(b, t, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(b, t, num_kv_heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return out.transpose(1, 2).reshape(b * t, num_heads, head_dim)
