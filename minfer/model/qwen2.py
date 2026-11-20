"""Qwen2 decoder-only transformer in plain PyTorch.

The forward pass operates on a *flattened* batch: ``n`` tokens with explicit positions,
no padding and no batch dimension. How those tokens attend to each other (and to earlier
tokens held in a KV cache) is decided by the :class:`~minfer.attention.base.AttentionContext`
passed in, which is what lets the same code serve the dense reference, ragged prefill
batches, and single-token decode batches.

Numerics deliberately mirror ``transformers``' Qwen2: RMSNorm is evaluated in float32,
RoPE uses the rotate-half formulation with tables computed in float32, and the MLP is
SwiGLU. Module names match the checkpoint's tensor names (minus the ``model.`` prefix).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minfer.attention.base import AttentionContext
from minfer.model.config import ModelConfig


def _linear(in_features: int, out_features: int, bias: bool) -> nn.Linear:
    """An ``nn.Linear`` whose weights are left uninitialized (they are always loaded)."""
    return nn.utils.skip_init(nn.Linear, in_features, out_features, bias=bias)


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, computed in float32 regardless of the input dtype."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the last dimension of ``x``."""
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE ``cos``/``sin`` tables indexed by absolute position.

    Tables are ``[max_positions, head_dim]`` with the frequency vector repeated twice, the
    layout the rotate-half formulation expects. They are computed in float32 and follow the
    module's dtype afterwards, matching the reference implementation's cast.
    """

    cos: Tensor
    sin: Tensor

    def __init__(self, head_dim: int, theta: float, max_positions: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = torch.arange(max_positions, dtype=torch.float32)[:, None] * inv_freq[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(cos, sin)`` of shape ``[n, 1, head_dim]`` for the given positions."""
        return self.cos[positions].unsqueeze(1), self.sin[positions].unsqueeze(1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``x`` (``[n, heads, head_dim]``) by its position's angles."""
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    """Grouped-query self-attention projections; the attention itself is delegated."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        hidden = config.hidden_size
        self.q_proj = _linear(hidden, self.num_heads * self.head_dim, bias=True)
        self.k_proj = _linear(hidden, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = _linear(hidden, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = _linear(self.num_heads * self.head_dim, hidden, bias=False)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, layer: int, attn: AttentionContext
    ) -> Tensor:
        """Project to q/k/v, apply RoPE, attend through ``attn``, and project back."""
        n = x.shape[0]
        q = self.q_proj(x).view(n, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(n, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(n, self.num_kv_heads, self.head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = attn.attend(layer, q, k, v)
        return self.o_proj(out.reshape(n, self.num_heads * self.head_dim))


class MLP(nn.Module):
    """SwiGLU feed-forward block: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = _linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = _linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = _linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the feed-forward block."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, layer: int, attn: AttentionContext
    ) -> Tensor:
        """Run attention and MLP sub-blocks with residual connections."""
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, layer, attn)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2Model(nn.Module):
    """Qwen2 causal language model over a flattened token batch.

    ``forward`` returns final hidden states; :meth:`logits` maps hidden states to vocabulary
    logits. They are separate because the engine only needs logits for a subset of rows
    (the last token of each prefill chunk), and the unembedding is the widest matmul in the
    network.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.utils.skip_init(nn.Embedding, config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rope = RotaryEmbedding(
            config.head_dim, config.rope_theta, config.max_position_embeddings
        )
        self.lm_head = _linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        """Device the weights live on."""
        return self.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype of the weights."""
        return self.embed_tokens.weight.dtype

    def forward(self, token_ids: Tensor, positions: Tensor, attn: AttentionContext) -> Tensor:
        """Run the transformer stack.

        Args:
            token_ids: ``[n]`` token ids of the flattened batch.
            positions: ``[n]`` absolute position of each token within its sequence.
            attn: Attention strategy for this pass.

        Returns:
            ``[n, hidden_size]`` hidden states after the final norm.
        """
        x = self.embed_tokens(token_ids)
        cos, sin = self.rope(positions)
        for index, layer in enumerate(self.layers):
            x = layer(x, cos, sin, index, attn)
        return self.norm(x)

    def logits(self, hidden: Tensor) -> Tensor:
        """Map ``[m, hidden_size]`` hidden states to ``[m, vocab_size]`` logits."""
        return self.lm_head(hidden)
