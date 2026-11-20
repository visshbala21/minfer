"""Token sampling: greedy, temperature, top-k, top-p, with per-request seeds.

Sampling is inverse-CDF with exactly one uniform random number per row. The uniforms are
drawn on the host from each request's own generator, so a seeded request produces the same
tokens no matter which other requests share its batch, while the tensor work stays
vectorized over the batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingParams:
    """Per-request generation settings.

    Attributes:
        max_tokens: Maximum number of tokens to generate.
        temperature: Softmax temperature; ``0`` selects greedy decoding.
        top_p: Nucleus mass in ``(0, 1]``; ``1`` disables the filter.
        top_k: Keep only the ``top_k`` most likely tokens; ``0`` disables the filter.
        stop_token_ids: Extra token ids that end generation (in addition to the model's EOS).
        ignore_eos: Keep generating past the model's EOS tokens.
        seed: Seed for this request's random stream; ``None`` draws from the engine's.
    """

    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")

    @property
    def greedy(self) -> bool:
        """Whether the request decodes greedily."""
        return self.temperature == 0.0


def greedy(logits: Tensor) -> Tensor:
    """Most likely token of each row of ``logits`` (``[b, vocab]``)."""
    return logits.argmax(dim=-1)


def _filtered_sorted_probs(
    logits: Tensor, temperature: Tensor, top_k: Tensor, top_p: Tensor
) -> tuple[Tensor, Tensor]:
    """Each row's sampling distribution in descending order, and the sort order.

    Filters are applied in the order temperature → top-k → top-p (the order transformers
    and vLLM use): top-p is measured on the distribution that survives top-k. The returned
    probabilities are zero for filtered tokens and sum to one.
    """
    vocab = logits.shape[-1]
    scaled = logits.float() / temperature[:, None]
    ordered, order = scaled.sort(dim=-1, descending=True)
    probs = torch.softmax(ordered, dim=-1)

    ranks = torch.arange(vocab, device=logits.device)
    limit = torch.where(top_k > 0, top_k, vocab)
    probs = probs.masked_fill(ranks[None, :] >= limit[:, None], 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Nucleus: drop a token when the mass strictly before it already reaches top_p, so the
    # token that crosses the threshold is kept and the head of the list always survives.
    mass_before = probs.cumsum(dim=-1) - probs
    probs = probs.masked_fill(mass_before >= top_p[:, None], 0.0)
    return probs / probs.sum(dim=-1, keepdim=True), order


def _inverse_cdf(probs: Tensor, uniforms: Tensor) -> Tensor:
    """Index drawn from each row of ``probs`` (rows sum to ~1) by its uniform in [0, 1)."""
    cdf = probs.cumsum(dim=-1)
    target = uniforms.to(cdf.dtype)[:, None] * cdf[:, -1:]
    index = torch.searchsorted(cdf, target, right=True).clamp_(max=probs.shape[-1] - 1)
    # Rounding at the tail can land past the last token with nonzero probability.
    last = probs.shape[-1] - 1 - (probs > 0).flip(-1).int().argmax(dim=-1, keepdim=True)
    return torch.minimum(index, last)


def sample(
    logits: Tensor, temperature: Tensor, top_k: Tensor, top_p: Tensor, uniforms: Tensor
) -> Tensor:
    """Draw one token per row.

    Args:
        logits: ``[b, vocab]`` unnormalized scores.
        temperature: ``[b]`` strictly positive temperatures.
        top_k: ``[b]`` int64; ``0`` (or anything ≥ vocab) keeps every token.
        top_p: ``[b]`` nucleus mass in ``(0, 1]``.
        uniforms: ``[b]`` uniform draws in ``[0, 1)``.

    Returns:
        ``[b]`` int64 token ids.
    """
    probs, order = _filtered_sorted_probs(logits, temperature, top_k, top_p)
    return order.gather(-1, _inverse_cdf(probs, uniforms)).squeeze(-1)


def warped_probs(logits: Tensor, temperature: Tensor, top_k: Tensor, top_p: Tensor) -> Tensor:
    """Each row's sampling distribution, indexed by token id (``[b, vocab]``).

    The same distribution :func:`sample` draws from; speculative decoding needs it
    explicitly to accept or reject drafted tokens.
    """
    probs, order = _filtered_sorted_probs(logits, temperature, top_k, top_p)
    return torch.zeros_like(probs).scatter_(-1, order, probs)


def sample_from_probs(probs: Tensor, uniforms: Tensor) -> Tensor:
    """Draw one token per row of an explicit distribution (``[b, vocab]``, token order)."""
    return _inverse_cdf(probs, uniforms).squeeze(-1)
