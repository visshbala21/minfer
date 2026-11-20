"""Sampling: filters, determinism, and the distribution actually drawn from."""

from __future__ import annotations

import pytest
import torch

from minfer.sampling import SamplingParams, greedy, sample


def _draw(logits, temperature=1.0, top_k=0, top_p=1.0, uniforms=None):
    b = logits.shape[0]
    if uniforms is None:
        uniforms = torch.rand(b, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    return sample(
        logits,
        torch.full((b,), temperature),
        torch.full((b,), top_k, dtype=torch.int64),
        torch.full((b,), top_p),
        uniforms,
    )


def test_params_validation():
    assert SamplingParams().greedy
    assert not SamplingParams(temperature=0.7).greedy
    for bad in (
        {"max_tokens": 0},
        {"temperature": -1.0},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"top_k": -1},
    ):
        with pytest.raises(ValueError):
            SamplingParams(**bad)


def test_greedy_is_argmax():
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 2.9]])
    assert greedy(logits).tolist() == [1, 0]


def test_top_k_one_is_argmax():
    logits = torch.randn(64, 50, generator=torch.Generator().manual_seed(1))
    assert torch.equal(_draw(logits, temperature=5.0, top_k=1), logits.argmax(-1))


def test_tiny_top_p_is_argmax():
    logits = torch.randn(64, 50, generator=torch.Generator().manual_seed(2))
    assert torch.equal(_draw(logits, top_p=1e-6), logits.argmax(-1))


def test_low_temperature_approaches_argmax():
    logits = torch.randn(64, 50, generator=torch.Generator().manual_seed(3))
    assert torch.equal(_draw(logits, temperature=1e-4), logits.argmax(-1))


def test_top_k_support():
    logits = torch.randn(1, 40, generator=torch.Generator().manual_seed(4)).expand(4000, -1)
    uniforms = torch.rand(4000, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    drawn = _draw(logits, top_k=5, uniforms=uniforms)
    allowed = set(logits[0].topk(5).indices.tolist())
    assert set(drawn.tolist()) == allowed


def test_top_p_support_keeps_the_crossing_token():
    # probabilities 0.5, 0.3, 0.15, 0.05: top_p=0.7 keeps {0, 1} (0.5 < 0.7 ≤ 0.8).
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = probs.log()[None, :].expand(4000, -1)
    uniforms = torch.rand(4000, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    drawn = _draw(logits, top_p=0.7, uniforms=uniforms)
    assert set(drawn.tolist()) == {0, 1}
    # Renormalized: token 0 should be drawn with probability 0.5 / 0.8.
    assert abs((drawn == 0).double().mean().item() - 0.625) < 0.03


def test_top_p_applies_after_top_k():
    # After top_k=2 the distribution is (0.4, 0.3)/0.7 = (0.571, 0.429); top_p=0.5 then
    # keeps only the first token. Without top-k, top_p=0.5 would keep two.
    probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
    logits = probs.log()[None, :].expand(500, -1)
    uniforms = torch.rand(500, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    assert set(_draw(logits, top_k=2, top_p=0.5, uniforms=uniforms).tolist()) == {0}
    assert set(_draw(logits, top_p=0.5, uniforms=uniforms).tolist()) == {0, 1}


def test_distribution_matches_softmax():
    logits = torch.tensor([1.0, 0.0, -1.0, 2.0, 0.5])
    n = 200_000
    uniforms = torch.rand(n, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    drawn = _draw(logits[None, :].expand(n, -1), temperature=0.7, uniforms=uniforms)
    empirical = torch.bincount(drawn, minlength=5).double() / n
    expected = torch.softmax(logits.double() / 0.7, dim=-1)
    assert (empirical - expected).abs().max().item() < 0.005


def test_rows_are_independent_of_batch_composition():
    generator = torch.Generator().manual_seed(9)
    logits = torch.randn(8, 100, generator=generator)
    temperature = torch.tensor([0.5, 1.0, 1.5, 0.8, 1.0, 2.0, 0.3, 1.1])
    top_k = torch.tensor([0, 10, 0, 3, 50, 0, 0, 7])
    top_p = torch.tensor([1.0, 0.9, 0.5, 1.0, 0.8, 0.95, 1.0, 0.6])
    uniforms = torch.rand(8, generator=generator, dtype=torch.float64)
    together = sample(logits, temperature, top_k, top_p, uniforms)
    for i in range(8):
        alone = sample(
            logits[i : i + 1],
            temperature[i : i + 1],
            top_k[i : i + 1],
            top_p[i : i + 1],
            uniforms[i : i + 1],
        )
        assert alone.item() == together[i].item()


def test_extreme_uniforms_stay_in_support():
    logits = torch.tensor([[0.0, 5.0, 1.0, -2.0]]).expand(2, -1)
    drawn = _draw(logits, top_k=2, uniforms=torch.tensor([0.0, 1.0 - 1e-16], dtype=torch.float64))
    assert set(drawn.tolist()) <= {1, 2}
