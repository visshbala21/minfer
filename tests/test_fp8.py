"""FP8 KV storage on the CPU: quantization, scaling, calibration."""

from __future__ import annotations

import pytest
import torch
from tiny import dense_logits, make_tiny_model

from minfer.attention import ReferencePagedBackend, SequenceSpan, build_paged_batch
from minfer.cache import BlockManager
from minfer.cache.kv_store import FP8, FP8_MAX, PagedKVStore

BLOCK = 4


def make_store(kv_dtype=FP8, layers=2) -> PagedKVStore:
    return PagedKVStore(layers, 6, BLOCK, 2, 8, "cpu", torch.float32, kv_dtype)


def test_fp8_store_is_half_the_size_of_bf16_and_a_quarter_of_fp32():
    fp32 = make_store(None)
    assert make_store().nbytes * 4 == fp32.nbytes
    assert make_store().data.dtype == FP8
    with pytest.raises(ValueError):
        make_store(torch.int8)


def test_round_trip_error_is_bounded_by_the_format():
    store = make_store()
    generator = torch.Generator().manual_seed(0)
    k = torch.randn(4, 2, 8, generator=generator) * 3
    v = torch.randn(4, 2, 8, generator=generator) * 3
    slots = torch.tensor([8, 9, 10, 17])
    store.write(1, slots, k, v)
    read_k, read_v = store.read(1, slots)
    # E4M3 has 3 mantissa bits: rounding error is at most 1/16 of the value.
    assert ((read_k - k).abs() <= k.abs() / 16 + 1e-3).all()
    assert ((read_v - v).abs() <= v.abs() / 16 + 1e-3).all()
    assert store.layer(0).view(torch.uint8).sum() == 0  # other layers untouched


def test_scales_prevent_saturation_and_underflow():
    big = torch.full((1, 2, 8), 3000.0)
    small = torch.full((1, 2, 8), 1e-4)
    slots = torch.tensor([0])

    unscaled = make_store()
    unscaled.write(0, slots, big, small)
    k, v = unscaled.read(0, slots)
    assert torch.all(k == FP8_MAX) and torch.all(v == 0)  # clipped, and flushed to zero

    scaled = make_store()
    scaled.k_scale[0] = 3000.0 / FP8_MAX
    scaled.v_scale[0] = 1e-4 / FP8_MAX
    scaled.write(0, slots, big, small)
    k, v = scaled.read(0, slots)
    assert torch.allclose(k, big, rtol=1 / 16) and torch.allclose(v, small, rtol=1 / 16)


def test_calibration_sets_per_layer_scales_from_observed_maxima():
    store = make_store()
    store.observe()
    slots = torch.tensor([0, 1])
    k0, v0 = torch.full((2, 2, 8), 20.0), torch.full((2, 2, 8), -0.5)
    k1, v1 = torch.full((2, 2, 8), 0.25), torch.full((2, 2, 8), 900.0)
    store.write(0, slots, k0, v0)
    store.write(1, slots, k1, v1)
    store.calibrate(headroom=1.0)
    expected_k = torch.tensor([20.0, 0.25]) / FP8_MAX
    expected_v = torch.tensor([0.5, 900.0]) / FP8_MAX
    assert torch.allclose(store.k_scale, expected_k) and torch.allclose(store.v_scale, expected_v)
    store.write(1, slots, k1, v1)
    _, v = store.read(1, slots)
    assert torch.allclose(v, v1, rtol=1 / 16)
    with pytest.raises(RuntimeError):
        store.calibrate()


def test_model_through_an_fp8_cache_stays_close_to_the_exact_model():
    """Prefill + decode with FP8 KV: logits move by quantization noise, not more."""
    model = make_tiny_model()
    config = model.config
    tokens = [(7 * i + 3) % 190 + 4 for i in range(40)]

    def run(kv_dtype, calibrate: bool) -> torch.Tensor:
        store = PagedKVStore(
            config.num_layers,
            32,
            BLOCK,
            config.num_kv_heads,
            config.head_dim,
            "cpu",
            model.dtype,
            kv_dtype,
        )
        backend, manager = ReferencePagedBackend(store), BlockManager(32, BLOCK)
        manager.ensure_capacity("s", len(tokens))
        table = manager.table("s")

        def forward(start, stop):
            batch = build_paged_batch(
                [SequenceSpan(tokens[start:stop], start, table)], BLOCK, "cpu"
            )
            return model(batch.token_ids, batch.positions, backend.prefill(batch))

        with torch.inference_mode():
            if calibrate:
                store.observe()
                forward(0, len(tokens))
                store.calibrate()
            forward(0, 30)
            hidden = torch.cat([forward(i, i + 1) for i in range(30, len(tokens))])
            return model.logits(hidden[-1:])[0]

    exact = dense_logits(model, tokens)
    assert torch.allclose(run(None, False), exact, atol=1e-4)
    quantized = run(FP8, True)
    error = (quantized - exact).abs()
    spread = exact.std().item()
    assert error.max().item() < 0.5 * spread and error.mean().item() < 0.1 * spread
    assert quantized.argmax() == exact.argmax()
