"""Fused Triton kernels for the transformer block's small element-wise stages.

A decode step for one token is latency-bound by the *number* of kernels, not by their
arithmetic: each of the ~45 PyTorch operators per layer becomes its own CUDA kernel, and a
kernel that touches 1,536 numbers still costs a few microseconds to launch. These kernels
collapse the three operator-heavy stages of a layer into one launch each:

* ``add_rms_norm``: residual add + RMSNorm (float32 statistics, weight applied after the
  cast back, exactly as the reference module does) — two outputs, one kernel instead of 9.
* ``rope_qkv``: split a merged q/k/v projection output and rotate q and k — one kernel
  instead of ~14.
* ``silu_mul``: split a merged gate/up projection output and compute ``silu(gate) * up``.

Numerics relative to the reference modules, in bf16: the norm and the activation round
to the model dtype exactly where the PyTorch implementation does and are bit-identical to
it. The rotation evaluates ``x·cos + rot(x)·sin`` in float32 and rounds once, where the
reference rounds both products and then the sum; it is the slightly more accurate of the
two and differs from the reference by rounding noise only (visible where the two products
nearly cancel). In float32 all three agree with the reference to ~1e-6.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _add_rms_norm_kernel(
    x_ptr,
    delta_ptr,
    weight_ptr,
    sum_ptr,
    out_ptr,
    eps,
    HAS_DELTA: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    x = tl.load(x_ptr + row * HIDDEN + cols, mask=mask, other=0.0)
    if HAS_DELTA:
        delta = tl.load(delta_ptr + row * HIDDEN + cols, mask=mask, other=0.0)
        x = (x.to(tl.float32) + delta.to(tl.float32)).to(x.dtype)
        tl.store(sum_ptr + row * HIDDEN + cols, x, mask=mask)
    wide = x.to(tl.float32)
    mean_square = tl.sum(wide * wide, axis=0) / HIDDEN
    normed = (wide * tl.rsqrt(mean_square + eps)).to(x.dtype)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    out = (normed.to(tl.float32) * weight.to(tl.float32)).to(x.dtype)
    tl.store(out_ptr + row * HIDDEN + cols, out, mask=mask)


def add_rms_norm(
    x: Tensor, delta: Tensor | None, weight: Tensor, eps: float
) -> tuple[Tensor, Tensor]:
    """``s = x + delta`` (or ``x``) and ``rms_norm(s) * weight``, for ``[n, hidden]`` inputs.

    Returns ``(s, normed)``. With ``delta=None`` the first element is ``x`` itself.
    """
    rows, hidden = x.shape
    out = torch.empty_like(x)
    total = torch.empty_like(x) if delta is not None else x
    _add_rms_norm_kernel[(rows,)](
        x,
        delta if delta is not None else x,
        weight,
        total,
        out,
        eps,
        HAS_DELTA=delta is not None,
        HIDDEN=hidden,
        BLOCK=triton.next_power_of_2(hidden),
    )
    return total, out


@triton.jit
def _rope_qkv_kernel(
    qkv_ptr,
    positions_ptr,
    cos_ptr,
    sin_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    stride_qkv,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)  # 0..NUM_HEADS-1: q heads; then k heads; then v heads
    position = tl.load(positions_ptr + row)
    half = tl.arange(0, HALF)
    source = qkv_ptr + row * stride_qkv + head * HEAD_DIM
    first = tl.load(source + half)
    second = tl.load(source + HALF + half)
    if head < NUM_HEADS + NUM_KV_HEADS:
        # Tables hold the frequency vector twice; one half is enough.
        cos = tl.load(cos_ptr + position * HEAD_DIM + half).to(tl.float32)
        sin = tl.load(sin_ptr + position * HEAD_DIM + half).to(tl.float32)
        a = first.to(tl.float32)
        b = second.to(tl.float32)
        # out = x * cos + rotate_half(x) * sin, evaluated in float32 and rounded once.
        out_first = (a * cos - b * sin).to(first.dtype)
        out_second = (b * cos + a * sin).to(first.dtype)
        if head < NUM_HEADS:
            target = q_ptr + (row * NUM_HEADS + head) * HEAD_DIM
        else:
            target = k_ptr + (row * NUM_KV_HEADS + head - NUM_HEADS) * HEAD_DIM
        tl.store(target + half, out_first)
        tl.store(target + HALF + half, out_second)
    else:
        target = v_ptr + (row * NUM_KV_HEADS + head - NUM_HEADS - NUM_KV_HEADS) * HEAD_DIM
        tl.store(target + half, first)
        tl.store(target + HALF + half, second)


def rope_qkv(
    qkv: Tensor,
    positions: Tensor,
    cos: Tensor,
    sin: Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Split ``[n, (h + 2·h_kv) · d]`` into rotated q, rotated k and v (``[n, heads, d]``).

    Args:
        qkv: Output of the merged q/k/v projection.
        positions: ``[n]`` int64 absolute positions.
        cos: ``[max_positions, d]`` RoPE cosine table in the model dtype.
        sin: ``[max_positions, d]`` RoPE sine table.
        num_heads: Query heads.
        num_kv_heads: Key/value heads.
        head_dim: Per-head dimension (even).
    """
    rows = qkv.shape[0]
    q = torch.empty((rows, num_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    k = torch.empty((rows, num_kv_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    v = torch.empty((rows, num_kv_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    _rope_qkv_kernel[(rows, num_heads + 2 * num_kv_heads)](
        qkv,
        positions,
        cos,
        sin,
        q,
        k,
        v,
        qkv.stride(0),
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        HALF=head_dim // 2,
    )
    return q, k, v


@triton.jit
def _silu_mul_kernel(gate_up_ptr, out_ptr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < WIDTH
    base = gate_up_ptr + row * 2 * WIDTH
    gate = tl.load(base + cols, mask=mask, other=0.0)
    up = tl.load(base + WIDTH + cols, mask=mask, other=0.0)
    wide = gate.to(tl.float32)
    activated = (wide * tl.sigmoid(wide)).to(gate.dtype)
    out = (activated.to(tl.float32) * up.to(tl.float32)).to(gate.dtype)
    tl.store(out_ptr + row * WIDTH + cols, out, mask=mask)


def silu_mul(gate_up: Tensor) -> Tensor:
    """``silu(gate) * up`` for a merged ``[n, 2 · width]`` gate/up projection output."""
    rows, double = gate_up.shape
    width = double // 2
    block = 1024
    out = torch.empty((rows, width), dtype=gate_up.dtype, device=gate_up.device)
    _silu_mul_kernel[(rows, triton.cdiv(width, block))](gate_up, out, WIDTH=width, BLOCK=block)
    return out
