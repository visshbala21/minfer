"""A paged-attention decode kernel in Triton, as an alternative to FlashInfer's.

One query token per sequence attends to that sequence's cached keys and values, which live
in fixed-size blocks scattered through the KV store and are found through a block table.

**Structure.** The grid is ``(batch, num_kv_heads)``: one program per sequence and KV head.
Grouped-query attention is handled *inside* a program — the ``G = num_heads /
num_kv_heads`` query heads that share a KV head are stacked into one small matrix, so each
key/value block is read from memory once and used by all ``G`` queries (a head-per-program
kernel would read it ``G`` times). A program walks its sequence ``PAGES_PER_ITER`` blocks at
a time: it looks the block ids up in the block table, gathers the keys and values, computes
the scores with one matmul, and folds them into a running softmax (the online
max/sum/accumulator recurrence), so nothing proportional to the context length is ever
materialized.

**Shapes.** ``tl.dot`` needs both operands to have at least 16 rows/columns, so the query
group is padded to 16 rows (masked on load and on store).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

_GROUP_ROWS = 16


@triton.jit
def _paged_decode_kernel(
    q_ptr,
    kv_ptr,
    out_ptr,
    block_table_ptr,
    seq_len_ptr,
    stride_q_batch,
    stride_q_head,
    stride_kv_block,
    stride_kv_select,
    stride_kv_token,
    stride_kv_head,
    stride_out_batch,
    stride_out_head,
    stride_table,
    scale,
    GROUP: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TOKENS_PER_ITER: tl.constexpr,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    seq_len = tl.load(seq_len_ptr + batch)

    rows = tl.arange(0, GROUP_ROWS)
    dims = tl.arange(0, HEAD_DIM)
    row_mask = rows < GROUP
    head_offsets = (kv_head * GROUP + rows) * stride_q_head
    q = tl.load(
        q_ptr + batch * stride_q_batch + head_offsets[:, None] + dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )

    running_max = tl.full([GROUP_ROWS], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([GROUP_ROWS], dtype=tl.float32)
    acc = tl.zeros([GROUP_ROWS, HEAD_DIM], dtype=tl.float32)

    tokens = tl.arange(0, TOKENS_PER_ITER)
    table_row = block_table_ptr + batch * stride_table
    for start in range(0, seq_len, TOKENS_PER_ITER):
        position = start + tokens
        valid = position < seq_len
        block = tl.load(table_row + position // BLOCK_SIZE, mask=valid, other=0)
        base = (
            block.to(tl.int64) * stride_kv_block
            + (position % BLOCK_SIZE) * stride_kv_token
            + kv_head * stride_kv_head
        )
        offsets = base[:, None] + dims[None, :]
        k = tl.load(kv_ptr + offsets, mask=valid[:, None], other=0.0)
        v = tl.load(kv_ptr + stride_kv_select + offsets, mask=valid[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        scores = tl.where(valid[None, :], scores, float("-inf"))
        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        decay = tl.exp(running_max - new_max)
        weights = tl.exp(scores - new_max[:, None])
        running_sum = running_sum * decay + tl.sum(weights, axis=1)
        acc = acc * decay[:, None] + tl.dot(weights.to(v.dtype), v, out_dtype=tl.float32)
        running_max = new_max

    out = acc / running_sum[:, None]
    tl.store(
        out_ptr
        + batch * stride_out_batch
        + (kv_head * GROUP + rows)[:, None] * stride_out_head
        + dims[None, :],
        out.to(q.dtype),
        mask=row_mask[:, None],
    )


def paged_decode_attention(
    q: Tensor,
    paged_kv: Tensor,
    block_table: Tensor,
    seq_lens: Tensor,
    out: Tensor | None = None,
    pages_per_iter: int = 4,
) -> Tensor:
    """Decode attention over a paged KV cache.

    Args:
        q: ``[batch, num_heads, head_dim]`` queries (already rotated).
        paged_kv: ``[num_blocks, 2, block_size, num_kv_heads, head_dim]``; index 0/1 on the
            second axis selects keys/values. Must contain the current token's own K/V.
        block_table: ``[batch, max_pages]`` int32 block ids of each sequence, in position
            order; entries past a sequence's last page are ignored.
        seq_lens: ``[batch]`` int32 number of cached positions per sequence (≥ 1).
        out: Optional ``[batch, num_heads, head_dim]`` output buffer.
        pages_per_iter: Blocks processed per loop iteration.

    Returns:
        ``[batch, num_heads, head_dim]`` attention outputs, in ``q``'s dtype.
    """
    batch, num_heads, head_dim = q.shape
    _, _, block_size, num_kv_heads, _ = paged_kv.shape
    group = num_heads // num_kv_heads
    if group * num_kv_heads != num_heads or group > _GROUP_ROWS:
        raise ValueError(f"unsupported head configuration {num_heads}/{num_kv_heads}")
    tokens_per_iter = pages_per_iter * block_size
    if tokens_per_iter < 16:
        raise ValueError("pages_per_iter * block_size must be at least 16")
    if out is None:
        out = torch.empty_like(q)
    _paged_decode_kernel[(batch, num_kv_heads)](
        q,
        paged_kv,
        out,
        block_table,
        seq_lens,
        q.stride(0),
        q.stride(1),
        paged_kv.stride(0),
        paged_kv.stride(1),
        paged_kv.stride(2),
        paged_kv.stride(3),
        out.stride(0),
        out.stride(1),
        block_table.stride(0),
        head_dim**-0.5,
        GROUP=group,
        GROUP_ROWS=_GROUP_ROWS,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        TOKENS_PER_ITER=tokens_per_iter,
    )
    return out
