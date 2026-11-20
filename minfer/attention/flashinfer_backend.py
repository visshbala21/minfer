"""Paged attention through FlashInfer's batch prefill and batch decode kernels."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import Tensor

from minfer.attention.base import AttentionContext
from minfer.attention.paged import PagedBatch, dense_block_table
from minfer.cache.kv_store import PagedKVStore


def _expose_venv_bin() -> None:
    """Put the interpreter's ``bin/`` on ``PATH``.

    FlashInfer JIT-compiles kernels by invoking ``ninja`` by bare name. The binary is
    installed next to the interpreter, which is only on ``PATH`` in an activated virtualenv.
    """
    bin_dir = str(Path(sys.executable).parent)
    paths = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in paths:
        os.environ["PATH"] = os.pathsep.join([bin_dir, *paths])


_expose_venv_bin()

import flashinfer  # noqa: E402 - must follow the PATH fix above

_WORKSPACE_BYTES = 256 * 1024 * 1024


Wrapper = (
    flashinfer.BatchPrefillWithPagedKVCacheWrapper | flashinfer.BatchDecodeWithPagedKVCacheWrapper
)


class KernelContext:
    """Writes the pass's K/V into the store, then runs a planned FlashInfer kernel.

    ``wrappers`` maps a KV storage dtype to the wrapper planned for it: a mixed-precision
    store has layers in two dtypes, and a FlashInfer kernel is specialized for one.
    """

    def __init__(
        self, store: PagedKVStore, slots: Tensor, wrappers: dict[torch.dtype, Wrapper]
    ) -> None:
        self.store = store
        self.slots = slots
        self.layer_wrappers = [wrappers[store.layer_dtype(i)] for i in range(store.num_layers)]

    def attend(self, layer: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """See :meth:`minfer.attention.base.AttentionContext.attend`."""
        store = self.store
        store.write(layer, self.slots, k, v)
        wrapper = self.layer_wrappers[layer]
        if not store.layer_is_fp8[layer]:
            return wrapper.run(q, store.layer(layer))
        # FP8 layer: the kernel multiplies the stored values back by the layer's scales.
        k_scale, v_scale = store.scale_pairs[layer]
        return wrapper.run(q, store.layer(layer), k_scale=k_scale, v_scale=v_scale)


def needs_tensor_cores(num_heads: int, num_kv_heads: int) -> bool:
    """Whether FlashInfer's decode must take its tensor-core path for this head layout.

    Its plain decode kernel is specialized for grouped-query group sizes 1, 2, 3, 4, 6, 8
    and 16 only (Qwen2.5-0.5B has 14 query heads over 2 KV heads: group size 7).
    """
    return num_heads // num_kv_heads not in (1, 2, 3, 4, 6, 8, 16)


class TritonDecodeContext:
    """Writes the pass's K/V into the store, then runs the Triton paged-decode kernel."""

    def __init__(
        self, store: PagedKVStore, slots: Tensor, block_table: Tensor, seq_lens: Tensor
    ) -> None:
        self.store = store
        self.slots = slots
        self.block_table = block_table
        self.seq_lens = seq_lens

    def attend(self, layer: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """See :meth:`minfer.attention.base.AttentionContext.attend`."""
        from minfer.attention.triton_decode import paged_decode_attention

        self.store.write(layer, self.slots, k, v)
        return paged_decode_attention(q, self.store.layer(layer), self.block_table, self.seq_lens)


class FlashInferBackend:
    """FlashInfer-backed implementation of :class:`PagedAttentionBackend`.

    One plan per forward pass is shared by all layers (the page tables are the same for
    every layer). RoPE is applied by the model, so the kernels run with no positional
    encoding. Both wrappers share one float workspace buffer; they are never used at the
    same time.

    ``decode_kernel="triton"`` swaps FlashInfer's decode kernel for the one in
    :mod:`minfer.attention.triton_decode`; prefill always goes through FlashInfer.
    """

    def __init__(
        self, store: PagedKVStore, num_heads: int, decode_kernel: str = "flashinfer"
    ) -> None:
        if store.device.type != "cuda":
            raise ValueError("FlashInfer needs the KV store on a CUDA device")
        self.store = store
        self.decode_kernel = decode_kernel
        self.num_heads = num_heads
        self.num_kv_heads, self.head_dim = store.layer(0).shape[3:]
        self.dtype = store.compute_dtype
        self.workspace = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=store.device)
        tensor_cores = needs_tensor_cores(num_heads, self.num_kv_heads)
        # One wrapper per KV storage dtype in the store (two when it is mixed-precision).
        self._prefill = {
            kv_dtype: flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, "NHD")
            for kv_dtype in store.tensors
        }
        self._decode = {
            kv_dtype: flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace, "NHD", use_tensor_cores=tensor_cores
            )
            for kv_dtype in store.tensors
        }

    def prefill(self, batch: PagedBatch) -> AttentionContext:
        """See :meth:`PagedAttentionBackend.prefill`."""
        for kv_dtype, wrapper in self._prefill.items():
            wrapper.plan(
                batch.qo_indptr,
                batch.kv_indptr,
                batch.kv_indices,
                batch.kv_last_page_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                batch.block_size,
                causal=True,
                pos_encoding_mode="NONE",
                q_data_type=self.dtype,
                kv_data_type=kv_dtype,
            )
        return KernelContext(self.store, batch.slots, self._prefill)

    def decode(self, batch: PagedBatch) -> AttentionContext:
        """See :meth:`PagedAttentionBackend.decode`."""
        if self.decode_kernel == "triton":
            table, lengths = dense_block_table(batch)
            device = self.store.device
            return TritonDecodeContext(
                self.store, batch.slots, table.to(device), lengths.to(device)
            )
        for kv_dtype, wrapper in self._decode.items():
            wrapper.plan(
                batch.kv_indptr,
                batch.kv_indices,
                batch.kv_last_page_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                batch.block_size,
                pos_encoding_mode="NONE",
                q_data_type=self.dtype,
                kv_data_type=kv_dtype,
            )
        return KernelContext(self.store, batch.slots, self._decode)
