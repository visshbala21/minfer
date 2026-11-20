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


class KernelContext:
    """Writes the pass's K/V into the store, then runs a planned FlashInfer kernel."""

    def __init__(
        self,
        store: PagedKVStore,
        slots: Tensor,
        wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper
        | flashinfer.BatchDecodeWithPagedKVCacheWrapper,
    ) -> None:
        self.store = store
        self.slots = slots
        self.wrapper = wrapper
        # FP8 cache: the kernel multiplies the stored values back by these per-layer scales.
        self.scales = (
            list(zip(store.k_scale.tolist(), store.v_scale.tolist(), strict=True))
            if store.quantized
            else None
        )

    def attend(self, layer: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """See :meth:`minfer.attention.base.AttentionContext.attend`."""
        store = self.store
        store.write(layer, self.slots, k, v)
        if self.scales is None:
            return self.wrapper.run(q, store.layer(layer))
        k_scale, v_scale = self.scales[layer]
        return self.wrapper.run(q, store.layer(layer), k_scale=k_scale, v_scale=v_scale)


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
        if not store.data.is_cuda:
            raise ValueError("FlashInfer needs the KV store on a CUDA device")
        self.store = store
        self.decode_kernel = decode_kernel
        self.num_heads = num_heads
        self.num_kv_heads = store.data.shape[4]
        self.head_dim = store.data.shape[5]
        self.dtype = store.compute_dtype
        self.kv_dtype = store.data.dtype
        self.workspace = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=store.data.device)
        self._prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, "NHD")
        self._decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.workspace, "NHD")

    def prefill(self, batch: PagedBatch) -> AttentionContext:
        """See :meth:`PagedAttentionBackend.prefill`."""
        self._prefill.plan(
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
            kv_data_type=self.kv_dtype,
        )
        return KernelContext(self.store, batch.slots, self._prefill)

    def decode(self, batch: PagedBatch) -> AttentionContext:
        """See :meth:`PagedAttentionBackend.decode`."""
        if self.decode_kernel == "triton":
            table, lengths = dense_block_table(batch)
            device = self.store.data.device
            return TritonDecodeContext(
                self.store, batch.slots, table.to(device), lengths.to(device)
            )
        self._decode.plan(
            batch.kv_indptr,
            batch.kv_indices,
            batch.kv_last_page_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            batch.block_size,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
            kv_data_type=self.kv_dtype,
        )
        return KernelContext(self.store, batch.slots, self._decode)
