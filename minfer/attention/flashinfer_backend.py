"""Paged attention through FlashInfer's batch prefill and batch decode kernels."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import Tensor

from minfer.attention.base import AttentionContext
from minfer.attention.paged import PagedBatch, dense_block_table, split_batch
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


def _destination(
    store: PagedKVStore, layer: int, slots: Tensor
) -> tuple[Tensor, Tensor, int] | None:
    """A pass's write target in ``store`` (see ``StoringAttentionContext.kv_destination``)."""
    target = store.write_target(layer)
    return None if target is None else (target[0], slots, target[1])


def _run(
    wrapper: Wrapper, store: PagedKVStore, layer: int, q: Tensor, out: Tensor | None = None
) -> Tensor:
    """Run a planned kernel over ``layer``'s cache (into ``out``, if given)."""
    if not store.layer_is_fp8[layer]:
        return wrapper.run(q, store.layer(layer), out=out)
    # FP8 layer: the kernel multiplies the stored values back by the layer's scales.
    k_scale, v_scale = store.scale_pairs[layer]
    return wrapper.run(q, store.layer(layer), k_scale=k_scale, v_scale=v_scale, out=out)


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
        self.store.write(layer, self.slots, k, v)
        return self.attend_stored(layer, q)

    def kv_destination(self, layer: int) -> tuple[Tensor, Tensor, int] | None:
        """See :meth:`minfer.attention.base.StoringAttentionContext.kv_destination`."""
        return _destination(self.store, layer, self.slots)

    def attend_stored(self, layer: int, q: Tensor) -> Tensor:
        """See :meth:`minfer.attention.base.StoringAttentionContext.attend_stored`."""
        return _run(self.layer_wrappers[layer], self.store, layer, q)


class JointContext:
    """One pass, two kernels: prefill spans first, then one-token decode rows.

    The first ``num_prefill_rows`` queries go through a planned batch-prefill kernel, the
    rest through a planned batch-decode kernel, each writing its part of one output. The
    linear layers around attention see a single batch, so a step that both prefills and
    decodes reads the weights once — and its decoding sequences keep the decode kernel,
    which serves one-token queries over long contexts much better than the prefill kernel
    does (DESIGN.md M-025, M-032).
    """

    def __init__(
        self,
        store: PagedKVStore,
        slots: Tensor,
        prefill: dict[torch.dtype, Wrapper],
        decode: dict[torch.dtype, Wrapper],
        num_prefill_rows: int,
    ) -> None:
        self.store = store
        self.slots = slots
        self.num_prefill_rows = num_prefill_rows
        dtypes = [store.layer_dtype(i) for i in range(store.num_layers)]
        self.prefill_wrappers = [prefill[dtype] for dtype in dtypes]
        self.decode_wrappers = [decode[dtype] for dtype in dtypes]

    def attend(self, layer: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """See :meth:`minfer.attention.base.AttentionContext.attend`."""
        self.store.write(layer, self.slots, k, v)
        return self.attend_stored(layer, q)

    def kv_destination(self, layer: int) -> tuple[Tensor, Tensor, int] | None:
        """See :meth:`minfer.attention.base.StoringAttentionContext.kv_destination`."""
        return _destination(self.store, layer, self.slots)

    def attend_stored(self, layer: int, q: Tensor) -> Tensor:
        """See :meth:`minfer.attention.base.StoringAttentionContext.attend_stored`."""
        rows = self.num_prefill_rows
        out = torch.empty_like(q)
        _run(self.prefill_wrappers[layer], self.store, layer, q[:rows], out[:rows])
        _run(self.decode_wrappers[layer], self.store, layer, q[rows:], out[rows:])
        return out


def needs_tensor_cores(num_heads: int, num_kv_heads: int) -> bool:
    """Whether FlashInfer's decode should take its tensor-core path for this head layout.

    Yes for grouped-query group sizes of 4 and above: that is the rule SGLang applies, and
    for group size 6 the tensor-core path is the faster one at every batch size and context
    measured (DESIGN.md M-019). FlashInfer's plain decode kernel is also only specialized
    for group sizes 1, 2, 3, 4, 6, 8 and 16 (Qwen2.5-0.5B and 7B have group size 7).
    """
    return num_heads // num_kv_heads >= 4


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
        self.store.write(layer, self.slots, k, v)
        return self.attend_stored(layer, q)

    def kv_destination(self, layer: int) -> tuple[Tensor, Tensor, int] | None:
        """See :meth:`minfer.attention.base.StoringAttentionContext.kv_destination`."""
        return _destination(self.store, layer, self.slots)

    def attend_stored(self, layer: int, q: Tensor) -> Tensor:
        """See :meth:`minfer.attention.base.StoringAttentionContext.attend_stored`."""
        from minfer.attention.triton_decode import paged_decode_attention

        return paged_decode_attention(q, self.store.layer(layer), self.block_table, self.seq_lens)


class FlashInferBackend:
    """FlashInfer-backed implementation of :class:`PagedAttentionBackend`.

    One plan per forward pass is shared by all layers (the page tables are the same for
    every layer). RoPE is applied by the model, so the kernels run with no positional
    encoding. All wrappers share one float workspace buffer: kernels run one after another
    on one stream, and none keeps state in it between runs.

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

    def _plan_prefill(self, batch: PagedBatch) -> None:
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

    def _plan_decode(self, batch: PagedBatch) -> None:
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

    def prefill(self, batch: PagedBatch) -> AttentionContext:
        """See :meth:`PagedAttentionBackend.prefill`."""
        self._plan_prefill(batch)
        return KernelContext(self.store, batch.slots, self._prefill)

    def decode(self, batch: PagedBatch) -> AttentionContext:
        """See :meth:`PagedAttentionBackend.decode`."""
        if self.decode_kernel == "triton":
            table, lengths = dense_block_table(batch)
            device = self.store.device
            return TritonDecodeContext(
                self.store, batch.slots, table.to(device), lengths.to(device)
            )
        self._plan_decode(batch)
        return KernelContext(self.store, batch.slots, self._decode)

    def joint(self, batch: PagedBatch, num_prefill_spans: int) -> AttentionContext:
        """See :meth:`PagedAttentionBackend.joint`.

        The decode rows of a joint pass always take FlashInfer's decode kernel, whatever
        ``decode_kernel`` selects for pure decode passes.
        """
        head, tail = split_batch(batch, num_prefill_spans)
        self._plan_prefill(head)
        self._plan_decode(tail)
        return JointContext(self.store, batch.slots, self._prefill, self._decode, head.num_tokens)
