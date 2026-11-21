"""The paged KV tensors."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

FP8 = torch.float8_e4m3fn
# Largest finite E4M3 value. Quantization maps a layer's largest |value| onto it.
FP8_MAX = 448.0


class PagedKVStore:
    """Key/value storage for every layer, addressed by block.

    Per layer, a tensor of shape ``[num_blocks, 2, block_size, num_kv_heads, head_dim]``;
    index 0/1 on the second axis selects keys/values. ``layer(i)`` is the 5-D paged cache
    in the "NHD" layout FlashInfer consumes.

    A token's *slot* is ``block * 2 * block_size + offset``: the row of its key in the
    layer's flattened ``[num_blocks * 2 * block_size, num_kv_heads, head_dim]`` view. Its
    value lives ``block_size`` rows later. Addressing by slot turns a KV write into a single
    ``index_copy_`` per tensor, with index tensors whose shape depends only on the number of
    tokens written (which is what CUDA-graph capture needs).

    **Mixed-precision FP8 storage.** With ``kv_dtype=torch.float8_e4m3fn`` a layer's
    elements take one byte instead of two. E4M3 has 3 mantissa bits: a *relative* rounding
    error of up to 1/16, whatever the scale. Values are divided by a per-layer scale before
    the cast so the layer's largest magnitude lands on 448 and nothing saturates, and
    multiplied back when read (FlashInfer does that inside its kernels, given the scales).
    A scale cannot shrink a relative error, though, and a layer whose keys have very large
    components turns 6% of those into attention-logit errors of many units. Such layers are
    listed in ``high_precision_layers`` and keep the model dtype; the two groups live in
    two tensors. Which layers those are, and the scales, come from calibration
    (:mod:`minfer.quant.calibration`).
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
        kv_dtype: torch.dtype | None = None,
        high_precision_layers: Iterable[int] = (),
    ) -> None:
        if kv_dtype not in (None, dtype, FP8):
            raise ValueError(f"unsupported KV dtype {kv_dtype}")
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.compute_dtype = dtype
        fp8 = kv_dtype == FP8
        self.high_precision_layers = frozenset(high_precision_layers) if fp8 else frozenset()
        # Which layers are stored in FP8.
        self.layer_is_fp8 = [fp8 and i not in self.high_precision_layers for i in range(num_layers)]
        self.quantized = any(self.layer_is_fp8)

        shape = (num_blocks, 2, block_size, num_kv_heads, head_dim)
        flat_shape = (num_blocks * 2 * block_size, num_kv_heads, head_dim)
        # One tensor per storage dtype, zero-initialized so that slots nobody has written
        # (padding rows read block 0) hold finite values.
        self.tensors: dict[torch.dtype, Tensor] = {}
        for layer_dtype in (FP8, dtype):
            count = sum(is_fp8 == (layer_dtype == FP8) for is_fp8 in self.layer_is_fp8)
            if count:
                self.tensors[layer_dtype] = torch.zeros(
                    (count, *shape), device=device, dtype=layer_dtype
                )
        self._layers: list[Tensor] = []
        self._flat: list[Tensor] = []
        seen = {FP8: 0, dtype: 0}
        for is_fp8 in self.layer_is_fp8:
            layer_dtype = FP8 if is_fp8 else dtype
            tensor = self.tensors[layer_dtype][seen[layer_dtype]]
            seen[layer_dtype] += 1
            self._layers.append(tensor)
            flat = tensor.view(flat_shape)
            # index_copy_/index_select are not implemented for FP8; the same bytes viewed
            # as uint8 are.
            self._flat.append(flat.view(torch.uint8) if is_fp8 else flat)
        self.k_scale = torch.ones(num_layers, device=device, dtype=torch.float32)
        self.v_scale = torch.ones(num_layers, device=device, dtype=torch.float32)
        # The same scales as host floats (kernels take them as scalars). Reading device
        # tensors per pass would sync, and cannot be done during CUDA-graph capture.
        self.scale_pairs: list[tuple[float, float]] = [(1.0, 1.0)] * num_layers

    @property
    def nbytes(self) -> int:
        """Bytes of device memory held by the store."""
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    @property
    def device(self) -> torch.device:
        """Device the store lives on."""
        return self._layers[0].device

    def layer(self, index: int) -> Tensor:
        """Layer ``index``'s cache, ``[num_blocks, 2, block_size, num_kv_heads, head_dim]``."""
        return self._layers[index]

    def layer_dtype(self, index: int) -> torch.dtype:
        """Storage dtype of layer ``index``."""
        return FP8 if self.layer_is_fp8[index] else self.compute_dtype

    def set_scales(self, k_scale: Tensor, v_scale: Tensor) -> None:
        """Install per-layer FP8 scales (``[num_layers]`` each).

        Must be called before anything is written: stored values are only meaningful
        together with the scales they were written with.
        """
        self.k_scale.copy_(k_scale)
        self.v_scale.copy_(v_scale)
        self.scale_pairs = list(zip(self.k_scale.tolist(), self.v_scale.tolist(), strict=True))

    def write(self, layer: int, slots: Tensor, k: Tensor, v: Tensor) -> None:
        """Store keys and values (``[n, num_kv_heads, head_dim]``) at ``slots`` (``[n]``)."""
        flat = self._flat[layer]
        if self.layer_is_fp8[layer]:
            k = (k.float() / self.k_scale[layer]).to(FP8).view(torch.uint8)
            v = (v.float() / self.v_scale[layer]).to(FP8).view(torch.uint8)
        flat.index_copy_(0, slots, k)
        flat.index_copy_(0, slots + self.block_size, v)

    def write_target(self, layer: int) -> tuple[Tensor, int] | None:
        """Where a kernel can store layer ``layer``'s keys and values itself.

        Returns the layer as ``[slots, num_kv_heads, head_dim]`` and the distance from a
        token's key slot to its value slot, or ``None`` for an FP8 layer (whose values must
        be scaled and cast on the way in: use :meth:`write`).
        """
        if self.layer_is_fp8[layer]:
            return None
        return self._flat[layer], self.block_size

    def copy_rotated(self, source: Tensor, target: Tensor, cos: Tensor, sin: Tensor) -> None:
        """Copy keys and values between slots in every layer, rotating the keys.

        The store holds keys with their rotary position encoding applied. A key cached at
        position ``p`` and reused at position ``p + Δ`` is the same vector rotated by Δ
        steps (rotations compose), so the copy needs no unrotated keys; values carry no
        position and are copied as they are.

        Args:
            source: ``[n]`` int64 slots to read (see the class docstring for slots).
            target: ``[n]`` int64 slots to write.
            cos: ``[n, head_dim]`` float32 cosines of each token's Δ·θ, in the layout of the
                model's RoPE tables (the frequency vector twice).
            sin: ``[n, head_dim]`` float32 sines, likewise (negative Δ: negative sines).
        """
        if self.quantized:
            raise ValueError("position-independent reuse does not support an FP8 KV cache")
        size = self.block_size
        tensor = self.tensors[self.compute_dtype]
        flat = tensor.view(tensor.shape[0], -1, *tensor.shape[4:])  # [layers, slots, heads, dim]
        keys = flat.index_select(1, source).float()
        half = keys.shape[-1] // 2
        swapped = torch.cat((-keys[..., half:], keys[..., :half]), dim=-1)
        rotated = keys * cos[None, :, None, :] + swapped * sin[None, :, None, :]
        values = flat.index_select(1, source + size)
        flat.index_copy_(1, target, rotated.to(flat.dtype))
        flat.index_copy_(1, target + size, values)

    def read(self, layer: int, slots: Tensor) -> tuple[Tensor, Tensor]:
        """Gather the keys and values stored at ``slots`` (dequantized if stored as FP8)."""
        flat = self._flat[layer]
        k, v = flat.index_select(0, slots), flat.index_select(0, slots + self.block_size)
        if not self.layer_is_fp8[layer]:
            return k, v
        k = k.view(FP8).float() * self.k_scale[layer]
        v = v.view(FP8).float() * self.v_scale[layer]
        return k.to(self.compute_dtype), v.to(self.compute_dtype)
