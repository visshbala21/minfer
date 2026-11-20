"""The paged KV tensors."""

from __future__ import annotations

import torch
from torch import Tensor

FP8 = torch.float8_e4m3fn
# Largest finite E4M3 value. Quantization maps a layer's largest |value| onto it.
FP8_MAX = 448.0


class PagedKVStore:
    """Key/value storage for every layer, addressed by block.

    One tensor of shape ``[num_layers, num_blocks, 2, block_size, num_kv_heads, head_dim]``;
    index 0/1 on the third axis selects keys/values. ``layer(i)`` is the 5-D paged cache in
    the "NHD" layout FlashInfer consumes.

    A token's *slot* is ``block * 2 * block_size + offset``: the row of its key in the
    layer's flattened ``[num_blocks * 2 * block_size, num_kv_heads, head_dim]`` view. Its
    value lives ``block_size`` rows later. Addressing by slot turns a KV write into a single
    ``index_copy_`` per tensor, with index tensors whose shape depends only on the number of
    tokens written (which is what CUDA-graph capture needs).

    **FP8 storage.** With ``kv_dtype=torch.float8_e4m3fn`` each element takes one byte
    instead of two. E4M3 has 3 mantissa bits (relative step 1/8) over the range
    ``[2^-6, 448]``; values are divided by a per-layer scale before the cast so that the
    layer's largest magnitude lands on 448 and nothing saturates, and multiplied back when
    read (FlashInfer does that inside its kernels, given the same scales). Scales default
    to 1 and are set by calibration (:meth:`observe` / :meth:`calibrate`).
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
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.compute_dtype = dtype
        self.quantized = kv_dtype == FP8
        if kv_dtype not in (None, dtype, FP8):
            raise ValueError(f"unsupported KV dtype {kv_dtype}")
        # Zero-initialized so that slots nobody has written (padding rows read block 0)
        # hold finite values.
        self.data = torch.zeros(
            (num_layers, num_blocks, 2, block_size, num_kv_heads, head_dim),
            device=device,
            dtype=kv_dtype or dtype,
        )
        shape = (num_layers, num_blocks * 2 * block_size, num_kv_heads, head_dim)
        self._flat = self.data.view(shape)
        # index_copy_/index_select are not implemented for FP8; the same bytes viewed as
        # uint8 are.
        self._flat_bytes = self._flat.view(torch.uint8) if self.quantized else None
        self.k_scale = torch.ones(num_layers, device=device, dtype=torch.float32)
        self.v_scale = torch.ones(num_layers, device=device, dtype=torch.float32)
        self._observed: Tensor | None = None

    @property
    def nbytes(self) -> int:
        """Bytes of device memory held by the store."""
        return self.data.numel() * self.data.element_size()

    def layer(self, index: int) -> Tensor:
        """Layer ``index``'s cache, ``[num_blocks, 2, block_size, num_kv_heads, head_dim]``."""
        return self.data[index]

    def write(self, layer: int, slots: Tensor, k: Tensor, v: Tensor) -> None:
        """Store keys and values (``[n, num_kv_heads, head_dim]``) at ``slots`` (``[n]``)."""
        if self._observed is not None:
            self._observed[layer, 0] = torch.maximum(self._observed[layer, 0], k.abs().max())
            self._observed[layer, 1] = torch.maximum(self._observed[layer, 1], v.abs().max())
        if self.quantized:
            flat = self._flat_bytes[layer]
            k = (k.float() / self.k_scale[layer]).to(FP8).view(torch.uint8)
            v = (v.float() / self.v_scale[layer]).to(FP8).view(torch.uint8)
        else:
            flat = self._flat[layer]
        flat.index_copy_(0, slots, k)
        flat.index_copy_(0, slots + self.block_size, v)

    def read(self, layer: int, slots: Tensor) -> tuple[Tensor, Tensor]:
        """Gather the keys and values stored at ``slots`` (dequantized if stored as FP8)."""
        if not self.quantized:
            flat = self._flat[layer]
            return flat.index_select(0, slots), flat.index_select(0, slots + self.block_size)
        flat = self._flat_bytes[layer]
        k = flat.index_select(0, slots).view(FP8).float() * self.k_scale[layer]
        v = flat.index_select(0, slots + self.block_size).view(FP8).float() * self.v_scale[layer]
        return k.to(self.compute_dtype), v.to(self.compute_dtype)

    # ------------------------------------------------------------ calibration

    def observe(self) -> None:
        """Start recording the largest key and value magnitude written to each layer."""
        self._observed = torch.zeros(self.num_layers, 2, device=self.data.device)

    def calibrate(self, headroom: float = 1.25) -> None:
        """Set the FP8 scales from what was observed, and stop observing.

        ``headroom`` leaves room above the calibration maximum before values saturate. The
        cost of headroom is small: E4M3 keeps its relative precision for values down to
        ``scale * 2^-6``, about four orders of magnitude below the largest.

        Everything already in the store was written with the old scales and must be
        discarded by the caller.
        """
        if self._observed is None:
            raise RuntimeError("calibrate() without observe()")
        peak = self._observed.clamp(min=1e-6) * headroom
        self.k_scale.copy_(peak[:, 0] / FP8_MAX)
        self.v_scale.copy_(peak[:, 1] / FP8_MAX)
        self._observed = None
