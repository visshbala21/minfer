"""The paged KV tensors."""

from __future__ import annotations

import torch
from torch import Tensor


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
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Zero-initialized so that slots nobody has written (padding rows read block 0)
        # hold finite values.
        self.data = torch.zeros(
            (num_layers, num_blocks, 2, block_size, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._flat = self.data.view(num_layers, num_blocks * 2 * block_size, num_kv_heads, head_dim)

    @property
    def nbytes(self) -> int:
        """Bytes of device memory held by the store."""
        return self.data.numel() * self.data.element_size()

    def layer(self, index: int) -> Tensor:
        """Layer ``index``'s cache, ``[num_blocks, 2, block_size, num_kv_heads, head_dim]``."""
        return self.data[index]

    def write(self, layer: int, slots: Tensor, k: Tensor, v: Tensor) -> None:
        """Store keys and values (``[n, num_kv_heads, head_dim]``) at ``slots`` (``[n]``)."""
        flat = self._flat[layer]
        flat.index_copy_(0, slots, k)
        flat.index_copy_(0, slots + self.block_size, v)

    def read(self, layer: int, slots: Tensor) -> tuple[Tensor, Tensor]:
        """Gather the keys and values stored at ``slots``."""
        flat = self._flat[layer]
        return flat.index_select(0, slots), flat.index_select(0, slots + self.block_size)
