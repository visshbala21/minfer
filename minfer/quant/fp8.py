"""FP8 (E4M3) linear layers: 8-bit weights and dynamically quantized activations.

A linear layer ``y = x Wᵀ + b`` is evaluated as an FP8×FP8 matmul with a bf16 result:

* **Weights** are quantized once, at load: ``W ≈ s_w · W₈`` with one scale per tensor
  mapping the largest magnitude onto E4M3's maximum (448).
* **Activations** are quantized on the fly: ``x ≈ s_x · x₈`` with ``s_x`` from the batch's
  own largest magnitude, so no calibration data is needed and nothing saturates.

E4M3 is a floating-point format (4 exponent bits, 3 mantissa bits): its *relative*
precision — a rounding error of at most 1/16 — holds from the scale's maximum down to about
3·10⁻⁵ of it. A single tensor-wide scale is therefore enough even for activations with a
few very large dimensions; values more than four decades below the maximum are the ones
that lose precision.

On CUDA the product runs on FP8 tensor cores through ``torch._scaled_mm`` (cuBLASLt). On
other devices the layer dequantizes and uses an ordinary matmul, which computes the same
quantized function and is what the CPU tests exercise.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


class QuantizedActivation(NamedTuple):
    """An activation tensor in FP8 with its scale: ``original ≈ values · scale``."""

    values: Tensor
    scale: Tensor
    dtype: torch.dtype


def quantize_activation(x: Tensor) -> QuantizedActivation:
    """Quantize ``x`` (``[n, features]``) to FP8 with a scale from its largest magnitude."""
    scale = x.abs().amax().float().clamp_(min=1e-12) / FP8_MAX
    return QuantizedActivation((x.float() / scale).to(FP8), scale, x.dtype)


class FP8Linear(nn.Module):
    """``nn.Linear`` replacement holding its weight in FP8.

    Call it with a tensor, or with a :class:`QuantizedActivation` to share one activation
    quantization between several layers that read the same input (q/k/v, gate/up).
    """

    weight: Tensor
    weight_scale: Tensor
    bias: Tensor | None

    def __init__(self, weight: Tensor, bias: Tensor | None) -> None:
        super().__init__()
        scale = weight.abs().amax().float().clamp_(min=1e-12) / FP8_MAX
        self.register_buffer("weight", (weight.float() / scale).to(FP8).contiguous())
        self.register_buffer("weight_scale", scale)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.out_features, self.in_features = weight.shape

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> FP8Linear:
        """Quantize an existing linear layer."""
        return cls(linear.weight.detach(), linear.bias)

    def dequantized_weight(self) -> Tensor:
        """The weight this layer actually applies, in float32."""
        return self.weight.float() * self.weight_scale

    def forward(self, x: Tensor | QuantizedActivation) -> Tensor:
        """Apply the layer to ``x`` (``[n, in_features]``)."""
        if not isinstance(x, QuantizedActivation):
            x = quantize_activation(x)
        if x.values.is_cuda:
            return torch._scaled_mm(
                x.values,
                self.weight.t(),
                scale_a=x.scale,
                scale_b=self.weight_scale,
                bias=self.bias,
                out_dtype=x.dtype,
            )
        activation = x.values.float() * x.scale
        bias = None if self.bias is None else self.bias.float()
        return F.linear(activation, self.dequantized_weight(), bias).to(x.dtype)

    def extra_repr(self) -> str:
        """Shape summary for ``print(module)``."""
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8"


def quantize_linears(model: nn.Module, skip: tuple[str, ...] = ("lm_head",)) -> int:
    """Replace every ``nn.Linear`` in ``model`` by an :class:`FP8Linear`, in place.

    Args:
        model: The module tree to convert.
        skip: Attribute names to leave alone. The unembedding stays in the model dtype: it
            is tied to the embedding table and maps directly to logits.

    Returns:
        The number of layers converted.
    """
    converted = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name not in skip:
                setattr(parent, name, FP8Linear.from_linear(child))
                converted += 1
    return converted
