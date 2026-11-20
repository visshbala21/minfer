"""Calibration of the FP8 KV cache: per-layer scales and which layers can take FP8 at all.

Calibration needs the *range* of each layer's keys and values and how much rounding the
keys perturbs the attention logits. Both are properties of the model far more than of the
text, so a handful of short, varied passages is enough (prose, code, structured data, a
chat exchange, non-English text).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from minfer.attention.base import AttentionContext
from minfer.cache.kv_store import FP8, FP8_MAX

CALIBRATION_TEXTS = (
    "The committee met on Thursday to review the budget for the coming year. Revenue was "
    "higher than forecast, mostly because of two large contracts signed in the spring, but "
    "costs had risen as well, and the treasurer warned that the reserve would not last.",
    "def merge_intervals(intervals):\n    intervals.sort(key=lambda pair: pair[0])\n"
    "    merged = [intervals[0]]\n    for start, end in intervals[1:]:\n"
    "        if start <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], end)\n"
    "        else:\n            merged.append([start, end])\n    return merged\n",
    '{"order_id": 48213, "customer": {"name": "R. Okafor", "tier": "gold"}, "items": '
    '[{"sku": "A-119", "qty": 3, "price": 12.5}, {"sku": "C-702", "qty": 1, "price": 249.0}], '
    '"shipping": {"method": "express", "eta_days": 2}, "total": 286.5}',
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    "How do I undo the last commit but keep my changes?<|im_end|>\n<|im_start|>assistant\n"
    "Use `git reset --soft HEAD~1`. The commit is removed and your changes stay staged."
    "<|im_end|>\n",
    "量子计算利用量子比特的叠加和纠缠来处理信息。与经典比特不同，量子比特可以同时处于多个状态，"
    "这使得某些问题的求解速度远远超过传统计算机。",
    'Traceback (most recent call last):\n  File "app/server.py", line 212, in handle\n'
    "    payload = json.loads(body)\njson.decoder.JSONDecodeError: Expecting value: line 1 "
    "column 1 (char 0)\n2026-03-14 09:21:07,113 ERROR request 7f3a failed after 3 retries",
    "| region | q1 | q2 | q3 | q4 |\n|---|---|---|---|---|\n| north | 1,204 | 1,377 | 1,102 | "
    "1,560 |\n| south | 980 | 1,015 | 1,240 | 1,198 |\nTotal growth was 7.4% year over year.",
    "Step 1: 48 / 6 = 8 boxes per shelf. Step 2: 8 * 3 = 24 boxes on three shelves. Step 3: "
    "24 - 5 = 19 boxes remain after the delivery. The answer is 19.",
)


@dataclass
class KVCalibration:
    """What calibration learned about a model's keys and values.

    Attributes:
        k_scale: ``[num_layers]`` FP8 scale of each layer's keys.
        v_scale: ``[num_layers]`` FP8 scale of each layer's values.
        logit_error: ``[num_layers]`` mean absolute change of the attention logits when the
            layer's keys are rounded to FP8.
        high_precision_layers: Layers whose keys must not be stored in FP8.
    """

    k_scale: Tensor
    v_scale: Tensor
    logit_error: Tensor
    high_precision_layers: tuple[int, ...]


class CalibrationContext:
    """Attention context wrapper that measures what FP8 storage would do to each layer.

    For every layer it records the largest key and value magnitudes (→ scales) and the
    mean absolute error that rounding the keys to E4M3 would add to the attention logits
    ``q·k/√d`` — the quantity that decides whether FP8 keys are tolerable. E4M3's rounding
    error is relative (up to 1/16 of each element), so a layer whose keys and queries have
    a few very large components can see logit errors of several units no matter how it is
    scaled; ordinary layers see about 0.1.
    """

    def __init__(self, inner: AttentionContext, peaks: Tensor, errors: Tensor) -> None:
        self.inner = inner
        self.peaks = peaks
        self.errors = errors

    def attend(self, layer: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """See :meth:`minfer.attention.base.AttentionContext.attend`."""
        keys = k.float()
        k_peak, v_peak = keys.abs().max(), v.float().abs().max()
        self.peaks[layer, 0] = torch.maximum(self.peaks[layer, 0], k_peak)
        self.peaks[layer, 1] = torch.maximum(self.peaks[layer, 1], v_peak)
        scale = k_peak.clamp(min=1e-6) / FP8_MAX
        rounding = (keys / scale).to(FP8).float() * scale - keys
        group = q.shape[1] // k.shape[1]
        logits = torch.einsum("nhd,mhd->hnm", q.float(), rounding.repeat_interleave(group, dim=1))
        error = logits.abs().mean() * q.shape[-1] ** -0.5
        self.errors[layer] = torch.maximum(self.errors[layer], error)
        return self.inner.attend(layer, q, k, v)


def summarize(
    peaks: Tensor, errors: Tensor, headroom: float = 1.25, max_logit_error: float = 1.0
) -> KVCalibration:
    """Turn observed peaks and logit errors into scales and the high-precision layer set.

    Args:
        peaks: ``[num_layers, 2]`` largest key and value magnitudes.
        errors: ``[num_layers]`` mean absolute logit error from FP8 keys.
        headroom: Margin above the observed peak before values saturate. Cheap: E4M3 keeps
            its relative precision down to ``2^-6`` of the scale's maximum.
        max_logit_error: Layers above this stay in the model dtype. Attention logits are
            compared through a softmax, so an error of 1 already changes weights by e.
    """
    scaled = peaks.clamp(min=1e-6) * headroom / FP8_MAX
    sensitive = tuple(i for i, e in enumerate(errors.tolist()) if e > max_logit_error)
    return KVCalibration(scaled[:, 0].clone(), scaled[:, 1].clone(), errors.clone(), sensitive)
