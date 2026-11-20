"""Calibration text for FP8 KV-cache scales.

The scales only need the *range* of each layer's keys and values, which is a property of
the model far more than of the text, so a handful of short, varied passages is enough
(prose, code, structured data, a chat exchange, non-English text). Calibration adds 25%
headroom on top of what these produce (see ``PagedKVStore.calibrate``).
"""

from __future__ import annotations

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
