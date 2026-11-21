"""Decode-step time at 7B, one decode-path change at a time, with a breakdown (Phase 16B).

    .venv/bin/python -m bench.decode_gap

For each engine configuration and each (batch size, context length), that many sequences
are prefilled and then decoded; every ``Engine.step()`` of steady-state decode is timed.
A second set of steps is timed with a device synchronization after the forward pass, which
splits a step into

* ``plan_ms``: attention planning on the host (page tables into the kernel's buffers),
* ``pass_ms``: the rest of the decode pass — building the batch, uploading it, replaying
  the CUDA graph, and waiting for the GPU to finish it,
* ``other_ms``: scheduling, sampling, reading the tokens back, bookkeeping.

Writes ``bench/results/decode_gap_7b.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bench import models
from bench.servers import code_version
from minfer.engine import Engine, EngineConfig
from minfer.sampling import SamplingParams

# name -> (label, EngineConfig overrides). Each adds one change to the one before it.
CONFIGS: dict[str, tuple[str, dict[str, Any]]] = {
    "separate_write": (
        "KV write as two index_copy_ per layer (before)",
        {"fused_kv_write": False},
    ),
    "fused_write": (
        "KV write fused into the RoPE kernel",
        {"fused_kv_write": True},
    ),
}
# (batch size, context tokens)
POINTS = ((1, 1024), (8, 1024), (32, 1024), (64, 1024), (128, 1024))
POINTS += ((1, 8192), (32, 8192), (64, 8192))


class _Timers:
    """Wraps the decode graphs' ``run`` and ``_plan`` to time them, synchronizing after a pass."""

    def __init__(self, engine: Engine) -> None:
        graphs = engine.decode_graphs
        self.plan = 0.0
        self.run = 0.0
        self.enabled = False
        inner_plan, inner_run = graphs._plan, graphs.run

        def plan(*args: Any) -> None:
            start = time.perf_counter()
            inner_plan(*args)
            self.plan += time.perf_counter() - start

        def run(*args: Any) -> torch.Tensor:
            start = time.perf_counter()
            logits = inner_run(*args)
            if self.enabled:
                torch.cuda.synchronize()
            self.run += time.perf_counter() - start
            return logits

        graphs._plan, graphs.run = plan, run


def measure(
    engine: Engine, timers: _Timers, batch_size: int, context: int, steps: int, warmup: int
) -> dict[str, Any]:
    """Time ``steps`` steady-state decode steps at ``batch_size``, twice (plain; with breakdown)."""
    rng = random.Random(batch_size * 100_003 + context)
    params = SamplingParams(max_tokens=warmup + 2 * steps + 256, ignore_eos=True)
    ids = [
        engine.add_request([rng.randrange(1000, 100_000) for _ in range(context)], params)
        for _ in range(batch_size)
    ]
    while len(engine.scheduler.running) < batch_size or any(
        seq.num_pending > 1 for seq in engine.scheduler.running
    ):
        engine.step()
    for _ in range(warmup):
        engine.step()
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        outputs = engine.step()
        times.append(time.perf_counter() - start)
        assert len(outputs) == batch_size
    timers.enabled, timers.plan, timers.run = True, 0.0, 0.0
    start = time.perf_counter()
    for _ in range(steps):
        engine.step()
    total = time.perf_counter() - start
    timers.enabled = False
    for seq_id in ids:
        engine.abort(seq_id)
    array = np.asarray(times) * 1e3
    return {
        "batch_size": batch_size,
        "context_tokens": context,
        "steps": steps,
        "step_ms_p50": float(np.percentile(array, 50)),
        "step_ms_p99": float(np.percentile(array, 99)),
        "tokens_per_second": batch_size / float(np.median(array)) * 1e3,
        "breakdown_ms": {
            "plan": timers.plan / steps * 1e3,
            "pass": (timers.run - timers.plan) / steps * 1e3,
            "other": (total - timers.run) / steps * 1e3,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=models.PRIMARY, help="model id, or 7b / 1.5b")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    model = models.resolve(args.model)
    out = args.out or models.results_path("bench/results/decode_gap.json", model)
    results: dict[str, Any] = {
        "model": model,
        "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(0),
        "code": code_version(),
        "what": "wall-clock time of Engine.step() during steady-state decode; breakdown from "
        "a second set of steps with a device synchronization after the pass",
        "configs": {},
    }
    for name in args.configs:
        label, overrides = CONFIGS[name]
        config = EngineConfig(prefill_chunk_tokens=None, **overrides)
        engine = Engine.from_pretrained(model, config)
        timers = _Timers(engine)
        rows = []
        for batch_size, context in POINTS:
            row = measure(engine, timers, batch_size, context, args.steps, args.warmup)
            rows.append(row)
            print(name, json.dumps(row), flush=True)
        results["configs"][name] = {"label": label, "overrides": overrides, "rows": rows}
        del engine, timers
        gc.collect()  # the timing wrappers and the graphs reference each other
        torch.cuda.empty_cache()
        out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
