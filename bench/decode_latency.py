"""Decode-step latency, one optimization at a time: CUDA graphs, Triton decode, fused kernels.

    .venv/bin/python -m bench.decode_latency

For each batch size, that many sequences with a fixed-length context are prefilled and then
decoded; the wall-clock time of each ``Engine.step()`` during steady-state decode is
recorded. A step includes everything the engine does per token — scheduling, building the
batch, planning attention, the forward pass, sampling and the host sync that reads the
sampled ids — so the numbers are end-to-end time per decode step, not kernel time.
Writes ``bench/results/decode_latency.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from minfer.engine import Engine, EngineConfig
from minfer.sampling import SamplingParams

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def measure(engine: Engine, batch_size: int, context: int, steps: int, warmup: int) -> dict:
    """Time ``steps`` steady-state decode steps at ``batch_size``."""
    rng = random.Random(batch_size * 100_003 + context)
    # Headroom: large batches take a few steps to prefill, so the first sequences are some
    # tokens ahead of the last; nobody may finish inside the timed window.
    params = SamplingParams(max_tokens=warmup + steps + 64, ignore_eos=True)
    for _ in range(batch_size):
        engine.add_request([rng.randrange(1000, 100_000) for _ in range(context)], params)
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
    while engine.has_work:
        engine.step()
    array = np.asarray(times) * 1e3
    return {
        "batch_size": batch_size,
        "context_tokens": context,
        "steps": steps,
        "step_ms_p50": float(np.percentile(array, 50)),
        "step_ms_p99": float(np.percentile(array, 99)),
        "step_ms_mean": float(array.mean()),
        "tokens_per_second": batch_size / float(np.median(array)) * 1e3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("bench/results/decode_latency.json"))
    parser.add_argument("--contexts", type=int, nargs="+", default=[256, 2048])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    results: dict = {
        "model": MODEL,
        "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(0),
        "what": "wall-clock time of Engine.step() during steady-state decode",
        "steps_per_point": args.steps,
        "runs": {},
    }
    # Each run adds one optimization to the previous one; the last is the default engine.
    runs = (
        (
            "cuda_graphs_off",
            {"cuda_graphs": False, "fused_kernels": False, "decode_kernel": "flashinfer"},
        ),
        ("cuda_graphs_on", {"fused_kernels": False, "decode_kernel": "flashinfer"}),
        ("cuda_graphs_on_triton_decode", {"fused_kernels": False, "decode_kernel": "triton"}),
        ("cuda_graphs_on_triton_decode_fused", {"fused_kernels": True, "decode_kernel": "triton"}),
    )
    for label, overrides in runs:
        # Chunked prefill off: only steady-state decode is timed, and the prefill of the
        # large batches should not outlast the first sequences' token budget.
        config = EngineConfig(
            num_blocks=40_000, max_batched_tokens=65_536, prefill_chunk_tokens=None, **overrides
        )
        engine = Engine.from_pretrained(MODEL, config)
        rows = []
        for context in args.contexts:
            for batch_size in BATCH_SIZES:
                row = measure(engine, batch_size, context, args.steps, args.warmup)
                rows.append(row)
                print(label, row)
        results["runs"][label] = rows
        del engine
        torch.cuda.empty_cache()

    def compare(before: str, after: str, before_key: str, after_key: str) -> list[dict]:
        return [
            {
                "batch_size": a["batch_size"],
                "context_tokens": a["context_tokens"],
                before_key: a["step_ms_p50"],
                after_key: b["step_ms_p50"],
                "speedup": a["step_ms_p50"] / b["step_ms_p50"],
            }
            for a, b in zip(results["runs"][before], results["runs"][after], strict=True)
        ]

    results["speedup_p50"] = compare("cuda_graphs_off", "cuda_graphs_on", "off_ms", "on_ms")
    results["triton_vs_flashinfer_p50"] = compare(
        "cuda_graphs_on", "cuda_graphs_on_triton_decode", "flashinfer_ms", "triton_ms"
    )
    results["fused_vs_unfused_p50"] = compare(
        "cuda_graphs_on_triton_decode",
        "cuda_graphs_on_triton_decode_fused",
        "unfused_ms",
        "fused_ms",
    )
    results["default_vs_phase3_p50"] = compare(
        "cuda_graphs_on", "cuda_graphs_on_triton_decode_fused", "phase3_ms", "default_ms"
    )
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
