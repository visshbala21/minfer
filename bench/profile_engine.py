"""Profile the engine's decode and prefill passes with torch.profiler (Phase 11).

    .venv/bin/python -m bench.profile_engine --tag before

For each scenario — a CUDA-graph decode step at batch 1 and 64, an eager decode step, and
an eager prefill pass of 2,048 tokens — a few engine steps are traced. Written to
``bench/results/profiles/<tag>/``: a Chrome trace per scenario (gzip; open in
``chrome://tracing`` or Perfetto) and ``summary.json`` with, per step, the number of CUDA
kernels launched, total GPU kernel time, wall time, and the kernels that account for the
most launches and the most GPU time.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from minfer.engine import Engine, EngineConfig
from minfer.sampling import SamplingParams

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
STEPS = 5


def _fill(engine: Engine, batch: int, context: int, seed: int) -> None:
    """Admit ``batch`` sequences with ``context``-token prompts and prefill them."""
    rng = random.Random(seed)
    params = SamplingParams(max_tokens=4096, ignore_eos=True)
    for _ in range(batch):
        engine.add_request([rng.randrange(1000, 100_000) for _ in range(context)], params)
    while any(seq.num_pending > 1 for seq in engine.scheduler.running) or engine.scheduler.waiting:
        engine.step()
    for _ in range(10):
        engine.step()


def _drain(engine: Engine) -> None:
    for seq in list(engine.scheduler.running) + list(engine.scheduler.waiting):
        engine.abort(seq.seq_id)


def _trace(steps: int, run_step: Any, out: Path) -> dict[str, Any]:
    """Profile ``steps`` calls of ``run_step``; write the trace; return a summary."""
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        start = time.perf_counter()
        for _ in range(steps):
            run_step()
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
    raw = out.with_suffix("")
    prof.export_chrome_trace(str(raw))
    with raw.open("rb") as source, gzip.open(out, "wb") as target:
        shutil.copyfileobj(source, target)
    raw.unlink()

    launches: Counter[str] = Counter()
    gpu_time: defaultdict[str, float] = defaultdict(float)
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA and "Memcpy" not in event.name:
            name = event.name.split("(")[0][:80]
            launches[name] += 1
            gpu_time[name] += event.device_time_total or event.cuda_time_total
    total_launches = sum(launches.values())
    return {
        "steps": steps,
        "wall_ms_per_step": wall / steps * 1e3,
        "cuda_kernels_per_step": total_launches / steps,
        "gpu_kernel_ms_per_step": sum(gpu_time.values()) / steps / 1e3,
        "top_kernels_by_launches": [
            {"kernel": name, "per_step": count / steps} for name, count in launches.most_common(8)
        ],
        "top_kernels_by_gpu_time": [
            {"kernel": name, "ms_per_step": t / steps / 1e3}
            for name, t in sorted(gpu_time.items(), key=lambda kv: -kv[1])[:8]
        ],
        "trace": out.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="subdirectory name, e.g. before / after")
    parser.add_argument("--engine-args", type=json.loads, default={}, help="EngineConfig overrides")
    args = parser.parse_args()
    out_dir = Path("bench/results/profiles") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "model": MODEL,
        "gpu": torch.cuda.get_device_name(0),
        "engine_overrides": args.engine_args,
        "note": "profiler overhead inflates wall times; compare kernel counts and GPU time",
        "scenarios": {},
    }
    base = {"num_blocks": 40_000, "max_batched_tokens": 65_536, "prefill_chunk_tokens": None}
    scenarios = [
        ("decode_graph_batch1", {"cuda_graphs": True}, 1),
        ("decode_graph_batch64", {"cuda_graphs": True}, 64),
        ("decode_eager_batch1", {"cuda_graphs": False}, 1),
    ]
    engines: dict[bool, Engine] = {}
    for name, overrides, batch in scenarios:
        graphs = overrides["cuda_graphs"]
        if graphs not in engines:
            engines[graphs] = Engine.from_pretrained(
                MODEL, EngineConfig(**{**base, **overrides, **args.engine_args})
            )
        engine = engines[graphs]
        _fill(engine, batch, 512, seed=batch)
        summary["scenarios"][name] = _trace(STEPS, engine.step, out_dir / f"{name}.json.gz")
        _drain(engine)
        print(
            name,
            json.dumps({k: v for k, v in summary["scenarios"][name].items() if "top" not in k}),
        )

    # Prefill: one 2,048-token prompt per step (each step admits and prefills one request).
    engine = engines[True]
    rng = random.Random(0)
    params = SamplingParams(max_tokens=1, ignore_eos=True)

    def prefill_step() -> None:
        engine.add_request([rng.randrange(1000, 100_000) for _ in range(2048)], params)
        engine.step()

    prefill_step()
    summary["scenarios"]["prefill_2048_tokens"] = _trace(
        STEPS, prefill_step, out_dir / "prefill_2048_tokens.json.gz"
    )
    print(
        "prefill",
        json.dumps(
            {k: v for k, v in summary["scenarios"]["prefill_2048_tokens"].items() if "top" not in k}
        ),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
