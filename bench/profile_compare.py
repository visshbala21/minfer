"""Kernel-level profile of steady-state decode at 7B, minfer next to vLLM (Phase 18).

    .venv/bin/python -m bench.profile_compare --engine minfer --batch 64 --context 4096
    VLLM_ENABLE_V1_MULTIPROCESSING=0 .venv-vllm/bin/python bench/profile_compare.py --engine vllm

``batch`` sequences of ``context`` random tokens are prefilled, then a generation of
``TOKENS`` more tokens each is traced with torch.profiler (CPU + CUDA activity; kernels
replayed from CUDA graphs are included). From the trace: GPU kernel time per decode step,
grouped into categories (GEMMs, attention, normalization / activation / RoPE, sampling,
other), the wall time per step, and how much of it the GPU sat idle. vLLM must run its
engine core in-process for its kernels to be in the trace.

Writes ``bench/results/profiles/compare_7b/<engine>_b<batch>_c<context>.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

MODEL = "Qwen/Qwen2.5-7B-Instruct"
TOKENS = 64
OUT = Path("bench/results/profiles/compare_7b")

CATEGORIES = (
    ("gemm", re.compile(r"gemm|gemv|sm90_xmma|cutlass|nvjet|splitK|matmul", re.I)),
    ("attention", re.compile(r"attention|flash|decode|prefill|paged|fmha|BatchDecode", re.I)),
    ("sampling", re.compile(r"argmax|topk|sort|softmax|sample|reduce", re.I)),
    ("norm_act_rope", re.compile(r"norm|silu|act_and_mul|rope|rotary|fused|triton", re.I)),
    ("copy", re.compile(r"copy|memcpy|memset|index|gather|scatter|cat", re.I)),
)


def category(name: str) -> str:
    for label, pattern in CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


def busy_time(intervals: list[tuple[float, float]]) -> float:
    """Total length of the union of ``intervals`` (microseconds)."""
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        if start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", choices=("minfer", "vllm"), required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--name", default=None, help="file name stem (default: the engine)")
    parser.add_argument("--set", nargs="*", default=[], help="minfer EngineConfig key=json")
    args = parser.parse_args()
    rng = random.Random(1)

    def prompts() -> list[list[int]]:
        return [
            [rng.randrange(1000, 100_000) for _ in range(args.context)] for _ in range(args.batch)
        ]

    if args.engine == "minfer":
        from minfer.engine import Engine, EngineConfig
        from minfer.sampling import SamplingParams

        overrides = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
        engine = Engine.from_pretrained(MODEL, EngineConfig(**overrides))
        params = SamplingParams(max_tokens=TOKENS, ignore_eos=True)

        def generate(batch: list[list[int]], tokens: int) -> None:
            engine.generate(batch, SamplingParams(max_tokens=tokens, ignore_eos=True))

        # Prefill outside the trace: admit, then step until every sequence decodes.
        def traced() -> int:
            for prompt in prompts():
                engine.add_request(prompt, params)
            while engine.scheduler.waiting or any(
                s.num_pending > 1 for s in engine.scheduler.running
            ):
                engine.step()
            torch.cuda.synchronize()
            return 0

        def run_traced() -> int:
            steps = 0
            while engine.has_work:
                engine.step()
                steps += 1
            return steps
    else:
        from vllm import LLM, SamplingParams, TokensPrompt

        llm = LLM(MODEL, dtype="bfloat16", max_model_len=32768)

        def generate(batch: list[list[int]], tokens: int) -> None:
            params = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
            llm.generate([TokensPrompt(prompt_token_ids=p) for p in batch], params, use_tqdm=False)

        engine_core = llm.llm_engine
        params = SamplingParams(temperature=0.0, max_tokens=TOKENS, ignore_eos=True)

        def traced() -> int:
            for index, prompt in enumerate(prompts()):
                engine_core.add_request(
                    f"p{time.time()}-{index}", TokensPrompt(prompt_token_ids=prompt), params
                )
            # Step until every request has its first token (prefill done).
            produced: dict[str, int] = {}
            while len(produced) < args.batch or min(produced.values()) < 1:
                for output in engine_core.step():
                    produced[output.request_id] = len(output.outputs[0].token_ids)
            torch.cuda.synchronize()
            return 0

        def run_traced() -> int:
            steps = 0
            while engine_core.has_unfinished_requests():
                engine_core.step()
                steps += 1
            return steps

    generate(prompts()[:8], 32)  # warm-up
    traced()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        start = time.perf_counter()
        steps = run_traced()
        torch.cuda.synchronize()
        wall = time.perf_counter() - start

    kernels = [
        e
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA and e.time_range.elapsed_us() > 0
    ]
    per_name: dict[str, float] = defaultdict(float)
    per_category: dict[str, float] = defaultdict(float)
    intervals = []
    for event in kernels:
        span = event.time_range
        intervals.append((span.start, span.end))
        name = event.name[:100]
        per_name[name] += span.elapsed_us()
        per_category[category(name)] += span.elapsed_us()
    # Tokens per sequence after the first: TOKENS - 1 decode steps (prefill emitted one).
    decode_steps = TOKENS - 1
    result: dict[str, Any] = {
        "engine": args.engine,
        "model": MODEL,
        "batch_size": args.batch,
        "context_tokens": args.context,
        "decode_steps": decode_steps,
        "engine_steps_traced": steps,
        "wall_ms_per_step": wall / decode_steps * 1e3,
        "gpu_busy_ms_per_step": busy_time(intervals) / decode_steps / 1e3,
        "kernel_ms_per_step": sum(per_name.values()) / decode_steps / 1e3,
        "kernels_per_step": len(kernels) / decode_steps,
        "by_category_ms_per_step": {
            k: v / decode_steps / 1e3
            for k, v in sorted(per_category.items(), key=lambda kv: -kv[1])
        },
        "top_kernels_ms_per_step": [
            {"kernel": k, "ms": v / decode_steps / 1e3}
            for k, v in sorted(per_name.items(), key=lambda kv: -kv[1])[:20]
        ],
        "note": "profiler overhead inflates wall time; GPU-side numbers are reliable",
    }
    result["gpu_idle_ms_per_step"] = result["wall_ms_per_step"] - result["gpu_busy_ms_per_step"]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.name or args.engine}_b{args.batch}_c{args.context}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "top_kernels_ms_per_step"}, indent=1))
    for row in result["top_kernels_ms_per_step"][:12]:
        print(f"  {row['ms']:.3f}  {row['kernel']}")


if __name__ == "__main__":
    main()
