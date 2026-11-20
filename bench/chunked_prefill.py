"""Chunked prefill on a mixed long/short workload (Phase 3).

    .venv/bin/python -m bench.chunked_prefill

Short interactive requests (small prompt, long output) share the engine with long-prompt
requests (thousands of prompt tokens, short output). Without chunking a long prompt is
prefilled in one pass and every running stream waits for it; with chunking the prompt is
split into pieces that alternate with decode steps. The same seeded arrival trace is
replayed for every setting. Writes ``bench/results/chunked_prefill.json``.

Reported per setting: per-request TPOT and per-token inter-token latency of the short
requests (what chunking protects), TTFT of both classes (what chunking costs), and output
throughput.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from bench.driver import TraceRequest, percentiles, run_trace
from minfer.engine import Engine, EngineConfig
from minfer.sampling import SamplingParams

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def build_trace(seed: int, duration: float, short_rate: float, long_rate: float) -> list:
    """Poisson arrivals of short and long requests over ``duration`` seconds."""
    rng = random.Random(seed)
    trace = []

    def arrivals(rate: float) -> list[float]:
        times, now = [], rng.expovariate(rate)
        while now < duration:
            times.append(now)
            now += rng.expovariate(rate)
        return times

    def tokens(count: int) -> list[int]:
        return [rng.randrange(1000, 100_000) for _ in range(count)]

    for at in arrivals(short_rate):
        params = SamplingParams(max_tokens=rng.randint(96, 192), ignore_eos=True)
        trace.append(TraceRequest(at, tokens(rng.randint(32, 256)), params, "short"))
    for at in arrivals(long_rate):
        params = SamplingParams(max_tokens=rng.randint(16, 48), ignore_eos=True)
        trace.append(TraceRequest(at, tokens(rng.randint(4000, 16000)), params, "long"))
    return sorted(trace, key=lambda r: r.arrival)


def summarize(timings: list, duration: float) -> dict:
    """Latency summary by request class."""
    summary: dict = {"duration_s": duration}
    generated = sum(len(t.token_times) for t in timings)
    summary["output_tokens"] = generated
    summary["output_tokens_per_second"] = generated / duration
    for kind in ("short", "long"):
        group = [t for t in timings if t.kind == kind]
        summary[kind] = {
            "requests": len(group),
            "ttft_ms": percentiles([t.ttft for t in group]),
            "tpot_ms": percentiles([t.tpot for t in group if t.tpot is not None]),
            "inter_token_ms": percentiles([gap for t in group for gap in t.inter_token_gaps]),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("bench/results/chunked_prefill.json"))
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--short-rate", type=float, default=12.0)
    parser.add_argument("--long-rate", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunks", type=int, nargs="+", default=[4096, 2048, 1024, 512, 256])
    args = parser.parse_args()

    trace = build_trace(args.seed, args.duration, args.short_rate, args.long_rate)
    warmup = build_trace(args.seed + 1, 3.0, args.short_rate, args.long_rate)
    results: dict = {
        "model": MODEL,
        "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(0),
        "workload": {
            "seed": args.seed,
            "duration_s": args.duration,
            "short": f"Poisson {args.short_rate:.1f} req/s, prompt 32-256 tok, output 96-192 tok",
            "long": f"Poisson {args.long_rate:.1f} req/s, prompt 4000-16000 tok, output 16-48 tok",
            "requests": {k: sum(r.kind == k for r in trace) for k in ("short", "long")},
            "prompt_tokens": sum(len(r.prompt) for r in trace),
        },
        "settings": {},
    }
    for chunk in [None, *args.chunks]:
        label = "off" if chunk is None else f"chunk_{chunk}"
        config = EngineConfig(
            num_blocks=60_000, max_batched_tokens=65_536, prefill_chunk_tokens=chunk
        )
        engine = Engine.from_pretrained(MODEL, config)
        run_trace(engine, warmup)
        timings, duration = run_trace(engine, trace)
        summary = summarize(timings, duration)
        summary["prefill_chunk_tokens"] = chunk
        summary["preemptions"] = engine.scheduler.num_preemptions
        results["settings"][label] = summary
        short = summary["short"]
        print(
            f"{label:>10}: short TPOT p50 {short['tpot_ms']['p50']:.2f} p99 "
            f"{short['tpot_ms']['p99']:.2f} | ITL p99 {short['inter_token_ms']['p99']:.1f} max "
            f"{short['inter_token_ms']['max']:.1f} | TTFT short p99 {short['ttft_ms']['p99']:.0f} "
            f"long p50 {summary['long']['ttft_ms']['p50']:.0f} p99 "
            f"{summary['long']['ttft_ms']['p99']:.0f} ms | "
            f"{summary['output_tokens_per_second']:.0f} tok/s in {duration:.1f}s"
        )
        del engine
        torch.cuda.empty_cache()
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
