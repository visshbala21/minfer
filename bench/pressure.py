"""KV memory pressure on real agent traces: concurrency swept past what the GPU can hold.

    .venv/bin/python -m bench.pressure
    .venv/bin/python -m bench.pressure --configs minfer/lru minfer/session --concurrency 32 64

Every agent session keeps tens of thousands of tokens of KV alive between its model calls
(the pauses are tool calls). With enough concurrent sessions their histories no longer fit
in GPU memory, and what an engine then does with a paused session's KV decides whether its
next call is a cache hit or a full prefill of its history.

For each configuration and each concurrency, a fresh server replays the real trace
(``bench/real_traces.py``: Open-SWE-Traces content, TraceLab tool-call pauses) closed-loop
for a fixed time. Configurations:

* minfer with least-recently-used eviction, with session-aware retention (the client
  names sessions and says which tools each response leads to), and with the host-memory
  tier on top;
* vLLM and SGLang with the flags of ``bench/servers.py``, and with the CPU-memory KV
  offloading each ships in the pinned version (vLLM's native KV offloading, SGLang's
  hierarchical cache).

``--kv-gb`` gives every engine the same KV-cache budget instead of the whole GPU, which
moves the capacity limit to a lower concurrency. Writes ``bench/results/pressure_7b.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Any

from bench import models
from bench import real_traces as rt
from bench.run_matrix import scrape
from bench.servers import code_version, launch

HOST_GB = 96
CONCURRENCY = (16, 32, 48, 64, 96)

# "<engine>/<name>": (label, server flags, send session hints)
CONFIGS: dict[str, tuple[str, list[str], bool]] = {
    "minfer/lru": ("minfer, LRU eviction", ["--eviction-policy", "lru"], False),
    "minfer/session": ("minfer, session-aware retention", [], True),
    "minfer/session_host": (
        f"minfer, session-aware retention + {HOST_GB} GB host tier",
        ["--host-kv-gb", str(HOST_GB)],
        True,
    ),
    "vllm/default": ("vLLM", [], False),
    "vllm/offload": (
        f"vLLM + native KV offloading ({HOST_GB} GB)",
        ["--kv-offloading-size", str(HOST_GB), "--kv-offloading-backend", "native"],
        False,
    ),
    "sglang/default": ("SGLang", [], False),
    "sglang/hicache": (
        f"SGLang + hierarchical cache ({HOST_GB} GB host)",
        ["--enable-hierarchical-cache", "--hicache-size", str(HOST_GB)],
        False,
    ),
}


def kv_budget_flags(engine: str, gigabytes: float, bytes_per_token: int) -> list[str]:
    """Flags that give ``engine`` a KV cache of ``gigabytes`` GiB."""
    size = int(gigabytes * 2**30)
    if engine == "minfer":
        return ["--kv-gb", str(gigabytes)]
    if engine == "vllm":
        return ["--kv-cache-memory-bytes", str(size)]
    return ["--max-total-tokens", str(size // bytes_per_token)]


def kv_bytes_per_token(model: str) -> int:
    """Bytes of bf16 KV one token occupies in ``model`` (all layers, keys and values)."""
    from minfer.model.config import ModelConfig
    from minfer.model.loader import resolve_model_dir

    config = ModelConfig.from_pretrained(resolve_model_dir(model))
    return config.num_layers * 2 * config.num_kv_heads * config.head_dim * 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=models.PRIMARY, help="model id, or 7b / 1.5b")
    parser.add_argument("--trace", type=Path, default=None, help="compiled real trace")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--concurrency", nargs="+", type=int, default=list(CONCURRENCY))
    parser.add_argument("--duration", type=float, default=240.0, help="seconds of load per run")
    parser.add_argument("--warmup", type=float, default=60.0, help="seconds excluded at the start")
    parser.add_argument(
        "--ramp", type=float, default=30.0, help="seconds to start the sessions over"
    )
    parser.add_argument("--kv-gb", type=float, default=None, help="KV budget for every engine")
    parser.add_argument("--port", type=int, default=8061)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="keep results already on disk")
    parser.add_argument(
        "--remeasure",
        action="store_true",
        help="with --resume: measure the selected configurations again, keep the others",
    )
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    from transformers import AutoTokenizer

    model = models.resolve(args.model)
    tag = models.tag(model)
    trace_path = args.trace or Path(f"bench/traces/real_{tag}.jsonl")
    out = args.out or models.results_path("bench/results/pressure.json", model)
    trace = rt.RealTrace.load(trace_path)
    tokenizer = AutoTokenizer.from_pretrained(model)
    per_token = kv_bytes_per_token(model)
    results: dict[str, Any] = {
        "model": model,
        "trace": {**trace.header, "sha256": trace.sha256, "sessions": len(trace.sessions)},
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "ramp_s": args.ramp,
        "kv_budget_gb": args.kv_gb,
        "kv_bytes_per_token": per_token,
        "configs": {},
    }
    if args.resume and out.exists():
        results = json.loads(out.read_text())
        if args.remeasure:
            for name in list(results["configs"]):
                if name in args.configs or name not in CONFIGS:
                    del results["configs"][name]
    for name in args.configs:
        engine, variant = name.split("/")
        label, flags, hints = CONFIGS[name]
        if args.kv_gb is not None:
            flags = [*flags, *kv_budget_flags(engine, args.kv_gb, per_token)]
        entry = results["configs"].setdefault(
            name, {"engine": engine, "label": label, "flags": flags, "session_hints": hints}
        )
        runs = entry.setdefault("runs", {})
        for concurrency in args.concurrency:
            if str(concurrency) in runs:
                continue
            log = Path("bench/logs") / f"pressure_{tag}_{engine}_{variant}_c{concurrency}.log"
            options = rt.ReplayOptions(
                concurrency, session_hints=hints, duration=args.duration, ramp=args.ramp
            )
            try:
                with launch(engine, model, args.port, log, flags) as (url, argv):
                    entry["command"] = argv
                    entry["code"] = code_version()
                    records, elapsed = asyncio.run(rt.replay(trace, tokenizer, url, model, options))
                    metrics = scrape(url)
            except RuntimeError as error:  # a configuration that does not start is a result
                runs[str(concurrency)] = {"error": str(error)}
                out.write_text(json.dumps(results, indent=1) + "\n")
                print(f"[{name}] c={concurrency} failed: {error}", flush=True)
                continue
            row = rt.summarize(records, elapsed, args.warmup)
            row["server_metrics"] = {
                key: value for key, value in metrics.items() if key.startswith("minfer_")
            }
            runs[str(concurrency)] = row
            out.write_text(json.dumps(results, indent=1) + "\n")
            hit = row["cache_hit_rate"]
            print(
                f"[{name}] c={concurrency}: ok {row['completed']}/{row['requests']} "
                f"{row['output_tokens_per_second']:.0f} tok/s ttft {row['ttft_ms']['p50']:.0f}/"
                f"{row['ttft_ms']['p99']:.0f} tpot {row['tpot_ms']['p50']:.2f} "
                f"hit {'n/a' if hit is None else f'{hit:.3f}'} "
                f"prefill {row['prefill_tokens_computed']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
