"""Composition study: agent-serving techniques alone and together, on real traces.

    .venv/bin/python -m bench.composition
    .venv/bin/python -m bench.composition --concurrency 48 --configs base prefix prefix+session

minfer as a testbed: every technique is a flag, so the same engine, model, trace and client
can be run with each valid combination of

* ``prefix``   — the exact-prefix (radix-tree) KV cache;
* ``session``  — session-aware retention instead of LRU eviction (needs ``prefix``);
* ``host``     — the host-memory KV tier (needs ``session``);
* ``reuse``    — position-independent KV reuse (needs ``prefix``; approximate);
* ``spec``     — prompt-lookup speculative decoding with graph-captured verification.

The workload is the real coding-agent trace of ``bench/real_traces.py`` replayed closed-loop
at a concurrency that puts the KV cache under memory pressure (``bench/pressure.py`` shows
where that is). Reported per configuration: prefill tokens computed, TTFT, TPOT,
throughput, and for the approximate technique the output-quality numbers come from
``bench/pic_quality.py``. vLLM and SGLang, each with the best configuration the pressure
benchmark found for it, are measured by ``bench/pressure.py`` on the same trace and settings.

Writes ``bench/results/composition_7b.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import shlex
import signal
import sys
from pathlib import Path
from typing import Any

from bench import models
from bench import real_traces as rt
from bench.pressure import HOST_GB, kv_budget_flags, kv_bytes_per_token
from bench.run_matrix import scrape
from bench.servers import code_version, launch

TECHNIQUES = ("prefix", "session", "host", "reuse", "spec")
REQUIRES = {"session": "prefix", "host": "session", "reuse": "prefix"}


def valid(combination: frozenset[str]) -> bool:
    """Whether every technique in the combination has what it depends on."""
    return all(REQUIRES.get(name, name) in combination for name in combination)


def combinations() -> list[frozenset[str]]:
    """Every valid combination, smallest first."""
    found = []
    for size in range(len(TECHNIQUES) + 1):
        for names in itertools.combinations(TECHNIQUES, size):
            if valid(frozenset(names)):
                found.append(frozenset(names))
    return found


def name_of(combination: frozenset[str]) -> str:
    """``prefix+session+host``-style name, in the canonical technique order."""
    return "+".join(t for t in TECHNIQUES if t in combination) or "base"


def flags_of(combination: frozenset[str]) -> list[str]:
    """``minfer serve`` flags for a combination (the defaults are: prefix + session)."""
    flags = []
    if "prefix" not in combination:
        flags += ["--no-prefix-cache", "--eviction-policy", "lru"]
    elif "session" not in combination:
        flags += ["--eviction-policy", "lru"]
    if "host" in combination:
        flags += ["--host-kv-gb", str(HOST_GB)]
    if "reuse" in combination:
        flags += ["--position-independent-reuse"]
    if "spec" in combination:
        flags += ["--speculation", "ngram"]
    return flags


def main() -> None:
    every = {name_of(c): c for c in combinations()}
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=models.PRIMARY, help="model id, or 7b / 1.5b")
    parser.add_argument("--trace", type=Path, default=None, help="compiled real trace")
    parser.add_argument("--configs", nargs="+", default=list(every), choices=list(every))
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--duration", type=float, default=240.0)
    parser.add_argument("--warmup", type=float, default=60.0)
    parser.add_argument("--ramp", type=float, default=30.0)
    parser.add_argument("--kv-gb", type=float, default=None, help="KV budget (as in pressure.py)")
    parser.add_argument("--spec-flags", default="", help="extra server flags with 'spec'")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="keep results already on disk")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    from transformers import AutoTokenizer

    model = models.resolve(args.model)
    tag = models.tag(model)
    out = args.out or models.results_path("bench/results/composition.json", model)
    trace = rt.RealTrace.load(args.trace or Path(f"bench/traces/real_{tag}.jsonl"))
    tokenizer = AutoTokenizer.from_pretrained(model)
    results: dict[str, Any] = {
        "model": model,
        "trace": {**trace.header, "sha256": trace.sha256, "sessions": len(trace.sessions)},
        "concurrency": args.concurrency,
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "ramp_s": args.ramp,
        "kv_budget_gb": args.kv_gb,
        "techniques": list(TECHNIQUES),
        "configs": {},
    }
    if args.resume and out.exists():
        results = json.loads(out.read_text())
    for name in args.configs:
        if name in results["configs"]:
            continue
        combination = every[name]
        flags = flags_of(combination)
        if "spec" in combination:
            flags += shlex.split(args.spec_flags)
        if args.kv_gb is not None:
            flags += kv_budget_flags("minfer", args.kv_gb, kv_bytes_per_token(model))
        options = rt.ReplayOptions(
            args.concurrency,
            session_hints="session" in combination,
            duration=args.duration,
            ramp=args.ramp,
        )
        log = Path("bench/logs") / f"composition_{tag}_{name}.log"
        with launch("minfer", model, args.port, log, flags) as (url, argv):
            code = code_version()
            records, elapsed = asyncio.run(rt.replay(trace, tokenizer, url, model, options))
            metrics = scrape(url)
        row = rt.summarize(records, elapsed, args.warmup)
        row.update(
            techniques=sorted(combination),
            flags=flags,
            command=argv,
            code=code,
            server_metrics={k: v for k, v in metrics.items() if k.startswith("minfer_")},
        )
        results["configs"][name] = row
        out.write_text(json.dumps(results, indent=1) + "\n")
        hit = row["cache_hit_rate"]
        print(
            f"[{name}] ok {row['completed']}/{row['requests']} "
            f"{row['output_tokens_per_second']:.0f} tok/s ttft {row['ttft_ms']['p50']:.0f}/"
            f"{row['ttft_ms']['p99']:.0f} tpot {row['tpot_ms']['p50']:.2f} "
            f"hit {'n/a' if hit is None else f'{hit:.3f}'} "
            f"prefill {row['prefill_tokens_computed']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
