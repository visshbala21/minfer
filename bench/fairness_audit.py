"""Fairness audit of the three-engine comparison: do the baselines' tuning flags change it?

    .venv/bin/python -m bench.fairness_audit
    .venv/bin/python -m bench.fairness_audit --configs vllm/default vllm/xxhash

The benchmark matrix runs vLLM and SGLang at their defaults. This script re-measures each
of them with the settings their documentation recommends for a latency-sensitive,
prefix-heavy workload, one at a time and then combined, on the agent traces the matrix
uses (same seeds, so the prompts are byte-identical), next to minfer at its defaults.

Every configuration gets a fresh server, the same warm-up as the matrix, and the same
client. Writes ``bench/results/fairness_audit.json`` (``_7b`` for the 7B model); the
combination that serves a baseline best is what ``bench/servers.py`` then uses for the
matrix.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

from bench import models
from bench.run_matrix import run_agent
from bench.servers import TUNED, launch

RATES = {"4": 2, "16": 4, "max": 5}  # rate -> trace seed (the matrix's seeds for these rates)

# "<engine>/<name>": (what the flags do, flags). ``default`` is the engine with none of
# the tuning flags of bench/servers.py; ``tuned`` is the set the matrix uses.
CONFIGS: dict[str, tuple[str, list[str]]] = {
    "minfer/default": ("minfer at its defaults", []),
    "vllm/default": ("vLLM at its defaults", []),
    "vllm/no_access_log": (
        "no per-request access-log line (minfer and SGLang write none)",
        ["--disable-uvicorn-access-log"],
    ),
    "vllm/xxhash": (
        "prefix-cache block hashes with xxHash instead of SHA-256",
        ["--prefix-caching-hash-algo", "xxhash"],
    ),
    "vllm/interactivity": (
        "performance mode 'interactivity' (fine-grained CUDA graphs, latency kernels)",
        ["--performance-mode", "interactivity"],
    ),
    "vllm/throughput": (
        "performance mode 'throughput' (doubles max-num-seqs and max-num-batched-tokens)",
        ["--performance-mode", "throughput"],
    ),
    "vllm/tuned": ("the flags the matrix uses for vLLM", TUNED["vllm"]),
    "sglang/default": ("SGLang at its defaults", []),
    "sglang/lpm": (
        "longest-prefix-match scheduling (the default here is fcfs)",
        ["--schedule-policy", "lpm"],
    ),
    "sglang/batch_tokenizer": (
        "dynamic batch tokenizer for concurrent requests",
        ["--enable-dynamic-batch-tokenizer"],
    ),
    "sglang/mixed_chunk": (
        "prefill and decode mixed in one batch",
        ["--enable-mixed-chunk"],
    ),
    "sglang/torch_compile": (
        "torch.compile for the model (documented for small models; experimental)",
        ["--enable-torch-compile"],
    ),
    "sglang/tuned": ("the flags the matrix uses for SGLang", TUNED["sglang"]),
    "minfer/token_ids": ("minfer at its defaults, prompts sent as token ids", []),
    "vllm/token_ids": ("vLLM at its defaults, prompts sent as token ids", []),
    "sglang/token_ids": ("SGLang at its defaults, prompts sent as token ids", []),
}
# Configurations whose client sends pre-tokenized prompts: the server's tokenizer and its
# plumbing are then out of the measured path, and what is left is the engine.
TOKEN_IDS = {name for name in CONFIGS if name.endswith("/token_ids")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=models.SECONDARY, help="model id, or 7b / 1.5b")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--resume", action="store_true", help="keep results already on disk")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    from transformers import AutoTokenizer

    model = models.resolve(args.model)
    args.out = args.out or models.results_path("bench/results/fairness_audit.json", model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    results: dict[str, Any] = {"model": model, "rates": list(RATES), "configs": {}}
    if args.resume and args.out.exists():
        results = json.loads(args.out.read_text())
    for name in args.configs:
        if name in results["configs"]:
            continue
        engine, variant = name.split("/")
        label, flags = CONFIGS[name]
        log = Path("bench/logs") / f"audit_{models.tag(model)}_{engine}_{variant}_server.log"
        ids = name in TOKEN_IDS
        entry: dict[str, Any] = {"engine": engine, "label": label, "flags": flags}
        entry["prompts_as_token_ids"] = ids
        try:
            with launch(engine, model, args.port, log, flags, tuned=False) as (url, argv):
                entry["command"] = argv
                run_agent(engine, url, "max", 10_000, tokenizer, model, ids)  # warm-up
                for rate, seed in RATES.items():
                    entry[rate] = run_agent(engine, url, rate, seed, tokenizer, model, ids)
        except RuntimeError as error:  # a configuration that does not start is a result
            entry["error"] = str(error)
        results["configs"][name] = entry
        args.out.write_text(json.dumps(results, indent=1) + "\n")
        for rate in RATES:
            if rate not in entry:
                print(f"[{name}] failed: {entry.get('error')}", flush=True)
                break
            row = entry[rate]
            print(
                f"[{name}] rate={rate}: ok {row['completed']}/{row['requests']} "
                f"{row['output_tokens_per_second']:.0f} tok/s ttft {row['ttft_ms']['p50']:.1f}/"
                f"{row['ttft_ms']['p99']:.1f} tpot {row['tpot_ms']['p50']:.2f}/"
                f"{row['tpot_ms']['p99']:.2f} hit {row['server_prefix_cache_hit_rate']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
