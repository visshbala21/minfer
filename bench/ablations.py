"""Ablations on the agent workload: the full system vs one feature changed at a time.

    .venv/bin/python -m bench.ablations

Each configuration starts a fresh ``minfer serve`` with the listed flags, warms it up, and
replays the same two agent traces the three-engine matrix uses for minfer (16 req/s and
"max"), so the ``full`` row here is the matrix's minfer row measured again.

Three groups:

* **default-on features, switched off one at a time** (prefix cache, cache-aware
  admission, CUDA graphs, prefill graphs, chunked prefill, fused kernels, async decode,
  joint prefill+decode passes), and the Triton decode kernel in place of FlashInfer's;
* **default-off features, switched on** (prompt-lookup and draft-model speculation, FP8 KV
  cache, FP8 weights);
* **constrained scenarios**, where the scheduling features have something to decide:
  16 running sequences at most (a queue forms at "max"), and a KV cache of 8% of GPU
  memory (the 64 sessions' contexts do not all fit, so cached prefixes get evicted) —
  each with and without the feature it is meant to exercise.

Writes ``bench/results/ablations.json`` (1.5B) or ``ablations_7b.json``.
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
from bench.servers import launch

RATES = {"16": 4, "max": 5}  # rate -> trace seed (the seeds the matrix uses for these rates)
SMALL_KV = ["--kv-memory-fraction", "0.08"]
FEW_SEQS = ["--max-num-seqs", "16"]

CONFIGS: dict[str, tuple[str, list[str]]] = {
    "full": ("full system (defaults)", []),
    "no_prefix_cache": ("− prefix cache", ["--no-prefix-cache"]),
    "no_cache_aware": ("− cache-aware admission (FCFS)", ["--no-cache-aware-scheduling"]),
    "no_cuda_graphs": ("− CUDA graphs (decode and prefill)", ["--no-cuda-graphs"]),
    "no_prefill_graphs": ("− prefill graphs only", ["--no-prefill-graphs"]),
    "no_chunked_prefill": ("− chunked prefill", ["--no-chunked-prefill"]),
    "no_fused_kernels": ("− fused kernels", ["--no-fused-kernels"]),
    "no_async_decode": ("− async decode", ["--no-async-decode"]),
    "no_joint_passes": ("− joint prefill+decode passes", ["--no-unified-batches"]),
    "triton_decode": (
        "Triton decode kernel instead of FlashInfer's",
        ["--decode-kernel", "triton"],
    ),
    "speculation_ngram": ("+ speculation: prompt lookup", ["--speculation", "ngram"]),
    "speculation_draft": ("+ speculation: draft model", ["--speculation", "draft"]),
    "fp8_kv": ("+ FP8 KV cache", ["--kv-cache-dtype", "fp8"]),
    "fp8_weights": ("+ FP8 weights", ["--weight-dtype", "fp8"]),
    "few_seqs": ("16 running sequences max", FEW_SEQS),
    "few_seqs_fcfs": ("16 running sequences max, FCFS", [*FEW_SEQS, "--no-cache-aware-scheduling"]),
    "small_kv": ("KV cache 8% of GPU memory", SMALL_KV),
    "small_kv_fcfs": ("KV cache 8%, FCFS", [*SMALL_KV, "--no-cache-aware-scheduling"]),
    "small_kv_fp8": ("KV cache 8%, FP8 KV", [*SMALL_KV, "--kv-cache-dtype", "fp8"]),
    "small_kv_no_cache": ("KV cache 8%, no prefix cache", [*SMALL_KV, "--no-prefix-cache"]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=models.PRIMARY, help="model id, or 7b / 1.5b")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--port", type=int, default=8041)
    parser.add_argument("--resume", action="store_true", help="keep results already on disk")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    from transformers import AutoTokenizer

    model = models.resolve(args.model)
    args.out = args.out or models.results_path("bench/results/ablations.json", model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    results: dict[str, Any] = {"model": model, "rates": list(RATES), "configs": {}}
    if args.resume and args.out.exists():
        results = json.loads(args.out.read_text())
    for name in args.configs:
        if name in results["configs"]:
            continue
        label, flags = CONFIGS[name]
        log = Path("bench/logs") / f"ablation_{models.tag(model)}_{name}_server.log"
        with launch("minfer", model, args.port, log, flags) as (base_url, argv):
            run_agent("minfer", base_url, "max", 10_000, tokenizer, model)  # warm-up
            runs = {
                rate: run_agent("minfer", base_url, rate, seed, tokenizer, model)
                for rate, seed in RATES.items()
            }
        results["configs"][name] = {"label": label, "flags": flags, "command": argv, **runs}
        args.out.write_text(json.dumps(results, indent=1) + "\n")
        for rate, row in runs.items():
            print(
                f"[{name}] rate={rate}: ok {row['completed']}/{row['requests']} "
                f"{row['output_tokens_per_second']:.0f} tok/s ttft {row['ttft_ms']['p50']:.1f}/"
                f"{row['ttft_ms']['p99']:.1f} tpot {row['tpot_ms']['p50']:.2f}/"
                f"{row['tpot_ms']['p99']:.2f} hit {row['server_prefix_cache_hit_rate']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
