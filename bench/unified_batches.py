"""A/B experiments on the engine's decode path, on the agent workload.

    .venv/bin/python -m bench.unified_batches                           # unified passes
    .venv/bin/python -m bench.unified_batches --experiment decode_kernel

* ``unified_batches``: a step that both prefills and decodes used to run two passes over
  the weights; with ``--unified-batches`` it runs one ragged pass in which decoding
  sequences are one-token spans.
* ``decode_kernel``: the Triton paged-decode kernel against FlashInfer's decode kernel
  (tensor-core path).

A fresh ``minfer serve`` per setting replays the matrix's agent traces at 4 and 16 req/s
and "max". Writes ``bench/results/<experiment>_7b.json``.
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
from bench.servers import code_version, launch

RATES = {"4": 2, "16": 4, "max": 5}  # rate -> trace seed (the matrix's seeds for these rates)
EXPERIMENTS = {
    "unified_batches": {
        "split": ("separate prefill and decode passes (before)", ["--no-unified-batches"]),
        "unified": ("one ragged pass per step (after)", ["--unified-batches"]),
    },
    "decode_kernel": {
        "triton": ("Triton paged-decode kernel", ["--decode-kernel", "triton"]),
        "flashinfer": ("FlashInfer decode kernel", ["--decode-kernel", "flashinfer"]),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=models.PRIMARY, help="model id, or 7b / 1.5b")
    parser.add_argument("--experiment", default="unified_batches", choices=list(EXPERIMENTS))
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    from transformers import AutoTokenizer

    model = models.resolve(args.model)
    out = args.out or models.results_path(f"bench/results/{args.experiment}.json", model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    results: dict[str, Any] = {"model": model, "rates": list(RATES), "configs": {}}
    for name, (label, flags) in EXPERIMENTS[args.experiment].items():
        log = Path("bench/logs") / f"{args.experiment}_{models.tag(model)}_{name}_server.log"
        entry: dict[str, Any] = {"label": label, "flags": flags}
        with launch("minfer", model, args.port, log, flags) as (url, argv):
            entry["command"] = argv
            entry["code"] = code_version()
            run_agent("minfer", url, "max", 10_000, tokenizer, model)  # warm-up
            for rate, seed in RATES.items():
                row = entry[rate] = run_agent("minfer", url, rate, seed, tokenizer, model)
                print(
                    f"[{name}] rate={rate}: {row['output_tokens_per_second']:.0f} tok/s "
                    f"ttft {row['ttft_ms']['p50']:.1f}/{row['ttft_ms']['p99']:.1f} "
                    f"tpot {row['tpot_ms']['p50']:.2f}/{row['tpot_ms']['p99']:.2f} "
                    f"e2e {row['e2e_ms']['p50']:.0f}",
                    flush=True,
                )
        results["configs"][name] = entry
        out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
