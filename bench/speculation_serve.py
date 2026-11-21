"""Speculative decoding through the server on the agent workload: does it lower TPOT?

    .venv/bin/python -m bench.speculation_serve

The engine-level benchmark (``bench/speculation.py``) runs saturated; the plan's target is
per-token latency at low and medium request rates, where a decode step has few rows and a
verification pass has room to carry drafted tokens. For each configuration a fresh
``minfer serve`` replays the matrix's agent traces at 1, 4 and 8 req/s:

* speculation off;
* prompt lookup with the verification pass replayed from CUDA graphs, for several n-gram
  sizes and draft lengths;
* the same without verification graphs (the eager verification pass of Phase 7), to
  separate what the graphs buy from what the drafts buy.

Acceptance rates come from the server's counters. Writes
``bench/results/speculation_serve_7b.json``.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

from bench import models
from bench.run_matrix import run_agent, scrape
from bench.servers import code_version, launch

RATES = {"1": 0, "4": 2, "8": 3}  # rate -> trace seed (the matrix's seeds for these rates)


def ngram(size: int, draft: int, *extra: str) -> list[str]:
    flags = ["--speculation", "ngram", "--ngram-size", str(size)]
    return [*flags, "--num-draft-tokens", str(draft), *extra]


CONFIGS: dict[str, tuple[str, list[str]]] = {
    "off": ("no speculation", []),
    "ngram_n3_k4": ("prompt lookup, n=3, up to 4 drafted", ngram(3, 4)),
    "ngram_n2_k4": ("prompt lookup, n=2, up to 4 drafted", ngram(2, 4)),
    "ngram_n4_k4": ("prompt lookup, n=4, up to 4 drafted", ngram(4, 4)),
    "ngram_n3_k8": ("prompt lookup, n=3, up to 8 drafted", ngram(3, 8)),
    "ngram_n5_k8": ("prompt lookup, n=5, up to 8 drafted", ngram(5, 8)),
    "ngram_n3_k4_always": (
        "prompt lookup, n=3, up to 4 drafted, never paused",
        ngram(3, 4, "--draft-min-acceptance", "0"),
    ),
    "ngram_n3_k4_eager": (
        "prompt lookup, n=3, up to 4 drafted, eager verification (Phase 7)",
        ngram(3, 4, "--no-verify-graphs"),
    ),
}
_COUNTERS = {
    "proposed": "minfer_spec_proposed_tokens_total",
    "accepted": "minfer_spec_accepted_tokens_total",
    "rounds": "minfer_spec_rounds_total",
    "generated": "minfer_generation_tokens_total",
    "steps": "minfer_engine_steps_total",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=models.PRIMARY, help="model id, or 7b / 1.5b")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--port", type=int, default=8071)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="keep results already on disk")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    from transformers import AutoTokenizer

    model = models.resolve(args.model)
    out = args.out or models.results_path("bench/results/speculation_serve.json", model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    results: dict[str, Any] = {"model": model, "rates": list(RATES), "configs": {}}
    if args.resume and out.exists():
        results = json.loads(out.read_text())
    for name in args.configs:
        if name in results["configs"]:
            continue
        label, flags = CONFIGS[name]
        log = Path("bench/logs") / f"speculation_{models.tag(model)}_{name}_server.log"
        entry: dict[str, Any] = {"label": label, "flags": flags}
        with launch("minfer", model, args.port, log, flags) as (url, argv):
            entry["command"] = argv
            entry["code"] = code_version()
            run_agent("minfer", url, "max", 10_000, tokenizer, model)  # warm-up
            for rate, seed in RATES.items():
                before = scrape(url)
                row = run_agent("minfer", url, rate, seed, tokenizer, model)
                after = scrape(url)
                delta = {k: after.get(m, 0.0) - before.get(m, 0.0) for k, m in _COUNTERS.items()}
                row["drafted_tokens"] = delta["proposed"]
                row["accepted_tokens"] = delta["accepted"]
                row["acceptance_rate"] = (
                    delta["accepted"] / delta["proposed"] if delta["proposed"] else None
                )
                row["drafted_share_of_output"] = (
                    delta["accepted"] / delta["generated"] if delta["generated"] else None
                )
                row["tokens_per_step"] = (
                    delta["generated"] / delta["steps"] if delta["steps"] else None
                )
                entry[rate] = row
                acceptance = row["acceptance_rate"]
                print(
                    f"[{name}] rate={rate}: tpot {row['tpot_ms']['p50']:.2f}/"
                    f"{row['tpot_ms']['p99']:.2f} ttft {row['ttft_ms']['p50']:.1f} "
                    f"e2e {row['e2e_ms']['p50']:.0f} accepted "
                    f"{'n/a' if acceptance is None else f'{acceptance:.0%}'} "
                    f"of {row['drafted_tokens']:.0f}",
                    flush=True,
                )
        results["configs"][name] = entry
        out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
