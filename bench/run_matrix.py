"""Benchmark matrix: {minfer, vLLM, SGLang} × {agent workload, ShareGPT} × request rates.

    .venv/bin/python -m bench.run_matrix                 # everything, on the 7B model
    .venv/bin/python -m bench.run_matrix --model 1.5b    # the secondary model
    .venv/bin/python -m bench.run_matrix --engines minfer --workloads agent --rates 4 max

One server runs at a time (the GPU is never shared). For each engine the server is started
once, warmed up, and then driven through every rate of both workloads:

* **agent** — traces from ``bench/agent_workload.py``, replayed by its client. A fresh seed
  per rate keeps one rate's sessions from being cache hits for the next; the agent types'
  system prompts are shared by construction (that is the workload).
* **sharegpt** — single-turn chat requests from ShareGPT, driven by ``vllm bench serve``:
  a control workload with little prefix sharing, measured by a third-party client.

Results are written per engine to ``bench/results/matrix/<engine>.json`` after every run,
so an interrupted matrix can be resumed (``--resume``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from bench import agent_workload as aw
from bench import models
from bench.servers import ENGINES, ENVIRONMENT, ROOT, code_version, launch

RATES = ("1", "2", "4", "8", "16", "max")
RESULTS = Path("bench/results/matrix")  # for the 1.5B model; see bench/models.py
TRACES = Path("bench/traces")
SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
MAX_RATE_SESSIONS = 64

# (hit tokens, looked-up tokens) counters of each server's prefix cache.
_CACHE_COUNTERS = {
    "minfer": ("minfer_prefix_cache_hit_tokens_total", "minfer_prefix_cache_lookup_tokens_total"),
    "vllm": ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"),
    "sglang": ("sglang:cached_tokens_total", "sglang:prompt_tokens_total"),
}


def agent_sessions(rate: str) -> int:
    """Sessions per agent run: at least 60 s of arrivals and at least ~240 requests."""
    if rate == "max":
        return MAX_RATE_SESSIONS
    window = max(60.0, 240.0 / float(rate))
    return round(float(rate) / aw.MEAN_TURNS * window)


def sharegpt_prompts(rate: str) -> int:
    """Requests per ShareGPT run: about a minute of load, at least 120 requests."""
    return 1000 if rate == "max" else max(120, int(float(rate) * 60))


def scrape(base_url: str) -> dict[str, float]:
    """Sum every sample of each metric on the server's ``/metrics`` page."""
    totals: dict[str, float] = {}
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as response:
            text = response.read().decode()
    except OSError:
        return totals
    for line in text.splitlines():
        match = re.match(r"^([a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf)$", line)
        if match:
            try:
                totals[match[1]] = totals.get(match[1], 0.0) + float(match[2])
            except ValueError:
                continue
    return totals


def server_hit_rate(engine: str, before: dict[str, float], after: dict[str, float]) -> Any:
    """Prefix-cache hit rate over a run, from the server's own counters (if it has them)."""
    hits, lookups = _CACHE_COUNTERS[engine]
    if hits not in after or lookups not in after:
        return None
    looked_up = after[lookups] - before.get(lookups, 0.0)
    return (after[hits] - before.get(hits, 0.0)) / looked_up if looked_up else None


def run_agent(
    engine: str,
    base_url: str,
    rate: str,
    seed: int,
    tokenizer: Any,
    model: str,
    token_ids: bool = False,
) -> dict[str, Any]:
    """Generate (or reuse) the trace for ``rate``, replay it, and summarize.

    ``token_ids`` sends every prompt already tokenized instead of as text.
    """

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    numeric = None if rate == "max" else float(rate)
    trace = aw.generate(agent_sessions(rate), numeric, seed, count)
    trace.save(TRACES / f"agent_rate_{rate}_seed_{seed}.jsonl")
    prompts = aw.render_prompts(trace, tokenizer)
    if token_ids:
        prompts = aw.tokenize_prompts(prompts, tokenizer)
    before = scrape(base_url)
    records, duration = asyncio.run(aw.replay(trace, prompts, base_url, model))
    after = scrape(base_url)
    return {
        "rate": rate,
        "seed": seed,
        "sessions": len(trace.sessions),
        "trace_sha256": trace.sha256,
        "server_prefix_cache_hit_rate": server_hit_rate(engine, before, after),
        **aw.summarize(records, duration),
    }


def run_sharegpt(
    base_url: str, rate: str, dataset: str, tag: str, model: str, results: Path
) -> dict[str, Any]:
    """One ``vllm bench serve`` run on ShareGPT; returns its result file's summary fields."""
    out_dir = results / "sharegpt_raw"
    name = f"{tag}_rate_{rate}.json"
    argv = [".venv-vllm/bin/vllm", "bench", "serve", "--backend", "openai-chat"]
    argv += ["--endpoint", "/v1/chat/completions", "--base-url", base_url, "--model", model]
    argv += ["--dataset-name", "sharegpt", "--dataset-path", dataset]
    argv += ["--num-prompts", str(sharegpt_prompts(rate)), "--seed", "0", "--temperature", "0"]
    argv += ["--ignore-eos", "--save-result", "--result-dir", str(out_dir)]
    argv += ["--result-filename", name, "--percentile-metrics", "ttft,tpot,itl,e2el"]
    argv += ["--request-rate", "inf" if rate == "max" else rate]
    log = Path("bench/logs") / f"sharegpt_{models.tag(model)}_{name}.log"
    with log.open("w") as handle:
        subprocess.run(argv, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    raw = json.loads((ROOT / out_dir / name).read_text())
    keys = ["completed", "failed", "duration", "total_input_tokens", "total_output_tokens"]
    keys += ["request_throughput", "output_throughput"]
    keys += [f"{stat}_{metric}_ms" for metric in ("ttft", "tpot", "itl", "e2el")
             for stat in ("mean", "median", "p99")]  # fmt: skip
    return {"rate": rate, "num_prompts": raw.get("num_prompts"), **{k: raw.get(k) for k in keys}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=models.PRIMARY, help="model id, or 7b / 1.5b")
    parser.add_argument("--engines", nargs="+", default=list(ENGINES), choices=ENGINES)
    parser.add_argument("--workloads", nargs="+", default=["agent", "sharegpt"])
    parser.add_argument("--rates", nargs="+", default=list(RATES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8031)
    parser.add_argument("--tag", default=None, help="results file stem (default: engine name)")
    parser.add_argument("--server-args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--resume", action="store_true", help="keep results already on disk")
    args = parser.parse_args()
    # Turn SIGTERM into an exception so a killed run still shuts its server down.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    model = models.resolve(args.model)
    out_dir = models.results_path(RESULTS, model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    dataset = hf_hub_download(SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    for engine in args.engines:
        tag = args.tag or engine
        path = out_dir / f"{tag}.json"
        results: dict[str, Any] = {}
        if args.resume and path.exists():
            results = json.loads(path.read_text())
        log = Path("bench/logs") / f"matrix_{models.tag(model)}_{tag}_server.log"
        started = time.time()
        with launch(engine, model, args.port, log, args.server_args) as (base_url, argv):
            results.update(
                engine=engine,
                model=model,
                server_command=argv,
                server_environment=ENVIRONMENT.get(engine, {}),
                server_startup_seconds=time.time() - started,
                code=code_version(),
            )
            # Warm-up: JIT, CUDA graphs, allocator pools, and the shared system prompts.
            run_agent(engine, base_url, "max", args.seed + 10_000, tokenizer, model)
            for workload in args.workloads:
                done = results.setdefault(workload, {})
                for index, rate in enumerate(args.rates):
                    if rate in done:
                        continue
                    if workload == "agent":
                        done[rate] = run_agent(
                            engine, base_url, rate, args.seed + index, tokenizer, model
                        )
                    else:
                        done[rate] = run_sharegpt(base_url, rate, dataset, tag, model, out_dir)
                    path.write_text(json.dumps(results, indent=1) + "\n")
                    row = done[rate]
                    print(
                        f"[{tag}] {workload} rate={rate}: " + json.dumps(_headline(row)), flush=True
                    )


def _headline(row: dict[str, Any]) -> dict[str, Any]:
    """The few numbers worth printing while the matrix runs."""
    if "ttft_ms" in row:
        return {
            "ok": f"{row['completed']}/{row['requests']}",
            "out_tok_s": round(row["output_tokens_per_second"]),
            "ttft_p50": round(row["ttft_ms"]["p50"], 1),
            "ttft_p99": round(row["ttft_ms"]["p99"], 1),
            "tpot_p50": round(row["tpot_ms"]["p50"], 2),
            "tpot_p99": round(row["tpot_ms"]["p99"], 2),
            "hit": row["server_prefix_cache_hit_rate"],
        }
    return {
        "ok": f"{row['completed']}/{row['num_prompts']}",
        "out_tok_s": round(row["output_throughput"]),
        "ttft_p50": round(row["median_ttft_ms"], 1),
        "ttft_p99": round(row["p99_ttft_ms"], 1),
        "tpot_p50": round(row["median_tpot_ms"], 2),
        "tpot_p99": round(row["p99_tpot_ms"], 2),
    }


if __name__ == "__main__":
    main()
