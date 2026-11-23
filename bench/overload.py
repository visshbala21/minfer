"""Where a request's latency goes when minfer is past capacity (Phase 19).

    .venv/bin/python -m bench.overload --variants base --concurrency 96 128
    .venv/bin/python -m bench.overload --variants base adaptive --resume

The pressure sweep's full-stack configuration (``minfer/session_host`` of
``bench/pressure.py``: session-aware retention + host tier), replayed the same way on the
real trace, with two diagnostics switched on:

* the server's request log (``--request-log``): for each request, when it arrived, was
  first admitted, got its first token and finished, how long it waited for host-tier
  copies, how many tokens it prefilled and how often it was preempted;
* ``/metrics`` sampled every few seconds: queue depth, requests restoring, KV blocks used
  and evictable, host-tier staging waits.

From the log, each request measured after the warm-up splits into **queue** (waiting for
admission, host copies excluded), **restore** (waiting for its history to come back from
host memory), **prefill** (admitted to first token: its prefill chunks, interleaved with
everybody else's work, and any recomputation after a preemption) and **decode** (first
token to last). The server-side TTFT is queue + restore + prefill.

Variants are extra server flags on top of the full stack (``VARIANTS``), so a change can
be A/B'd on the same trace. Writes ``bench/results/overload_7b.json`` and keeps the raw
request logs and metric samples in ``bench/results/overload/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from bench import models
from bench import real_traces as rt
from bench.pressure import CONFIGS
from bench.run_matrix import scrape
from bench.servers import code_version, launch

BASE = "minfer/session_host"
# variant -> extra server flags on top of the full stack
VARIANTS: dict[str, list[str]] = {
    "base": [],
}
SAMPLED = (
    "minfer_requests_waiting",
    "minfer_requests_running",
    "minfer_requests_restoring",
    "minfer_kv_blocks_used",
    "minfer_kv_blocks_evictable",
    "minfer_kv_blocks",
    "minfer_preemptions_total",
    "minfer_host_staging_waits_total",
    "minfer_host_staging_wait_seconds_total",
    "minfer_engine_step_seconds_total",
    "minfer_prefill_tokens_total",
    "minfer_decode_tokens_total",
    "minfer_admission_stops_memory_total",
    "minfer_admission_stops_does_not_fit_total",
    "minfer_admission_stops_slots_total",
    "minfer_admission_shortfall_blocks_total",
    "minfer_admission_promised_blocks",
    "minfer_kv_blocks_free",
)


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {}
    return {
        "mean": float(array.mean()),
        **{f"p{p}": float(np.percentile(array, p)) for p in (50, 90, 99)},
    }


def breakdown(records: list[dict[str, Any]], warmup: float) -> dict[str, Any]:
    """Split the server-side latency of the requests that arrived after ``warmup`` seconds."""
    if not records:
        return {}
    start = min(r["arrival"] for r in records)
    rows = [r for r in records if r["arrival"] - start >= warmup and r["admitted_s"] is not None]
    parts: dict[str, list[float]] = {
        k: [] for k in ("queue", "restore", "prefill", "decode", "ttft")
    }
    for r in rows:
        parts["queue"].append(r["admitted_s"] - r["restore_s"])
        parts["restore"].append(r["restore_s"])
        parts["prefill"].append(r["first_token_s"] - r["admitted_s"])
        parts["decode"].append(r["finished_s"] - r["first_token_s"])
        parts["ttft"].append(r["first_token_s"])
    summary: dict[str, Any] = {"requests": len(rows)}
    summary.update({f"{k}_s": _stats(v) for k, v in parts.items()})
    ttft = np.asarray(parts["ttft"])
    # What the slowest tenth of first tokens spent their time on.
    if ttft.size:
        slow = ttft >= np.percentile(ttft, 90)
        summary["slowest_decile_mean_s"] = {
            k: float(np.asarray(parts[k])[slow].mean()) for k in ("queue", "restore", "prefill")
        }
    summary["prefill_computed_per_request"] = _stats([r["prefill_computed"] for r in rows])
    summary["preempted_requests"] = sum(r["preemptions"] > 0 for r in rows)
    summary["restored_requests"] = sum(r["restore_s"] > 0 for r in rows)
    return summary


class MetricsPoller(threading.Thread):
    """Samples the server's ``/metrics`` every ``interval`` seconds until stopped."""

    def __init__(self, url: str, interval: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.url, self.interval = url, interval
        self.samples: list[dict[str, float]] = []
        self.stop = threading.Event()

    def run(self) -> None:
        began = time.monotonic()
        while not self.stop.wait(self.interval):
            metrics = scrape(self.url)
            row = {"t": time.monotonic() - began}
            row.update({k: metrics[k] for k in SAMPLED if k in metrics})
            self.samples.append(row)


def summarize_samples(samples: list[dict[str, float]], warmup: float) -> dict[str, Any]:
    """Mean and peak of each gauge after the warm-up; rates of the counters over it."""
    rows = [s for s in samples if s["t"] >= warmup]
    if len(rows) < 2:
        return {}
    out: dict[str, Any] = {}
    for key in ("waiting", "running", "restoring"):
        values = [s.get(f"minfer_requests_{key}", 0.0) for s in rows]
        out[f"{key}_mean"], out[f"{key}_max"] = float(np.mean(values)), float(np.max(values))
    blocks = rows[-1].get("minfer_kv_blocks", 0.0) or 1.0
    used = [s.get("minfer_kv_blocks_used", 0.0) for s in rows]
    evictable = [s.get("minfer_kv_blocks_evictable", 0.0) for s in rows]
    out["kv_used_fraction_mean"] = float(np.mean(used)) / blocks
    out["kv_held_by_requests_fraction_mean"] = float(np.mean(np.subtract(used, evictable))) / blocks
    for key in ("admission_promised_blocks", "kv_blocks_free", "kv_blocks_evictable"):
        values = [s.get(f"minfer_{key}", 0.0) for s in rows]
        out[f"{key}_mean"], out[f"{key}_max"] = float(np.mean(values)), float(np.max(values))
    span = rows[-1]["t"] - rows[0]["t"]
    for key in (
        "minfer_host_staging_waits_total",
        "minfer_host_staging_wait_seconds_total",
        "minfer_engine_step_seconds_total",
        "minfer_preemptions_total",
        "minfer_prefill_tokens_total",
        "minfer_decode_tokens_total",
        "minfer_admission_stops_memory_total",
        "minfer_admission_stops_does_not_fit_total",
        "minfer_admission_stops_slots_total",
        "minfer_admission_shortfall_blocks_total",
    ):
        if key in rows[-1]:
            name = key.removeprefix("minfer_").removesuffix("_total")
            out[f"{name}_per_s"] = (rows[-1][key] - rows[0].get(key, 0.0)) / span
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="7b", help="model id, or 7b / 1.5b")
    parser.add_argument("--trace", type=Path, default=None, help="compiled real trace")
    parser.add_argument("--variants", nargs="+", default=["base"], choices=list(VARIANTS))
    parser.add_argument("--concurrency", nargs="+", type=int, default=[96, 128])
    parser.add_argument("--duration", type=float, default=360.0)
    parser.add_argument("--warmup", type=float, default=120.0)
    parser.add_argument("--ramp", type=float, default=30.0)
    parser.add_argument("--repeat", type=int, default=1, help="repetition index (file names)")
    parser.add_argument(
        "--name", default=None, help="key of the results (default: the variant's name)"
    )
    parser.add_argument("--port", type=int, default=8063)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="keep results already on disk")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    from transformers import AutoTokenizer

    model = models.resolve(args.model)
    tag = models.tag(model)
    trace_path = args.trace or Path(f"bench/traces/real_{tag}.jsonl")
    out = args.out or Path(f"bench/results/overload_{tag}.json")
    raw = Path("bench/results/overload")
    raw.mkdir(parents=True, exist_ok=True)
    trace = rt.RealTrace.load(trace_path)
    tokenizer = AutoTokenizer.from_pretrained(model)
    label, base_flags, hints = CONFIGS[BASE]
    results: dict[str, Any] = {
        "model": model,
        "trace": {**trace.header, "sha256": trace.sha256, "sessions": len(trace.sessions)},
        "base": {"config": BASE, "label": label, "flags": base_flags},
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "ramp_s": args.ramp,
        "variants": {},
    }
    if args.resume and out.exists():
        results = json.loads(out.read_text())
    for variant in args.variants:
        name = args.name or variant
        entry = results["variants"].setdefault(name, {"flags": VARIANTS[variant], "runs": {}})
        for concurrency in args.concurrency:
            key = f"{concurrency}/r{args.repeat}"
            if key in entry["runs"]:
                continue
            stem = f"{tag}_{name}_c{concurrency}_r{args.repeat}"
            request_log = raw / f"requests_{stem}.jsonl"
            request_log.unlink(missing_ok=True)
            flags = [*base_flags, *VARIANTS[variant], "--request-log", str(request_log)]
            log = Path("bench/logs") / f"overload_{stem}.log"
            options = rt.ReplayOptions(
                concurrency, session_hints=hints, duration=args.duration, ramp=args.ramp
            )
            try:
                with launch("minfer", model, args.port, log, flags) as (url, argv):
                    entry["command"] = argv
                    code = code_version()
                    poller = MetricsPoller(url)
                    poller.start()
                    records, elapsed = asyncio.run(rt.replay(trace, tokenizer, url, model, options))
                    poller.stop.set()
                    poller.join()
                    metrics = scrape(url)
            except RuntimeError as error:
                entry["runs"][key] = {"error": str(error)}
                out.write_text(json.dumps(results, indent=1) + "\n")
                print(f"[{name}] c={concurrency} failed: {error}", flush=True)
                continue
            (raw / f"metrics_{stem}.jsonl").write_text(
                "".join(json.dumps(s) + "\n" for s in poller.samples)
            )
            lines = request_log.read_text().splitlines() if request_log.exists() else []
            row = rt.summarize(records, elapsed, args.warmup)
            row["code"] = code
            row["breakdown"] = breakdown([json.loads(x) for x in lines], args.warmup)
            row["gauges"] = summarize_samples(poller.samples, args.warmup)
            row["server_metrics"] = {k: v for k, v in metrics.items() if k.startswith("minfer_")}
            entry["runs"][key] = row
            out.write_text(json.dumps(results, indent=1) + "\n")
            parts = row["breakdown"]
            print(
                f"[{name}] c={concurrency} r{args.repeat}: "
                f"{row['output_tokens_per_second']:.0f} tok/s, TTFT p50/p99 "
                f"{row['ttft_ms']['p50']:.0f}/{row['ttft_ms']['p99']:.0f} ms; server-side mean "
                + ", ".join(
                    f"{k} {parts.get(f'{k}_s', {}).get('mean', float('nan')):.3f} s"
                    for k in ("queue", "restore", "prefill", "decode")
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
