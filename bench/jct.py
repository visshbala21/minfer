"""Agent job completion time (JCT) on real traces: whole sessions, real tool-call pauses.

    .venv/bin/python -m bench.jct --out bench/results/jct_7b_r1.json
    .venv/bin/python -m bench.jct --configs minfer/session_host --concurrency 16

The metric Continuum (arXiv 2511.02230) optimizes: how long an agent takes to finish its
whole job — every model call of a session plus the tool-call pauses between them — rather
than any single request's latency. The trace is the real coding-agent trace of
``bench/real_traces.py`` (Open-SWE-Traces content, TraceLab tool-call pauses), replayed
closed-loop at a fixed number of sessions in flight, as in ``bench/pressure.py``.

**What is measured.** At concurrency ``C`` the workers take sessions in trace order. The
first ``C`` sessions start the load (over a ramp, against an empty cache) and are not
measured; the next ``K`` sessions (``MEASURED``) are. The load keeps going — workers keep
starting later sessions — until every measured session has finished, so each one runs
start to end at the full concurrency; then the replay stops. Every engine is measured on
the same ``K`` sessions with the same pauses. A session's JCT is from its first call being
sent to its last call completing; ``model_time`` is its JCT minus the pauses it waited,
the part an engine can change.

Configurations are those of ``bench/pressure.py``; the default set is each engine's best
under memory pressure (minfer full stack, vLLM and SGLang with their host-memory KV
offloading). One file is one repetition; ``bench/medians.py`` combines repetitions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bench import agent_workload as aw
from bench import models
from bench import real_traces as rt
from bench.pressure import CONFIGS
from bench.run_matrix import scrape
from bench.servers import code_version, launch

DEFAULT_CONFIGS = ("minfer/session_host", "vllm/offload", "sglang/hicache")
# concurrency -> measured sessions. 16: well within GPU KV capacity; 64: at it; 128: well
# past it (bench/results/pressure_7b.json).
MEASURED = {16: 32, 64: 64, 128: 96}


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {}
    return {
        "mean": float(array.mean()),
        **{f"p{p}": float(np.percentile(array, p)) for p in (50, 90, 99)},
        "max": float(array.max()),
    }


def summarize_sessions(
    sessions: list[rt.SessionRecord],
    records: list[aw.RequestRecord],
    measured: range,
    elapsed: float,
) -> dict[str, Any]:
    """JCT statistics of the measured sessions, plus their requests' latency summary."""
    chosen = sorted((s for s in sessions if s.index in measured), key=lambda s: s.index)
    complete = [s for s in chosen if s.finished is not None and not s.failed]
    ids = {s.session_id for s in chosen}
    calls = [
        r
        for r in records
        if r.session_id in ids and (r.finished is not None or r.error is not None)
    ]
    summary = aw.summarize(calls, elapsed)
    good = [r for r in calls if r.error is None and r.cached_tokens is not None]
    prompt = sum(r.prompt_tokens for r in good)
    computed = sum(r.prompt_tokens - r.cached_tokens for r in good)
    return {
        "measured_sessions": len(measured),
        "completed_sessions": len(complete),
        "failed_sessions": sum(s.failed for s in chosen),
        "elapsed_s": elapsed,
        "jct_s": _percentiles([s.completion_time for s in complete]),
        "model_time_s": _percentiles([s.completion_time - s.pause_s for s in complete]),
        "pause_s": _percentiles([s.pause_s for s in complete]),
        "per_session": [
            {
                "index": s.index,
                "session_id": s.session_id,
                "jct_s": s.completion_time,
                "pause_s": s.pause_s,
                "calls": s.calls,
            }
            for s in complete
        ],
        "requests": {
            key: summary[key]
            for key in ("requests", "completed", "failed", "errors", "ttft_ms", "tpot_ms")
        },
        "prefill_tokens_computed": computed,
        "cache_hit_rate": 1.0 - computed / prompt if prompt else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=models.PRIMARY, help="model id, or 7b / 1.5b")
    parser.add_argument("--trace", type=Path, default=None, help="compiled real trace")
    parser.add_argument(
        "--configs", nargs="+", default=list(DEFAULT_CONFIGS), choices=list(CONFIGS)
    )
    parser.add_argument("--concurrency", nargs="+", type=int, default=list(MEASURED))
    parser.add_argument("--ramp", type=float, default=30.0, help="seconds to start the load over")
    parser.add_argument(
        "--measured", type=int, default=None, help="measured sessions (default: MEASURED)"
    )
    parser.add_argument("--port", type=int, default=8062)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="keep results already on disk")
    parser.add_argument(
        "--remeasure",
        action="store_true",
        help="with --resume: measure the selected configurations again, keep the others",
    )
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    from transformers import AutoTokenizer

    def count(concurrency: int) -> int:
        return args.measured or MEASURED.get(concurrency, concurrency)

    model = models.resolve(args.model)
    tag = models.tag(model)
    trace = rt.RealTrace.load(args.trace or Path(f"bench/traces/real_{tag}.jsonl"))
    tokenizer = AutoTokenizer.from_pretrained(model)
    results: dict[str, Any] = {
        "model": model,
        "trace": {**trace.header, "sha256": trace.sha256, "sessions": len(trace.sessions)},
        "ramp_s": args.ramp,
        "measured": {str(c): [c, c + count(c)] for c in args.concurrency},
        "configs": {},
    }
    if args.resume and args.out.exists():
        results = json.loads(args.out.read_text())
        if args.remeasure:
            for name in args.configs:
                results["configs"].pop(name, None)
    for name in args.configs:
        engine, variant = name.split("/")
        label, flags, hints = CONFIGS[name]
        entry = results["configs"].setdefault(
            name, {"engine": engine, "label": label, "flags": flags, "session_hints": hints}
        )
        runs = entry.setdefault("runs", {})
        for concurrency in args.concurrency:
            if str(concurrency) in runs:
                continue
            measured = range(concurrency, concurrency + count(concurrency))
            if measured.stop > len(trace.sessions):
                raise SystemExit(f"the trace has too few sessions for c={concurrency}")
            log = Path("bench/logs") / f"jct_{tag}_{engine}_{variant}_c{concurrency}.log"
            options = rt.ReplayOptions(
                concurrency, session_hints=hints, ramp=args.ramp, measured=measured
            )
            sessions: list[rt.SessionRecord] = []
            try:
                with launch(engine, model, args.port, log, flags) as (url, argv):
                    entry["command"] = argv
                    entry["code"] = code_version()
                    records, elapsed = asyncio.run(
                        rt.replay(trace, tokenizer, url, model, options, session_records=sessions)
                    )
                    metrics = scrape(url)
            except RuntimeError as error:
                runs[str(concurrency)] = {"error": str(error)}
                args.out.write_text(json.dumps(results, indent=1) + "\n")
                print(f"[{name}] c={concurrency} failed: {error}", flush=True)
                continue
            row = summarize_sessions(sessions, records, measured, elapsed)
            row["server_metrics"] = {
                key: value for key, value in metrics.items() if key.startswith("minfer_")
            }
            runs[str(concurrency)] = row
            args.out.write_text(json.dumps(results, indent=1) + "\n")
            jct = row["jct_s"]
            print(
                f"[{name}] c={concurrency}: {row['completed_sessions']}/{len(measured)} sessions "
                f"in {elapsed:.0f} s; JCT p50 {jct.get('p50', float('nan')):.1f} "
                f"p90 {jct.get('p90', float('nan')):.1f} p99 {jct.get('p99', float('nan')):.1f} s; "
                f"model time p50 {row['model_time_s'].get('p50', float('nan')):.1f} s; "
                f"hit {row['cache_hit_rate']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
