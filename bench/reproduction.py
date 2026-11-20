"""Compare two complete runs of ``make bench`` on their headline numbers.

    .venv/bin/python -m bench.reproduction --rerun /path/to/other/clone/bench/results \
        --rerun-commit <sha> --note "how the second run was produced"

``--reference`` (default: this checkout's ``bench/results``) is the run the README is
generated from; ``--rerun`` is the ``bench/results`` directory of an independent run of the
same code. The output, ``bench/results/reproduction.json``, holds both values of every
headline metric of the three-engine matrix, their ratio, and whether each qualitative claim
the README makes holds in both runs. Nothing is averaged or filtered: a metric that moved
is reported with how far it moved.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

ENGINES = ("minfer", "vllm", "sglang")

AGENT_METRICS: dict[str, Callable[[dict], float]] = {
    "ttft_p50_ms": lambda r: r["ttft_ms"]["p50"],
    "ttft_p99_ms": lambda r: r["ttft_ms"]["p99"],
    "tpot_p50_ms": lambda r: r["tpot_ms"]["p50"],
    "e2e_p50_ms": lambda r: r["e2e_ms"]["p50"],
    "output_tokens_per_second": lambda r: r["output_tokens_per_second"],
    "prefix_cache_hit_rate": lambda r: r["server_prefix_cache_hit_rate"],
}
SHAREGPT_METRICS: dict[str, Callable[[dict], float]] = {
    "ttft_p50_ms": lambda r: r["median_ttft_ms"],
    "tpot_p50_ms": lambda r: r["median_tpot_ms"],
    "output_tokens_per_second": lambda r: r["output_throughput"],
}
WORKLOADS = {"agent": AGENT_METRICS, "sharegpt": SHAREGPT_METRICS}


def load_matrix(results: Path) -> dict[str, Any]:
    """The three per-engine matrix files of one run."""
    return {e: json.loads((results / "matrix" / f"{e}.json").read_text()) for e in ENGINES}


def compare(reference: dict[str, Any], rerun: dict[str, Any]) -> dict[str, Any]:
    """``workload -> engine -> rate -> metric -> {reference, rerun, ratio}``."""
    out: dict[str, Any] = {}
    for workload, metrics in WORKLOADS.items():
        for engine in ENGINES:
            for rate, ref_row in reference[engine][workload].items():
                new_row = rerun[engine][workload][rate]
                cell = out.setdefault(workload, {}).setdefault(engine, {}).setdefault(rate, {})
                if workload == "agent":
                    cell["same_trace"] = ref_row["trace_sha256"] == new_row["trace_sha256"]
                    cell["failed"] = {"reference": ref_row["failed"], "rerun": new_row["failed"]}
                for name, get in metrics.items():
                    a, b = get(ref_row), get(new_row)
                    cell[name] = {"reference": a, "rerun": b, "ratio": b / a}
    return out


def compare_ablations(reference: Path, rerun: Path) -> dict[str, Any]:
    """Saturated throughput and TPOT p50 of every ablation configuration in both runs."""
    ref = json.loads((reference / "ablations.json").read_text())["configs"]
    new = json.loads((rerun / "ablations.json").read_text())["configs"]
    out: dict[str, Any] = {}
    for name, config in ref.items():
        out[name] = {"label": config["label"]}
        for metric in ("output_tokens_per_second", "tpot_p50_ms"):
            a = AGENT_METRICS[metric](config["max"])
            b = AGENT_METRICS[metric](new[name]["max"])
            out[name][metric] = {"reference": a, "rerun": b, "ratio": b / a}
    return out


def claims(matrix: dict[str, Any]) -> dict[str, bool]:
    """The README's qualitative statements about the agent workload, evaluated on one run."""
    agent = {e: matrix[e]["agent"] for e in ENGINES}
    steady = [r for r in agent["minfer"] if r != "max"]
    ttft50, ttft99 = AGENT_METRICS["ttft_p50_ms"], AGENT_METRICS["ttft_p99_ms"]
    tpot50, tok = AGENT_METRICS["tpot_p50_ms"], AGENT_METRICS["output_tokens_per_second"]
    others = ("vllm", "sglang")

    def minfer_lower(get: Callable[[dict], float], rates: list[str]) -> bool:
        return all(get(agent["minfer"][r]) < get(agent[o][r]) for r in rates for o in others)

    return {
        "minfer has the lowest TTFT p50 at every rate from 1 to 16 req/s": minfer_lower(
            ttft50, steady
        ),
        "minfer has the lowest TTFT p99 at every rate from 1 to 16 req/s": minfer_lower(
            ttft99, steady
        ),
        "minfer has the lowest TTFT p50 and p99 at saturation": minfer_lower(ttft50, ["max"])
        and minfer_lower(ttft99, ["max"]),
        "minfer's TPOT p50 is higher than vLLM's at every rate": all(
            tpot50(agent["minfer"][r]) > tpot50(agent["vllm"][r]) for r in [*steady, "max"]
        ),
        "minfer's saturated throughput is below vLLM's and SGLang's": all(
            tok(agent["minfer"]["max"]) < tok(agent[o]["max"]) for o in others
        ),
        "no request failed for any engine on either workload": all(
            row["failed"] == 0
            for e in ENGINES
            for workload in WORKLOADS
            for row in matrix[e][workload].values()
        ),
    }


def spread(comparison: dict[str, Any]) -> dict[str, Any]:
    """Smallest and largest rerun/reference ratio per workload, engine and metric."""
    out: dict[str, Any] = {}
    for workload, engines in comparison.items():
        for engine, rates in engines.items():
            for name in WORKLOADS[workload]:
                ratios = {rate: cell[name]["ratio"] for rate, cell in rates.items()}
                low, high = min(ratios, key=ratios.get), max(ratios, key=ratios.get)
                out.setdefault(workload, {}).setdefault(engine, {})[name] = {
                    "min_ratio": ratios[low],
                    "min_at_rate": low,
                    "max_ratio": ratios[high],
                    "max_at_rate": high,
                }
    return out


def _git_head(path: Path) -> str:
    done = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", type=Path, default=Path("bench/results"))
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--reference-commit", help="commit that holds the reference results")
    parser.add_argument("--rerun-commit", help="commit the rerun was produced from")
    parser.add_argument("--note", default="", help="how the rerun was produced")
    parser.add_argument("--out", type=Path, default=Path("bench/results/reproduction.json"))
    args = parser.parse_args()

    reference, rerun = load_matrix(args.reference), load_matrix(args.rerun)
    comparison = compare(reference, rerun)
    report = {
        "note": args.note,
        "reference_commit": args.reference_commit or _git_head(args.reference),
        "rerun_commit": args.rerun_commit or _git_head(args.rerun),
        "claims": {
            text: {"reference": held, "rerun": claims(rerun)[text]}
            for text, held in claims(reference).items()
        },
        "ratio_spread": spread(comparison),
        "ablations_at_max": compare_ablations(args.reference, args.rerun),
        "comparison": comparison,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for text, held in report["claims"].items():
        print(f"{'OK ' if held['reference'] and held['rerun'] else 'DIFFERS'}  {text}  {held}")


if __name__ == "__main__":
    main()
