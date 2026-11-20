"""Fill the README's result tables from the committed results JSON.

    .venv/bin/python -m bench.readme_tables          # rewrite README.md in place
    .venv/bin/python -m bench.readme_tables --check  # fail if README.md is out of date

The README contains marker pairs ``<!-- BEGIN name -->`` / ``<!-- END name -->``. Everything
between a pair is generated here from files under ``bench/results/``; no benchmark number
in the README is typed by hand. A section whose results file does not exist yet renders as
a "not measured" note rather than disappearing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

RESULTS = Path("bench/results")
README = Path("README.md")
ENGINES = {"minfer": "minfer", "vllm": "vLLM", "sglang": "SGLang"}
MISSING = "_Not measured yet._"


def _load(relative: str) -> Any:
    path = RESULTS / relative
    return json.loads(path.read_text()) if path.exists() else None


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _matrix() -> dict[str, Any]:
    loaded = {name: _load(f"matrix/{name}.json") for name in ENGINES}
    return {name: data for name, data in loaded.items() if data}


def _best(values: list[float | None], lower_is_better: bool) -> int | None:
    present = [(v, i) for i, v in enumerate(values) if v is not None]
    if len(present) < 2:
        return None
    return (min(present) if lower_is_better else max(present))[1]


def _matrix_table(
    workload: str, columns: list[tuple[str, Callable[[dict], Any], str, bool]]
) -> str:
    """One row per (rate, engine); the best engine per rate and column is bold."""
    matrix = _matrix()
    engines = [e for e in matrix if matrix[e].get(workload)]
    if not engines:
        return MISSING
    rates = list(matrix[engines[0]][workload])
    rows = []
    for rate in rates:
        present = [e for e in engines if rate in matrix[e][workload]]
        cells = [[get(matrix[e][workload][rate]) for e in present] for _, get, _, _ in columns]
        for index, engine in enumerate(present):
            row = [rate if index == 0 else "", ENGINES[engine]]
            for (_, _, fmt, lower), values in zip(columns, cells, strict=True):
                texts = ["—" if v is None else format(v, fmt) for v in values]
                best = _best(values, lower)
                # Bold only a best value that is distinguishable at the printed precision.
                unique = best is not None and texts.count(texts[best]) == 1
                row.append(f"**{texts[index]}**" if unique and best == index else texts[index])
            rows.append(row)
    return _table(["rate (req/s)", "engine", *[name for name, _, _, _ in columns]], rows)


def agent_table() -> str:
    """Agent workload: every engine at every rate."""
    return _matrix_table(
        "agent",
        [
            ("out tok/s", lambda r: r["output_tokens_per_second"], ".0f", False),
            ("TTFT p50 (ms)", lambda r: r["ttft_ms"]["p50"], ".1f", True),
            ("TTFT p99 (ms)", lambda r: r["ttft_ms"]["p99"], ".1f", True),
            ("TPOT p50 (ms)", lambda r: r["tpot_ms"]["p50"], ".2f", True),
            ("TPOT p99 (ms)", lambda r: r["tpot_ms"]["p99"], ".2f", True),
            ("E2E p50 (ms)", lambda r: r["e2e_ms"]["p50"], ".0f", True),
            ("E2E p99 (ms)", lambda r: r["e2e_ms"]["p99"], ".0f", True),
            ("prefix hit rate", lambda r: r["server_prefix_cache_hit_rate"], ".1%", False),
            ("failed", lambda r: r["failed"], "d", True),
        ],
    )


def sharegpt_table() -> str:
    """ShareGPT control workload (``vllm bench serve``)."""
    return _matrix_table(
        "sharegpt",
        [
            ("out tok/s", lambda r: r["output_throughput"], ".0f", False),
            ("TTFT p50 (ms)", lambda r: r["median_ttft_ms"], ".1f", True),
            ("TTFT p99 (ms)", lambda r: r["p99_ttft_ms"], ".1f", True),
            ("TPOT p50 (ms)", lambda r: r["median_tpot_ms"], ".2f", True),
            ("TPOT p99 (ms)", lambda r: r["p99_tpot_ms"], ".2f", True),
            ("E2E p50 (ms)", lambda r: r["median_e2el_ms"], ".0f", True),
            ("E2E p99 (ms)", lambda r: r["p99_e2el_ms"], ".0f", True),
            ("failed", lambda r: r["failed"], "d", True),
        ],
    )


def decode_latency_table() -> str:
    """Decode-step latency as optimizations are added."""
    data = _load("decode_latency.json")
    if not data:
        return MISSING
    runs = data["runs"]
    stages = [
        ("cuda_graphs_off", "eager"),
        ("cuda_graphs_on", "+ CUDA graphs"),
        ("cuda_graphs_on_triton_decode", "+ Triton decode"),
        ("cuda_graphs_on_triton_decode_fused", "+ fused kernels (default)"),
    ]
    rows = []
    for index, first in enumerate(runs[stages[0][0]]):
        cells = [str(first["batch_size"]), str(first["context_tokens"])]
        cells += [f"{runs[key][index]['step_ms_p50']:.2f}" for key, _ in stages]
        total = first["step_ms_p50"] / runs[stages[-1][0]][index]["step_ms_p50"]
        rows.append([*cells, f"{total:.1f}×"])
    header = ["batch", "context (tokens)", *[f"{label} (ms)" for _, label in stages]]
    return _table([*header, "eager → default"], rows)


def prefill_latency_table() -> str:
    """One prefill pass: eager, fused, graph."""
    data = _load("prefill_latency.json")
    if not data:
        return MISSING
    runs = data["runs"]
    rows = []
    for index, first in enumerate(runs["eager_unfused"]):
        values = [
            runs[key][index]["pass_ms_p50"]
            for key in ("eager_unfused", "eager_fused", "graph_fused")
        ]
        rows.append(
            [str(first["tokens"]), *[f"{v:.2f}" for v in values], f"{values[0] / values[2]:.1f}×"]
        )
    header = [
        "prompt tokens",
        "eager (ms)",
        "eager + fused kernels (ms)",
        "default (ms)",
        "speedup",
    ]
    return _table(header, rows)


def ablation_table() -> str:
    """Agent workload with one feature changed at a time."""
    data = _load("ablations.json")
    if not data:
        return MISSING
    rows = []
    for config in data["configs"].values():
        row = [config["label"], " ".join(f"`{flag}`" for flag in config["flags"]) or "—"]
        for rate in data["rates"]:
            r = config[rate]
            hit = r["server_prefix_cache_hit_rate"]
            row += [
                f"{r['output_tokens_per_second']:.0f}",
                f"{r['ttft_ms']['p50']:.0f} / {r['ttft_ms']['p99']:.0f}",
                f"{r['tpot_ms']['p50']:.2f} / {r['tpot_ms']['p99']:.2f}",
                "—" if hit is None else f"{hit:.1%}",
            ]
        rows.append(row)
    header = ["configuration", "flags"]
    for rate in data["rates"]:
        name = f"{rate} req/s" if rate != "max" else "max"
        header += [f"{name}: tok/s", "TTFT p50 / p99 (ms)", "TPOT p50 / p99 (ms)", "hit rate"]
    return _table(header, rows)


def speculation_table() -> str:
    """Speculative decoding vs off."""
    data = _load("speculation.json")
    if not data:
        return MISSING
    rows = []
    for workload, settings in data["workloads"].items():
        for name, r in settings.items():
            versus = r.get("versus_off")
            same = "—"
            if versus:
                total = versus["identical_outputs"] + versus["diverged_outputs"]
                same = f"{versus['identical_outputs']}/{total}"
            accepted = r["acceptance_rate"]
            rows.append(
                [
                    workload,
                    name.replace("_", " "),
                    f"{r['output_tokens_per_second']:.0f}",
                    "—" if name == "off" else f"{r['throughput_vs_off']:.2f}×",
                    f"{r['tpot_ms_p50']:.2f}",
                    f"{r['tokens_per_step']:.1f}",
                    "—" if accepted is None else f"{accepted:.0%}",
                    same,
                ]
            )
    header = ["workload", "speculation", "out tok/s", "vs off", "TPOT p50 (ms)", "tokens/step"]
    return _table([*header, "accepted", "outputs identical to off"], rows)


def fp8_table() -> str:
    """GSM8K accuracy, capacity and speed under FP8."""
    data = _load("gsm8k_fp8.json")
    if not data:
        return MISSING
    base = data["settings"]["bf16"]
    labels = {
        "bf16": "bf16",
        "fp8_kv": "FP8 KV cache",
        "fp8_weights": "FP8 weights",
        "fp8_weights_and_kv": "FP8 weights + KV",
    }
    rows = []
    for name, r in data["settings"].items():
        steps = " / ".join(f"{d['step_ms_p50']:.2f}" for d in r["decode_step"])
        rows.append(
            [
                labels.get(name, name),
                f"{r['accuracy']:.1%}",
                f"{r['kv_capacity_tokens'] / base['kv_capacity_tokens']:.2f}×",
                f"{r['weight_bytes'] / 2**30:.2f} GB",
                steps,
            ]
        )
    header = [
        "setting",
        "GSM8K (200)",
        "KV capacity",
        "weights",
        "decode step ms (batch 1 / 64 / 256)",
    ]
    return _table(header, rows)


def triton_table() -> str:
    """Triton vs FlashInfer decode attention, replayed from a CUDA graph."""
    data = _load("triton_kernel.json")
    if not data:
        return MISSING
    rows = [
        [
            str(r["batch_size"]),
            f"{r['context_tokens']:,}",
            f"{r['flashinfer_graph_ms'] * 1e3:.1f}",
            f"{r['triton_graph_ms'] * 1e3:.1f}",
            f"{r['speedup_in_graph']:.2f}×",
            f"{r['triton_vs_flashinfer_max_abs']:.4f}",
        ]
        for r in data["rows"]
    ]
    header = ["batch", "context", "FlashInfer (µs/layer)", "Triton (µs/layer)", "speedup"]
    return _table([*header, "max abs difference"], rows)


def chunked_prefill_table() -> str:
    """Chunked prefill settings on the mixed workload."""
    data = _load("chunked_prefill.json")
    if not data:
        return MISSING
    rows = []
    for name, s in data["settings"].items():
        short, long = s["short"], s["long"]
        rows.append(
            [
                "off" if name == "off" else name.split("_")[1],
                f"{short['inter_token_ms']['p99']:.1f}",
                f"{short['inter_token_ms']['max']:.0f}",
                f"{short['tpot_ms']['p50']:.2f}",
                f"{short['tpot_ms']['p99']:.2f}",
                f"{long['ttft_ms']['p50']:.0f}",
                f"{s['output_tokens_per_second']:.0f}",
            ]
        )
    header = ["chunk (tokens)", "ITL p99 (ms)", "ITL max (ms)", "TPOT p50 (ms)", "TPOT p99 (ms)"]
    return _table([*header, "long-prompt TTFT p50 (ms)", "out tok/s"], rows)


SECTIONS: dict[str, Callable[[], str]] = {
    "agent-results": agent_table,
    "sharegpt-results": sharegpt_table,
    "decode-latency": decode_latency_table,
    "prefill-latency": prefill_latency_table,
    "chunked-prefill": chunked_prefill_table,
    "ablations": ablation_table,
    "speculation": speculation_table,
    "fp8": fp8_table,
    "triton": triton_table,
}


def render(text: str) -> str:
    """Replace every generated section of ``text``."""
    for name, build in SECTIONS.items():
        body = build()
        pattern = re.compile(rf"(<!-- BEGIN {name} -->\n).*?(<!-- END {name} -->)", re.DOTALL)
        text = pattern.sub(lambda m, body=body: f"{m[1]}{body}\n{m[2]}", text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if README.md would change")
    args = parser.parse_args()
    current = README.read_text()
    updated = render(current)
    if args.check:
        sys.exit(0 if updated == current else "README.md is out of date: run bench.readme_tables")
    README.write_text(updated)


if __name__ == "__main__":
    main()
