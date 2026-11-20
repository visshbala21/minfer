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
    """CUDA graphs on vs off."""
    data = _load("decode_latency.json")
    if not data:
        return MISSING
    rows = [
        [
            str(r["batch_size"]),
            str(r["context_tokens"]),
            f"{r['off_ms']:.2f}",
            f"{r['on_ms']:.2f}",
            f"{r['speedup']:.2f}×",
        ]
        for r in data["speedup_p50"]
    ]
    header = ["batch", "context (tokens)", "graphs off (ms)", "graphs on (ms)", "speedup"]
    return _table(header, rows)


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
    "chunked-prefill": chunked_prefill_table,
}


def render(text: str) -> str:
    """Replace every generated section of ``text``."""
    for name, build in SECTIONS.items():
        body = build()
        pattern = re.compile(rf"(<!-- BEGIN {name} -->\n).*?(<!-- END {name} -->)", re.DOTALL)
        text, count = pattern.subn(lambda m, body=body: f"{m[1]}{body}\n{m[2]}", text)
        if count == 0:
            print(f"warning: README has no section {name!r}", file=sys.stderr)
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
