"""Generate every chart from the committed results JSON.

    .venv/bin/python -m bench.plot

Reads ``bench/results/`` and writes PNGs to ``bench/plots/``. Nothing is typed in by hand:
a chart exists only if its results file does, and shows exactly what the file contains.

Style: one y-axis per panel (different measures get their own panels), engines in a fixed
color order (so a color always means the same engine), thin lines with markers, a legend
plus direct labels at the line ends, and recessive grid lines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

RESULTS = Path("bench/results")
PLOTS = Path("bench/plots")

# Fixed categorical order: color follows the engine, never its rank in a chart.
SERIES = {"minfer": "#2a78d6", "vllm": "#eb6834", "sglang": "#1baf7a"}
LABELS = {"minfer": "minfer", "vllm": "vLLM", "sglang": "SGLang"}
OFF_ON = {"off": "#52514e", "on": "#2a78d6"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "font.size": 9.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.5,
        "legend.frameon": False,
    }
)


def _line(ax: Axes, xs: list, ys: list, color: str, label: str) -> None:
    """One series: a thin line with ringed markers and a direct label at its end."""
    ring = {"markeredgecolor": SURFACE, "markeredgewidth": 1.5}
    ax.plot(xs, ys, color=color, marker="o", label=label, **ring)
    last = ys[-1]
    text = f" {last:,.0f}" if abs(last) >= 100 else f" {last:.3g}"
    ax.annotate(text, (xs[-1], last), color=MUTED, fontsize=8.5, va="center")


def _finish(fig: plt.Figure, axes: list[Axes], title: str, note: str, name: str) -> None:
    """Title, footnote, a legend when there is more than one series, and save."""
    handles, labels = axes[0].get_legend_handles_labels()
    if len(labels) > 1:
        fig.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(0.99, 1))
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.005, note, color=MUTED, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / name, dpi=160)
    plt.close(fig)
    print(f"wrote {PLOTS / name}")


def load_matrix() -> dict[str, dict[str, Any]]:
    """Per-engine results of ``bench/run_matrix.py``, in the fixed engine order."""
    found = {}
    for engine in SERIES:
        path = RESULTS / "matrix" / f"{engine}.json"
        if path.exists():
            found[engine] = json.loads(path.read_text())
    return found


def _agent_value(row: dict[str, Any], metric: str) -> float:
    group, _, stat = metric.partition(".")
    return row[group][stat] if stat else row[group]


def _sharegpt_value(row: dict[str, Any], metric: str) -> float:
    names = {
        "output_tokens_per_second": "output_throughput",
        "ttft_ms.p50": "median_ttft_ms",
        "ttft_ms.p99": "p99_ttft_ms",
        "tpot_ms.p50": "median_tpot_ms",
        "tpot_ms.p99": "p99_tpot_ms",
        "e2e_ms.p50": "median_e2el_ms",
        "e2e_ms.p99": "p99_e2el_ms",
    }
    return row[names[metric]]


PANELS = [
    ("output_tokens_per_second", "Output throughput (tok/s)"),
    ("ttft_ms.p50", "TTFT p50 (ms)"),
    ("ttft_ms.p99", "TTFT p99 (ms)"),
    ("tpot_ms.p50", "TPOT p50 (ms)"),
    ("tpot_ms.p99", "TPOT p99 (ms)"),
    ("e2e_ms.p50", "End-to-end latency p50 (ms)"),
]


def plot_workload(matrix: dict[str, dict[str, Any]], workload: str, title: str, note: str) -> None:
    """Six small multiples (one measure each) of every engine across request rates."""
    engines = [e for e in matrix if matrix[e].get(workload)]
    if not engines:
        return
    value = _agent_value if workload == "agent" else _sharegpt_value
    rates = list(matrix[engines[0]][workload])
    fig, grid = plt.subplots(2, 3, figsize=(13, 7))
    axes = list(grid.flat)
    for ax, (metric, label) in zip(axes, PANELS, strict=True):
        for engine in engines:
            rows = matrix[engine][workload]
            xs = [r for r in rates if r in rows]
            _line(ax, xs, [value(rows[r], metric) for r in xs], SERIES[engine], LABELS[engine])
        ax.set_title(label)
        ax.set_xlabel("request rate (req/s)")
        ax.set_ylim(bottom=0)
        ax.margins(x=0.12)
    _finish(fig, axes, title, note, f"{workload}_vs_rate.png")


def plot_agent_hit_rate(matrix: dict[str, dict[str, Any]]) -> None:
    """Prefix-cache hit rate and first-turn vs later-turn TTFT on the agent workload."""
    engines = [e for e in matrix if matrix[e].get("agent")]
    if not engines:
        return
    rates = list(matrix[engines[0]]["agent"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    specs = [
        ("server_prefix_cache_hit_rate", "Prefix-cache hit rate (server counters)"),
        ("ttft_first_turn_ms.p50", "TTFT p50, first turn (ms)"),
        ("ttft_later_turns_ms.p50", "TTFT p50, later turns (ms)"),
    ]
    for ax, (metric, label) in zip(axes, specs, strict=True):
        for engine in engines:
            rows = matrix[engine]["agent"]
            points = [(r, _agent_value(rows[r], metric)) for r in rates if r in rows]
            points = [(r, v) for r, v in points if v is not None]
            if points:
                xs, ys = zip(*points, strict=True)
                _line(ax, list(xs), list(ys), SERIES[engine], LABELS[engine])
        ax.set_title(label)
        ax.set_xlabel("request rate (req/s)")
        ax.set_ylim(bottom=0)
        ax.margins(x=0.12)
    note = "Hit rate = cached prompt tokens / prompt tokens, from each server's own metrics."
    _finish(fig, list(axes), "Agent workload: prefix reuse", note, "agent_prefix_cache.png")


def plot_decode_latency() -> None:
    """Decode-step latency as optimizations are added (Phases 3, 8, 11)."""
    path = RESULTS / "decode_latency.json"
    if not path.exists():
        return
    runs = json.loads(path.read_text())["runs"]
    stages = [
        ("cuda_graphs_off", "eager", MUTED),
        ("cuda_graphs_on", "+ CUDA graphs", SERIES["sglang"]),
        ("cuda_graphs_on_triton_decode", "+ Triton decode", SERIES["vllm"]),
        ("cuda_graphs_on_triton_decode_fused", "+ fused kernels (default)", SERIES["minfer"]),
    ]
    stages = [stage for stage in stages if stage[0] in runs]
    contexts = sorted({r["context_tokens"] for r in runs[stages[0][0]]})
    fig, axes = plt.subplots(1, len(contexts), figsize=(6.2 * len(contexts), 4.2), squeeze=False)
    for ax, context in zip(axes[0], contexts, strict=True):
        for key, label, color in stages:
            subset = [r for r in runs[key] if r["context_tokens"] == context]
            xs = [str(r["batch_size"]) for r in subset]
            _line(ax, xs, [r["step_ms_p50"] for r in subset], color, label)
        ax.set_title(f"Decode step p50 (ms), {context}-token contexts")
        ax.set_xlabel("batch size")
        ax.set_ylim(bottom=0)
        ax.margins(x=0.1)
    note = "Wall-clock time of one engine step in steady-state decode (bench/decode_latency.py)."
    _finish(
        fig,
        list(axes[0]),
        "Decode-step latency, one optimization at a time",
        note,
        "decode_latency.png",
    )


def plot_chunked_prefill() -> None:
    """Latency of short requests vs prefill chunk size (Phase 3)."""
    path = RESULTS / "chunked_prefill.json"
    if not path.exists():
        return
    settings = json.loads(path.read_text())["settings"]
    xs = ["off" if k == "off" else k.split("_")[1] for k in settings]
    specs = [
        (("short", "inter_token_ms", "p99"), "Inter-token latency p99 (ms)"),
        (("short", "tpot_ms", "p99"), "Per-request TPOT p99 (ms)"),
        (("long", "ttft_ms", "p50"), "Long-prompt TTFT p50 (ms)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, ((kind, group, stat), label) in zip(axes, specs, strict=True):
        ys = [s[kind][group][stat] for s in settings.values()]
        _line(ax, xs, ys, SERIES["minfer"], "minfer")
        ax.set_title(label)
        ax.set_xlabel("prefill chunk (tokens per step)")
        ax.set_ylim(bottom=0)
        ax.margins(x=0.1)
    note = "Mixed long/short workload (bench/chunked_prefill.py); latencies of the short requests."
    _finish(
        fig, list(axes), "Chunked prefill: what it buys, what it costs", note, "chunked_prefill.png"
    )


def plot_triton_kernel() -> None:
    """Triton vs FlashInfer paged decode attention, per context length (Phase 8)."""
    path = RESULTS / "triton_kernel.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())["rows"]
    contexts = sorted({r["context_tokens"] for r in rows})
    fig, axes = plt.subplots(1, len(contexts), figsize=(3.6 * len(contexts), 4), squeeze=False)
    for ax, context in zip(axes[0], contexts, strict=True):
        subset = [r for r in rows if r["context_tokens"] == context]
        xs = [str(r["batch_size"]) for r in subset]
        micro = [
            [r[key] * 1e3 for r in subset] for key in ("flashinfer_graph_ms", "triton_graph_ms")
        ]
        _line(ax, xs, micro[0], SERIES["vllm"], "FlashInfer decode kernel")
        _line(ax, xs, micro[1], SERIES["minfer"], "minfer Triton kernel")
        ax.set_title(f"{context:,}-token contexts (µs per layer)")
        ax.set_xlabel("batch size")
        ax.set_ylim(bottom=0)
        ax.margins(x=0.14)
    note = (
        "One attention layer, 12 heads / 2 KV heads, bf16; GPU time per call when replayed "
        "from a CUDA graph (bench/triton_kernel.py)."
    )
    _finish(
        fig,
        list(axes[0]),
        "Paged decode attention: Triton vs FlashInfer",
        note,
        "triton_kernel.png",
    )


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    matrix = load_matrix()
    agent_note = (
        "Multi-turn agent sessions (bench/agent_workload.py), identical traces for every engine; "
        "greedy; 'max' = all sessions at once, no think time."
    )
    plot_workload(matrix, "agent", "Agent workload: minfer vs vLLM vs SGLang", agent_note)
    plot_agent_hit_rate(matrix)
    sharegpt_note = (
        "ShareGPT single-turn requests driven by `vllm bench serve` (control workload); greedy."
    )
    plot_workload(matrix, "sharegpt", "ShareGPT control workload", sharegpt_note)
    plot_decode_latency()
    plot_chunked_prefill()
    plot_triton_kernel()


if __name__ == "__main__":
    main()
