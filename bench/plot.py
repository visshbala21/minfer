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

from bench import models

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


def load_matrix(model: str) -> dict[str, dict[str, Any]]:
    """Per-engine results of ``bench/run_matrix.py`` for ``model``, in the fixed engine order."""
    found = {}
    for engine in SERIES:
        path = models.results_path(RESULTS / "matrix", model) / f"{engine}.json"
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


def plot_workload(
    matrix: dict[str, dict[str, Any]], workload: str, title: str, note: str, suffix: str
) -> None:
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
    _finish(fig, axes, title, note, f"{workload}_vs_rate{suffix}.png")


def plot_agent_hit_rate(matrix: dict[str, dict[str, Any]], suffix: str) -> None:
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
    _finish(
        fig, list(axes), "Agent workload: prefix reuse", note, f"agent_prefix_cache{suffix}.png"
    )


def plot_decode_latency() -> None:
    """Decode-step latency as optimizations are added (Phases 3, 8, 11)."""
    path = RESULTS / "decode_latency.json"
    if not path.exists():
        return
    runs = json.loads(path.read_text())["runs"]
    stages = [
        ("cuda_graphs_off", "eager", MUTED),
        ("cuda_graphs_on", "+ CUDA graphs", SERIES["sglang"]),
        ("cuda_graphs_on_fused", "+ fused kernels (default)", SERIES["minfer"]),
        ("cuda_graphs_on_triton_decode_fused", "Triton decode kernel instead", SERIES["vllm"]),
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
        keys = ("flashinfer_graph_ms", "flashinfer_tc_graph_ms", "triton_graph_ms")
        micro = [[r[key] * 1e3 for r in subset] for key in keys]
        _line(ax, xs, micro[0], SERIES["sglang"], "FlashInfer decode, plain")
        _line(ax, xs, micro[1], SERIES["vllm"], "FlashInfer decode, tensor cores")
        _line(ax, xs, micro[2], SERIES["minfer"], "minfer Triton kernel")
        ax.set_title(f"{context:,}-token contexts (µs per layer)")
        ax.set_xlabel("batch size")
        ax.set_ylim(bottom=0)
        ax.margins(x=0.14)
    note = (
        "One attention layer, 12 heads / 2 KV heads, bf16; kernel-only GPU time per call when "
        "replayed from a CUDA graph, no host-side planning (bench/triton_kernel.py)."
    )
    _finish(
        fig,
        list(axes[0]),
        "Paged decode attention: Triton vs FlashInfer",
        note,
        "triton_kernel.png",
    )


PRESSURE_STYLE = {
    "minfer/lru": ("#9ec5f0", "minfer, LRU"),
    "minfer/session": ("#2a78d6", "minfer, session-aware"),
    "minfer/session_host": ("#0b3d7a", "minfer, session-aware + host tier"),
    "vllm/default": ("#eb6834", "vLLM"),
    "vllm/offload": ("#8f3410", "vLLM + KV offloading"),
    "sglang/default": ("#1baf7a", "SGLang"),
    "sglang/hicache": ("#0c6b49", "SGLang + hierarchical cache"),
}


def plot_pressure(name: str = "pressure_7b") -> None:
    """Hit rate, TTFT and throughput against concurrent sessions on the real trace."""
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    panels = [
        ("cache_hit_rate", "Prompt tokens served from cache"),
        ("ttft_ms.p50", "TTFT p50 (ms)"),
        ("ttft_ms.p99", "TTFT p99 (ms)"),
        ("output_tokens_per_second", "Output throughput (tok/s)"),
    ]
    levels = sorted({int(c) for entry in data["configs"].values() for c in entry["runs"]})
    position = {level: index for index, level in enumerate(levels)}
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2))
    ring = {"markeredgecolor": SURFACE, "markeredgewidth": 1.5}
    for ax, (metric, label) in zip(axes, panels, strict=True):
        for config in sorted(data["configs"], key=list(PRESSURE_STYLE).index):
            entry = data["configs"][config]
            runs = {int(c): r for c, r in entry["runs"].items() if "error" not in r}
            xs = sorted(runs)
            ys = [_agent_value(runs[x], metric) for x in xs]
            color, text = PRESSURE_STYLE.get(config, (MUTED, entry["label"]))
            ax.plot([position[x] for x in xs], ys, color=color, marker="o", label=text, **ring)
        ax.set_title(label)
        ax.set_xlabel("concurrent sessions")
        ax.set_xticks(range(len(levels)), [str(level) for level in levels])
        if metric.startswith("ttft"):
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.margins(x=0.08)
    budget = data.get("kv_budget_gb")
    note = (
        "Real coding-agent trace (Open-SWE-Traces content, TraceLab tool-call pauses), "
        f"{data['model'].rsplit('/', 1)[-1]}, closed loop; TTFT on a log scale; "
        + (f"KV cache limited to {budget:g} GB for every engine." if budget else "whole GPU.")
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(
        "KV memory pressure: sessions past GPU capacity",
        x=0.01, ha="left", fontsize=13, fontweight="bold",
    )  # fmt: skip
    fig.text(0.01, 0.005, note, color=MUTED, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.8))
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / f"{name}.png", dpi=160)
    plt.close(fig)
    print(f"wrote {PLOTS / f'{name}.png'}")


COMPOSITION_PATH = [
    ("base", "none"),
    ("prefix", "+ prefix\ncache"),
    ("prefix+session", "+ session\nretention"),
    ("prefix+session+host", "+ host\ntier"),
    ("prefix+session+host+reuse", "+ shifted\nreuse"),
    ("prefix+session+host+reuse+spec", "+ specu-\nlation"),
]


def plot_composition(name: str = "composition_7b") -> None:
    """Decomposition: the techniques added one at a time, on three measures."""
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    configs = data["configs"]
    steps = [(key, label) for key, label in COMPOSITION_PATH if key in configs]
    if len(steps) < 2:
        return
    panels = [
        (
            "Prefill tokens computed per call",
            lambda r: r["prefill_tokens_computed"] / r["completed"],
        ),
        ("TTFT p50 (ms)", lambda r: r["ttft_ms"]["p50"]),
        ("TTFT p99 (ms)", lambda r: r["ttft_ms"]["p99"]),
        ("TPOT p50 (ms)", lambda r: r["tpot_ms"]["p50"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    labels = [label for _, label in steps]
    for ax, (title, get) in zip(axes, panels, strict=True):
        values = [get(configs[key]) for key, _ in steps]
        bars = ax.bar(range(len(values)), values, color=SERIES["minfer"], width=0.62)
        bars[0].set_color(MUTED)
        for index, value in enumerate(values):
            text = f"{value:,.0f}" if value >= 100 else f"{value:.1f}"
            ax.annotate(text, (index, value), ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(values)), labels, fontsize=8.5)
        ax.set_axisbelow(True)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    budget = data.get("kv_budget_gb")
    note = (
        f"Real coding-agent trace, {data['model'].rsplit('/', 1)[-1]}, {data['concurrency']} "
        "concurrent sessions; techniques added left to right (prefix cache, session-aware "
        "retention, host-memory tier, position-independent reuse, prompt-lookup speculation); "
        + (f"KV cache limited to {budget:g} GB." if budget else "whole GPU.")
    )
    _finish(fig, list(axes), "What each technique adds", note, f"{name}.png")


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    # The 7B model's charts carry the plain file names; the 1.5B model's a suffix.
    for model, suffix in ((models.PRIMARY, ""), (models.SECONDARY, "_1.5b")):
        matrix = load_matrix(model)
        name = model.rsplit("/", 1)[-1]
        agent_note = (
            f"{name}. Multi-turn agent sessions (bench/agent_workload.py), identical traces "
            "for every engine; greedy; 'max' = all sessions at once, no think time."
        )
        title = f"Agent workload, {name}: minfer vs vLLM vs SGLang"
        plot_workload(matrix, "agent", title, agent_note, suffix)
        plot_agent_hit_rate(matrix, suffix)
        sharegpt_note = (
            f"{name}. ShareGPT single-turn requests driven by `vllm bench serve` "
            "(control workload); greedy."
        )
        title = f"ShareGPT control workload, {name}"
        plot_workload(matrix, "sharegpt", title, sharegpt_note, suffix)
    plot_decode_latency()
    plot_chunked_prefill()
    plot_triton_kernel()
    plot_pressure()
    plot_composition()


if __name__ == "__main__":
    main()
