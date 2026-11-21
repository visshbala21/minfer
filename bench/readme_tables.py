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
import functools
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bench import models

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


def _matrix(model: str) -> dict[str, Any]:
    """Per-engine matrix results for ``model`` (a full id; see bench/models.py)."""
    directory = models.results_path("matrix", model)
    loaded = {name: _load(f"{directory}/{name}.json") for name in ENGINES}
    return {name: data for name, data in loaded.items() if data}


def _best(values: list[float | None], lower_is_better: bool) -> int | None:
    present = [(v, i) for i, v in enumerate(values) if v is not None]
    if len(present) < 2:
        return None
    return (min(present) if lower_is_better else max(present))[1]


def _matrix_table(
    workload: str, columns: list[tuple[str, Callable[[dict], Any], str, bool]], model: str
) -> str:
    """One row per (rate, engine); the best engine per rate and column is bold."""
    matrix = _matrix(model)
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


def agent_table(model: str = models.PRIMARY) -> str:
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
        model,
    )


def sharegpt_table(model: str = models.PRIMARY) -> str:
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
        model,
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
        ("cuda_graphs_on_fused", "+ fused kernels (default)"),
        ("cuda_graphs_on_triton_decode_fused", "with the Triton decode kernel instead"),
    ]
    if any(key not in runs for key, _ in stages):
        return MISSING
    rows = []
    for index, first in enumerate(runs[stages[0][0]]):
        cells = [str(first["batch_size"]), str(first["context_tokens"])]
        cells += [f"{runs[key][index]['step_ms_p50']:.2f}" for key, _ in stages]
        total = first["step_ms_p50"] / runs["cuda_graphs_on_fused"][index]["step_ms_p50"]
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
    """Triton vs FlashInfer decode attention: kernel only and end to end."""
    data = _load("triton_kernel.json")
    if not data:
        return MISSING
    rows = [
        [
            str(r["batch_size"]),
            f"{r['context_tokens']:,}",
            f"{r['flashinfer_graph_ms'] * 1e3:.1f}",
            f"{r['flashinfer_tc_graph_ms'] * 1e3:.1f}",
            f"{r['triton_graph_ms'] * 1e3:.1f}",
            f"{r['speedup_in_graph_vs_best']:.2f}×",
            f"{r['flashinfer_step_ms']:.2f}",
            f"{r['triton_step_ms']:.2f}",
            f"{r['speedup_end_to_end']:.2f}×",
            f"{r['triton_vs_flashinfer_max_abs']:.4f}",
        ]
        for r in data["rows"]
    ]
    header = ["batch", "context", "FlashInfer, plain (µs/layer)"]
    header += ["FlashInfer, tensor cores (µs/layer)", "Triton (µs/layer)"]
    header += ["kernel only: Triton vs best FlashInfer"]
    header += ["step: FlashInfer plan + 28 layers (ms)", "step: Triton prep + 28 layers (ms)"]
    header += ["end to end: Triton vs FlashInfer", "max abs difference"]
    kernel = [r["speedup_in_graph_vs_best"] for r in data["rows"]]
    step = [r["speedup_end_to_end"] for r in data["rows"]]
    faster = sum(value > 1.05 for value in kernel)
    slower = sum(value < 0.95 for value in kernel)
    summary = (
        f"Kernel only, against the faster FlashInfer configuration, the Triton kernel is "
        f"{min(kernel):.2f}–{max(kernel):.2f}× as fast: more than 5% faster at {faster} of "
        f"{len(kernel)} points, more than 5% slower at {slower}, within 5% at "
        f"{len(kernel) - faster - slower}. End to end (host preparation included) it is "
        f"{min(step):.2f}–{max(step):.2f}×."
    )
    return _table(header, rows) + "\n\n" + summary


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


def profile_table() -> str:
    """Kernels per step and GPU time, before and after the Phase 11 optimizations."""
    before, after = _load("profiles/before/summary.json"), _load("profiles/after/summary.json")
    if not before or not after:
        return MISSING
    rows = []
    for name, b in before["scenarios"].items():
        a = after["scenarios"][name]
        rows.append(
            [
                name.replace("_", " "),
                f"{b['cuda_kernels_per_step']:.0f}",
                f"{a['cuda_kernels_per_step']:.0f}",
                f"{b['gpu_kernel_ms_per_step']:.2f}",
                f"{a['gpu_kernel_ms_per_step']:.2f}",
            ]
        )
    header = ["scenario", "CUDA kernels/step before", "after", "GPU ms/step before", "after"]
    return _table(header, rows)


def equivalence_table() -> str:
    """Greedy-output comparisons: against vLLM, and prefix cache on vs off."""
    vllm, cache = _load("vllm_greedy_equivalence.json"), _load("prefix_cache_equivalence.json")
    if not vllm or not cache:
        return MISSING
    prompts = vllm["num_prompts"]
    forced = vllm["teacher_forced"]
    requests = cache["workload"]["requests"]
    floor = cache["noise_floor_cache_off_vs_cache_off_rebatched"]
    tokens = cache["tokens_through_the_model"]
    rows = [
        [
            "minfer vs vLLM, 20 prompts, one request at a time",
            f"{vllm['one_at_a_time']['identical']}/{prompts} outputs identical",
            f"{vllm['one_at_a_time']['max_fp32_margin_at_divergence']:.2f}",
        ],
        [
            "minfer (batched) vs vLLM (one at a time)",
            f"{vllm['batched_20']['identical']}/{prompts} outputs identical",
            f"{vllm['batched_20']['max_fp32_margin_at_divergence']:.2f}",
        ],
        [
            "**vLLM (batched) vs vLLM (one at a time)** — the noise floor",
            f"{vllm['vllm_one_at_a_time_vs_vllm_batched_identical']}/{prompts} outputs identical",
            "—",
        ],
        [
            "minfer fed vLLM's tokens (teacher-forced), per position",
            f"{forced['minfer_bf16_argmax_equals_vllm_token']}/{forced['positions']} tokens equal",
            f"{forced['max_fp32_margin_at_disagreement']:.2f}",
        ],
        [
            f"prefix cache on vs off, {requests} agent-style requests",
            f"{cache['identical_outputs']}/{requests} outputs identical",
            f"{cache['max_fp32_margin_at_divergence']:.2f}",
        ],
        [
            "**cache off vs cache off, different batching** — the noise floor",
            f"{floor['identical_outputs']}/{requests} outputs identical",
            f"{floor['max_fp32_margin_at_divergence']:.2f}",
        ],
    ]
    header = ["comparison (greedy, bf16)", "result", "largest float32 margin at a difference"]
    saved = (
        f"\n\nIn the prefix-cache run the cache served {cache['prefix_cache']['hit_rate']:.1%} of "
        f"prompt tokens and cut the tokens run through the model from {tokens['cache_off']:,} "
        f"to {tokens['cache_on']:,}."
    )
    return _table(header, rows) + saved


def _span(values: list[float], fmt: str) -> str:
    low, high = format(min(values), fmt), format(max(values), fmt)
    return low if low == high else f"{low}–{high}"


def headline(model: str = models.PRIMARY) -> str:
    """The comparison in a few generated sentences (agent workload)."""
    matrix = _matrix(model)
    if not all(matrix.get(e, {}).get("agent") for e in ENGINES):
        return MISSING
    steady = [r for r in matrix["minfer"]["agent"] if r != "max"]

    def over(engine: str, get: Callable[[dict], float], fmt: str) -> str:
        return _span([get(matrix[engine]["agent"][r]) for r in steady], fmt)

    def at_max(engine: str, get: Callable[[dict], float], fmt: str) -> str:
        return format(get(matrix[engine]["agent"]["max"]), fmt)

    def ttft50(r: dict) -> float:
        return r["ttft_ms"]["p50"]

    def ttft99(r: dict) -> float:
        return r["ttft_ms"]["p99"]

    def tpot50(r: dict) -> float:
        return r["tpot_ms"]["p50"]

    def hit(r: dict) -> float:
        return r["server_prefix_cache_hit_rate"]

    def tok(r: dict) -> float:
        return r["output_tokens_per_second"]

    lines = [
        f"On the multi-turn agent workload at {steady[0]}–{steady[-1]} req/s "
        f"({model.rsplit('/', 1)[-1]}, bf16, one H100, greedy, identical traces):",
        "",
        "| | minfer | vLLM | SGLang |",
        "|---|---|---|---|",
    ]
    for label, get, fmt in (
        ("TTFT p50 (ms)", ttft50, ".0f"),
        ("TTFT p99 (ms)", ttft99, ".0f"),
        ("TPOT p50 (ms)", tpot50, ".2f"),
        ("prefix-cache hit rate", hit, ".1%"),
    ):
        lines.append(f"| {label} | " + " | ".join(over(e, get, fmt) for e in ENGINES) + " |")
    lines.append(
        '| saturated output tok/s ("max") | '
        + " | ".join(at_max(e, tok, ".0f") for e in ENGINES)
        + " |"
    )

    def relative(metric: Callable[[dict], float], rates: list[str]) -> str:
        cells = []
        for engine in ENGINES:
            ratios = [
                metric(matrix[engine]["agent"][r]) / metric(matrix["vllm"]["agent"][r])
                for r in rates
            ]
            cells.append(_span(ratios, ".2f") + "×")
        return " | ".join(cells)

    lines.append(f"| TTFT p50 relative to vLLM | {relative(ttft50, steady)} |")
    lines.append(f"| TPOT p50 relative to vLLM | {relative(tpot50, steady)} |")
    lines.append(f"| saturated tok/s relative to vLLM | {relative(tok, ['max'])} |")
    lines.append(
        "| saturated TTFT p50 / p99 (ms) | "
        + " | ".join(f"{at_max(e, ttft50, '.0f')} / {at_max(e, ttft99, '.0f')}" for e in ENGINES)
        + " |"
    )
    return "\n".join(lines)


def reproduction_table() -> str:
    """Headline metrics of the README run next to an independent rerun."""
    report = _load("reproduction.json")
    if not report:
        return MISSING
    agent = report["comparison"]["agent"]
    rows = []
    for label, metric, fmt, rates in (
        ("TTFT p50, 1–16 req/s (ms)", "ttft_p50_ms", ".0f", None),
        ("TTFT p99, 1–16 req/s (ms)", "ttft_p99_ms", ".0f", None),
        ("TPOT p50, 1–16 req/s (ms)", "tpot_p50_ms", ".2f", None),
        ('TTFT p50, "max" (ms)', "ttft_p50_ms", ".0f", ["max"]),
        ('TPOT p50, "max" (ms)', "tpot_p50_ms", ".2f", ["max"]),
        ('output tok/s, "max"', "output_tokens_per_second", ".0f", ["max"]),
    ):
        for index, (engine, name) in enumerate(ENGINES.items()):
            cells = [
                agent[engine][r][metric] for r in rates or agent[engine] if rates or r != "max"
            ]
            rows.append(
                [
                    label if index == 0 else "",
                    name,
                    _span([c["reference"] for c in cells], fmt),
                    _span([c["rerun"] for c in cells], fmt),
                    _span([c["ratio"] for c in cells], ".2f") + "×",
                ]
            )
    table = _table(["agent workload", "engine", "README run", "rerun", "rerun / README run"], rows)
    held = [
        f"- {text}: {'holds in both runs' if v['reference'] and v['rerun'] else '**differs**'}"
        + (
            ""
            if v["reference"] and v["rerun"]
            else f" (README run: {v['reference']}, rerun: {v['rerun']})"
        )
        for text, v in report["claims"].items()
    ]
    return "\n".join([table, "", *held])


def fairness_audit_table() -> str:
    """The baselines under their tuning flags, and all engines with pre-tokenized prompts."""
    data = _load("fairness_audit.json")
    if not data:
        return MISSING
    rates = data["rates"]
    rows = []
    for name, entry in data["configs"].items():
        engine, variant = name.split("/")
        flags = " ".join(f"`{flag}`" for flag in entry["flags"])
        if entry.get("prompts_as_token_ids"):
            flags = "(prompts sent as token ids)"
        cells = [f"{ENGINES[engine]} / {variant.replace('_', ' ')}", flags or "—"]
        for rate in rates:
            row = entry.get(rate)
            if row is None:
                cells += ["failed to start", "", ""]
                continue
            cells += [
                f"{row['output_tokens_per_second']:.0f}" if rate == "max" else "",
                f"{row['ttft_ms']['p50']:.0f} / {row['ttft_ms']['p99']:.0f}",
                f"{row['tpot_ms']['p50']:.2f} / {row['tpot_ms']['p99']:.2f}",
            ]
        rows.append([cell for cell in cells if cell != ""])
    header = ["engine / configuration", "flags added to the defaults"]
    for rate in rates:
        prefix = "max" if rate == "max" else f"{rate} req/s"
        if rate == "max":
            header.append("max: tok/s")
        header += [f"{prefix}: TTFT p50 / p99 (ms)", f"{prefix}: TPOT p50 / p99 (ms)"]
    return _table(header, rows)


def real_trace_table() -> str:
    """Shape of the real coding-agent trace and what each kind of KV reuse could serve."""
    data = _load("real_trace_stats_7b.json")
    if not data:
        return MISSING

    def spread(key: str, fmt: str = ",.0f") -> str:
        d = data[key]
        return f"{d['p50']:{fmt}} / {d['p90']:{fmt}} / {d['max']:{fmt}}"

    pause = data["pause_seconds"]
    over = data["pause_seconds_fraction_over"]
    tools = ", ".join(f"{name} {count:,}" for name, count in data["tool_calls"].items())
    rows = [
        ["sessions (cut at the 32k-token context limit)", f"{data['sessions']} "
         f"({data['sessions_truncated_at_context_limit']})"],
        ["model calls", f"{data['model_calls']:,}"],
        ["model calls per session, p50 / p90 / max", spread("calls_per_session")],
        ["prompt tokens per call, p50 / p90 / max", spread("prompt_tokens")],
        ["tokens appended per call, p50 / p90 / max", spread("appended_tokens_per_call")],
        ["output tokens per call, p50 / p90 / max", spread("output_tokens")],
        ["tool calls", tools],
        ["pause after a call (s), p50 / p75 / p90 / p99",
         f"{pause['p50']:.2f} / {pause['p75']:.1f} / {pause['p90']:.0f} / {pause['p99']:.0f}"],
        ["pauses longer than 1 s / 10 s / 30 s",
         f"{over['1s']:.0%} / {over['10s']:.0%} / {over['30s']:.0%}"],
    ]  # fmt: skip
    for label, key in (
        ("all prompt tokens", "token_fractions"),
        ("first call of a session", "token_fractions_first_call"),
        ("later calls", "token_fractions_later_calls"),
    ):
        f = data[key]
        rows.append(
            [
                f"prefix / shifted duplicate / novel — {label}",
                f"{f['prefix']:.1%} / {f['shifted_duplicate']:.1%} / {f['novel']:.1%}",
            ]
        )
    return _table(["real trace (Qwen2.5-7B tokenizer)", "value"], rows)


def pressure_table(name: str = "pressure_7b.json") -> str:
    """KV memory pressure on the real trace: every configuration at every concurrency."""
    data = _load(name)
    if not data:
        return MISSING
    levels = sorted({int(c) for entry in data["configs"].values() for c in entry["runs"]})
    rows = []
    for level in levels:
        first = True
        for entry in data["configs"].values():
            run = entry["runs"].get(str(level))
            if run is None:
                continue
            head = [str(level) if first else "", entry["label"]]
            first = False
            if "error" in run:
                rows.append([*head, "did not start", "", "", "", "", "", ""])
                continue
            hit = run["cache_hit_rate"]
            rows.append(
                [
                    *head,
                    f"{run['completed']}",
                    "—" if hit is None else f"{hit:.1%}",
                    f"{run['prefill_tokens_computed'] / max(1, run['completed']):,.0f}",
                    f"{run['ttft_ms']['p50']:.0f} / {run['ttft_ms']['p99']:.0f}",
                    f"{run['tpot_ms']['p50']:.1f}",
                    f"{run['output_tokens_per_second']:.0f}",
                    str(run["failed"]),
                ]
            )
    header = ["concurrent sessions", "configuration", "calls completed", "cache hit rate"]
    header += ["prefill tokens computed per call", "TTFT p50 / p99 (ms)", "TPOT p50 (ms)"]
    return _table([*header, "out tok/s", "failed"], rows)


def full_stack_table() -> str:
    """Each engine in its best measured configuration under memory pressure."""
    data = _load("pressure_7b.json")
    if not data:
        return MISSING
    configs = data["configs"]
    newest = "minfer/session_host_fi"
    best = {
        "minfer": newest if newest in configs else "minfer/session_host",
        "vLLM": "vllm/offload",
        "SGLang": "sglang/hicache",
    }
    if any(name not in configs for name in best.values()):
        return MISSING
    levels = [c for c in ("64", "96", "128") if all(c in configs[n]["runs"] for n in best.values())]
    metrics: list[tuple[str, Callable[[dict], float], str, bool]] = [
        ("TTFT p50 (ms)", lambda r: r["ttft_ms"]["p50"], ",.0f", True),
        ("TTFT p99 (ms)", lambda r: r["ttft_ms"]["p99"], ",.0f", True),
        (
            "prefill tokens computed per call",
            lambda r: r["prefill_tokens_computed"] / max(1, r["completed"]),
            ",.0f",
            True,
        ),
        ("TPOT p50 (ms)", lambda r: r["tpot_ms"]["p50"], ".1f", True),
        ("output tok/s", lambda r: r["output_tokens_per_second"], ",.0f", False),
        ("calls completed in 240 s", lambda r: r["completed"], ",d", False),
        ("failed requests", lambda r: r["failed"], "d", True),
    ]
    rows = []
    for level in levels:
        for index, (label, get, fmt, lower) in enumerate(metrics):
            values = [get(configs[name]["runs"][level]) for name in best.values()]
            texts = [format(value, fmt) for value in values]
            winner = _best(values, lower)
            if winner is not None and texts.count(texts[winner]) == 1:
                texts[winner] = f"**{texts[winner]}**"
            rows.append([level if index == 0 else "", label, *texts])
    header = ["concurrent sessions", "metric"]
    header += [
        "minfer: prefix cache + session-aware retention + host tier",
        "vLLM + native KV offloading",
        "SGLang + hierarchical cache",
    ]
    footer = (
        "\n\nReal coding-agent trace, Qwen2.5-7B-Instruct, one H100, 96 GB of host memory for "
        "each engine's KV tier. No approximate technique is in any of these configurations, so "
        "output quality is that of exact bf16 inference in all three."
    )
    return _table(header, rows) + footer


def host_tier_cost_table() -> str:
    """Restoring KV from host memory vs recomputing it."""
    data = _load("host_tier_cost_7b.json")
    if not data:
        return MISSING
    rows = [
        [
            f"{r['tokens']:,}",
            f"{r['megabytes']:,.0f}",
            f"{r['restore_ms']:.1f}",
            f"{r['restore_gb_per_s']:.0f}",
            f"{r['recompute_ms']:.0f}",
            f"{r['recompute_over_restore']:.0f}×",
        ]
        for r in data["rows"]
    ]
    header = ["history (tokens)", "KV (MB)", "restore from host (ms)", "GB/s"]
    return _table([*header, "recompute (ms)", "recompute / restore"], rows)


def pic_quality_table() -> str:
    """Output quality of position-independent reuse at several recompute ratios."""
    data = _load("pic_quality_7b.json")
    if not data:
        return MISSING
    exact = data["exact"]
    rows = [
        [
            "exact prefill (reference)",
            f"{exact['computed_fraction']:.1%}",
            "—",
            f"{exact['top1_is_recorded_token']:.1%}",
            "—",
            "—",
        ]
    ]
    for r in data["variants"]:
        rows.append(
            [
                r["variant"],
                f"{r['computed_fraction']:.1%}",
                f"{r['top1_same_as_exact']:.2%}",
                f"{r['top1_is_recorded_token']:.1%}",
                f"{r['mean_kl']:.4f}",
                f"{r['max_kl']:.2f}",
            ]
        )
    header = ["prefill of a session's first call", "prompt tokens computed"]
    header += ["next-token top-1 same as exact", "top-1 = recorded token"]
    footer = (
        f"\n\n{data['measured_calls']} first calls after {data['warm_sessions']} cached "
        f"sessions; the first {data['reply_tokens_compared_per_call']} tokens of each "
        "recorded reply, teacher-forced."
    )
    return _table([*header, "mean KL from exact (nats)", "max KL"], rows) + footer


def speculation_serve_table() -> str:
    """Prompt-lookup speculation through the server at low and medium request rates."""
    data = _load("speculation_serve_7b.json")
    if not data:
        return MISSING
    rates = data["rates"]
    base = data["configs"].get("off")
    rows = []
    for entry in data["configs"].values():
        cells = [entry["label"]]
        for rate in rates:
            run = entry[rate]
            tpot = run["tpot_ms"]["p50"]
            relative = f" ({tpot / base[rate]['tpot_ms']['p50']:.2f}×)" if base else ""
            cells.append(f"{tpot:.2f}{relative}")
        for rate in rates:
            cells.append(f"{entry[rate]['e2e_ms']['p50']:.0f}")
        accepted = [entry[rate]["acceptance_rate"] for rate in rates]
        share = [entry[rate]["drafted_share_of_output"] for rate in rates]
        if accepted[0] is None:
            cells += ["—", "—"]
        else:
            cells.append(_span([a * 100 for a in accepted], ".0f") + "%")
            cells.append(_span([a * 100 for a in share], ".0f") + "%")
        rows.append(cells)
    header = ["configuration"]
    header += [f"TPOT p50 at {rate} req/s (ms)" for rate in rates]
    header += [f"E2E p50 at {rate} req/s (ms)" for rate in rates]
    return _table([*header, "drafted tokens accepted", "output tokens from drafts"], rows)


_TECHNIQUE_LABEL = {
    "prefix": "prefix cache",
    "session": "session-aware retention",
    "host": "host-memory tier",
    "reuse": "position-independent reuse",
    "spec": "prompt-lookup speculation",
}


def _composition_name(techniques: list[str]) -> str:
    order = [t for t in _TECHNIQUE_LABEL if t in techniques]
    return " + ".join(_TECHNIQUE_LABEL[t] for t in order) or "none"


def composition_table() -> str:
    """Every valid combination of techniques on the real trace under memory pressure."""
    data = _load("composition_7b.json")
    if not data:
        return MISSING
    rows = []
    for row in data["configs"].values():
        calls = max(1, row["completed"])
        rows.append(
            [
                _composition_name(row["techniques"]),
                f"{row['completed']}",
                f"{row['prefill_tokens_computed'] / calls:,.0f}",
                f"{row['cache_hit_rate']:.1%}",
                f"{row['ttft_ms']['p50']:.0f} / {row['ttft_ms']['p99']:.0f}",
                f"{row['tpot_ms']['p50']:.1f}",
                f"{row['output_tokens_per_second']:.0f}",
            ]
        )
    header = ["techniques", "calls completed", "prefill tokens computed per call"]
    header += ["prompt tokens not computed", "TTFT p50 / p99 (ms)", "TPOT p50 (ms)", "out tok/s"]
    budget = data.get("kv_budget_gb")
    footer = (
        f"\n\n{data['concurrency']} concurrent sessions of the real trace, "
        f"{data['duration_s'] - data['warmup_s']:.0f} s measured after "
        f"{data['warmup_s']:.0f} s of warm-up"
        + (f", KV cache limited to {budget:g} GB." if budget else ", whole GPU.")
    )
    return _table(header, rows) + footer


def interaction_table() -> str:
    """Pairwise interactions: is the effect of two techniques together the sum of each alone?"""
    data = _load("composition_7b.json")
    if not data:
        return MISSING
    configs = data["configs"]

    def metric(names: tuple[str, ...], key: str) -> float | None:
        name = "+".join(t for t in _TECHNIQUE_LABEL if t in names) or "base"
        row = configs.get(name)
        if row is None:
            return None
        if key == "prefill":
            return row["prefill_tokens_computed"] / max(1, row["completed"])
        if key == "ttft":
            return row["ttft_ms"]["p50"]
        return row["tpot_ms"]["p50"]

    pairs = [
        (("prefix",), "session", "reuse"),
        (("prefix",), "session", "spec"),
        (("prefix",), "reuse", "spec"),
        (("prefix", "session"), "host", "reuse"),
        (("prefix", "session"), "host", "spec"),
    ]
    rows = []
    for context, a, b in pairs:
        for key, label, fmt in (
            ("prefill", "prefill tokens / call", ",.0f"),
            ("ttft", "TTFT p50 (ms)", ".0f"),
            ("tpot", "TPOT p50 (ms)", ".2f"),
        ):
            values = [
                metric(context, key),
                metric((*context, a), key),
                metric((*context, b), key),
                metric((*context, a, b), key),
            ]
            if any(v is None for v in values):
                continue
            base, with_a, with_b, both = values
            additive = with_a + with_b - base
            rows.append(
                [
                    f"{_TECHNIQUE_LABEL[a]} × {_TECHNIQUE_LABEL[b]}",
                    label,
                    format(base, fmt),
                    f"{with_a - base:+{fmt}}",
                    f"{with_b - base:+{fmt}}",
                    f"{both - base:+{fmt}}",
                    f"{both - additive:+{fmt}}",
                ]
            )
    header = ["pair (on top of what both need)", "metric", "without either", "first alone"]
    return _table([*header, "second alone", "both", "both − sum of each"], rows)


SECTIONS: dict[str, Callable[[], str]] = {
    "headline": headline,
    "agent-results": agent_table,
    "sharegpt-results": sharegpt_table,
    "headline-1.5b": functools.partial(headline, models.SECONDARY),
    "agent-results-1.5b": functools.partial(agent_table, models.SECONDARY),
    "sharegpt-results-1.5b": functools.partial(sharegpt_table, models.SECONDARY),
    "decode-latency": decode_latency_table,
    "prefill-latency": prefill_latency_table,
    "chunked-prefill": chunked_prefill_table,
    "ablations": ablation_table,
    "speculation": speculation_table,
    "fp8": fp8_table,
    "triton": triton_table,
    "fairness-audit": fairness_audit_table,
    "full-stack": full_stack_table,
    "real-trace": real_trace_table,
    "pressure": pressure_table,
    "host-tier-cost": host_tier_cost_table,
    "pic-quality": pic_quality_table,
    "speculation-serve": speculation_serve_table,
    "composition": composition_table,
    "interactions": interaction_table,
    "equivalence": equivalence_table,
    "profile": profile_table,
    "reproduction": reproduction_table,
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
