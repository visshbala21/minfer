"""Prefix cache on vs off on the real model: same outputs, less prefill (Phase 4).

    .venv/bin/python -m bench.prefix_cache_equivalence

A small multi-turn agent-style workload (shared system prompt + tool schemas per agent
type, history growing every turn) is run twice with identical prompts: once with the
prefix cache off, once with it on. Greedy outputs are compared token for token; a
difference is examined with a float32 cache-free pass exactly as in
``bench/vllm_equivalence.py`` (bf16 arithmetic depends on batch composition and on which
kernel filled a KV block, so near-ties can flip — see DESIGN.md M-002). As a noise floor,
the cache-off run is repeated with a different concurrency limit, which changes batch
composition and nothing else. Also records the hit rate and the prefill work saved.
Writes ``bench/results/prefix_cache_equivalence.json``.

This is a correctness check on a toy workload; the performance benchmark is
``bench/agent_workload.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from bench.vllm_equivalence import reference_margin
from minfer.engine import Engine, EngineConfig
from minfer.model import Qwen2Model, load_model
from minfer.sampling import SamplingParams

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
AGENTS = {
    "coder": ("read_file", "write_file", "run_tests", "search_code", "git_diff"),
    "analyst": ("sql_query", "plot_chart", "fetch_url", "summarize_table"),
}
QUESTIONS = (
    "Find where the retry logic is implemented and explain it.",
    "Now add exponential backoff with jitter to it.",
    "Which tests cover this path? Run them.",
    "Summarize what changed, in three bullet points.",
)


def system_prompt(agent: str, tools: tuple[str, ...]) -> str:
    """A system prompt with JSON tool schemas, a few hundred tokens long."""
    schemas = [
        {
            "name": tool,
            "description": f"Tool `{tool}` available to the {agent} agent. Call it with JSON "
            "arguments; the result is returned as a tool message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": f"What to apply {tool} to."},
                    "options": {"type": "object", "description": "Tool-specific options."},
                    "timeout_s": {"type": "integer", "minimum": 1, "maximum": 600},
                },
                "required": ["target"],
            },
        }
        for tool in tools
    ]
    return (
        f"You are a careful {agent} agent. Think step by step, call tools when needed, and "
        "answer concisely.\n\n# Tools\n" + json.dumps(schemas, indent=2)
    )


def run(
    engine: Engine,
    tokenizer: Any,
    num_sessions: int,
    max_tokens: int,
    script: list[list[list[int]]] | None,
) -> tuple[list[list[list[int]]], list[list[list[int]]], float]:
    """Run all sessions concurrently, each turn sent when the previous one finishes.

    With ``script`` (prompts of an earlier run) the prompts are replayed verbatim;
    otherwise each turn's prompt is built from this run's own outputs.
    """
    params = SamplingParams(max_tokens=max_tokens)
    names = list(AGENTS)
    messages = [
        [{"role": "system", "content": system_prompt(names[i % 2], AGENTS[names[i % 2]])}]
        for i in range(num_sessions)
    ]
    prompts: list[list[list[int]]] = [[] for _ in range(num_sessions)]
    outputs: list[list[list[int]]] = [[] for _ in range(num_sessions)]
    owner: dict[int, int] = {}

    def submit(session: int) -> None:
        turn = len(prompts[session])
        if script is not None:
            prompt = script[session][turn]
        else:
            messages[session].append(
                {"role": "user", "content": f"[session {session}] {QUESTIONS[turn]}"}
            )
            text = tokenizer.apply_chat_template(
                messages[session], tokenize=False, add_generation_prompt=True
            )
            prompt = tokenizer(text, add_special_tokens=False)["input_ids"]
        prompts[session].append(prompt)
        outputs[session].append([])
        owner[engine.add_request(prompt, params)] = session

    start = time.perf_counter()
    for session in range(num_sessions):
        submit(session)
    while engine.has_work:
        for out in engine.step():
            session = owner[out.seq_id]
            outputs[session][-1].append(out.token_id)
            if out.finish_reason is None:
                continue
            if script is None:
                reply = tokenizer.decode(outputs[session][-1], skip_special_tokens=True)
                messages[session].append({"role": "assistant", "content": reply})
            if len(prompts[session]) < len(QUESTIONS):
                submit(session)
    return prompts, outputs, time.perf_counter() - start


def compare(
    reference: Qwen2Model,
    prompts: list[list[list[int]]],
    off: list[list[list[int]]],
    on: list[list[list[int]]],
) -> list[dict[str, Any]]:
    """One entry per request whose outputs differ, with the float32 view of the first flip."""
    divergences = []
    for session, (session_prompts, a_turns, b_turns) in enumerate(
        zip(prompts, off, on, strict=True)
    ):
        for turn, (prompt, a, b) in enumerate(zip(session_prompts, a_turns, b_turns, strict=True)):
            if a == b:
                continue
            first = next(
                (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
                min(len(a), len(b)),
            )
            entry: dict[str, Any] = {"session": session, "turn": turn, "first_divergence": first}
            if first < min(len(a), len(b)):
                # reference_margin names its arguments for the vLLM comparison:
                # "minfer" is the cache-on token here and "vllm" the cache-off one.
                margin = reference_margin(reference, prompt + a[:first], b[first], a[first])
                entry.update(
                    cache_on_token=b[first],
                    cache_off_token=a[first],
                    fp32_top2_margin=margin["fp32_top2_margin"],
                    both_in_fp32_top2=margin["both_in_fp32_top2"],
                )
            divergences.append(entry)
    return divergences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("bench/results/prefix_cache_equivalence.json")
    )
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    off_engine = Engine.from_pretrained(MODEL, EngineConfig(num_blocks=16_384, prefix_cache=False))
    on_engine = Engine.from_pretrained(MODEL, EngineConfig(num_blocks=16_384, prefix_cache=True))
    reference = load_model(MODEL, dtype=torch.float32)

    prompts, off, off_seconds = run(off_engine, tokenizer, args.sessions, args.max_tokens, None)
    _, on, on_seconds = run(on_engine, tokenizer, args.sessions, args.max_tokens, prompts)
    divergences = compare(reference, prompts, off, on)
    # Noise floor: no cache in either run, only the batching differs.
    rebatched_engine = Engine.from_pretrained(
        MODEL, EngineConfig(num_blocks=16_384, prefix_cache=False, max_num_seqs=5)
    )
    _, rebatched, _ = run(rebatched_engine, tokenizer, args.sessions, args.max_tokens, prompts)
    floor = compare(reference, prompts, off, rebatched)
    requests = sum(map(len, prompts))
    stats = on_engine.prefix_cache.stats
    results = {
        "model": MODEL,
        "dtype": "bfloat16",
        "workload": {
            "sessions": args.sessions,
            "turns_per_session": len(QUESTIONS),
            "requests": requests,
            "prompt_tokens": sum(len(p) for s in prompts for p in s),
            "max_tokens": args.max_tokens,
        },
        "identical_outputs": requests - len(divergences),
        "diverged_outputs": len(divergences),
        "max_fp32_margin_at_divergence": max(
            (d["fp32_top2_margin"] for d in divergences if "fp32_top2_margin" in d), default=None
        ),
        "divergences": divergences,
        "noise_floor_cache_off_vs_cache_off_rebatched": {
            "what": "both runs without the cache; second run limited to 5 concurrent sequences",
            "identical_outputs": requests - len(floor),
            "diverged_outputs": len(floor),
            "max_fp32_margin_at_divergence": max(
                (d["fp32_top2_margin"] for d in floor if "fp32_top2_margin" in d), default=None
            ),
        },
        "prefix_cache": {
            "hit_rate": stats.hit_rate,
            "lookups": stats.lookups,
            "hits": stats.hits,
            "lookup_tokens": stats.lookup_tokens,
            "hit_tokens": stats.hit_tokens,
            "catchup_tokens": stats.catchup_tokens,
            "prefill_tokens_saved": stats.prefill_tokens_saved,
            "cached_blocks_at_end": on_engine.prefix_cache.num_blocks,
            "nodes_at_end": on_engine.prefix_cache.num_nodes,
        },
        "tokens_through_the_model": {
            "cache_off": off_engine.stats.prefill_tokens + off_engine.stats.decode_tokens,
            "cache_on": on_engine.stats.prefill_tokens + on_engine.stats.decode_tokens,
        },
        "wall_seconds": {"cache_off": off_seconds, "cache_on": on_seconds},
    }
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "divergences"}, indent=1))
    for d in divergences:
        print("   ", d)


if __name__ == "__main__":
    main()
