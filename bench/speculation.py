"""Speculative decoding: acceptance rate, speed, and output equality (Phase 7).

    .venv/bin/python -m bench.speculation

Engine-level (no HTTP). Three settings — speculation off, prompt-lookup (n-gram) drafting,
and draft-model drafting with Qwen2.5-0.5B — on two workloads:

* **agent**: sessions from ``bench/agent_workload.py``, all started at once, each turn sent
  when the previous one finishes. Unlike the server benchmark, requests here stop at EOS
  (``ignore_eos`` off) so that the model writes what it wants to write — tool calls that
  copy from the prompt — which is the behaviour prompt lookup is meant to exploit.
* **sharegpt**: the first human turn of ShareGPT conversations as single-turn chats.

For each (workload, setting): output throughput, per-request TPOT, the fraction of drafted
tokens accepted, tokens emitted per engine step, and how many greedy outputs are identical
to the speculation-off run (differences are examined with a float32 pass, as in
``bench/vllm_equivalence.py``). Writes ``bench/results/speculation.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bench import agent_workload as aw
from bench.vllm_equivalence import reference_margin
from minfer.engine import Engine, EngineConfig
from minfer.model import load_model
from minfer.sampling import SamplingParams

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
SETTINGS: dict[str, dict[str, Any]] = {
    "off": {},
    "ngram": {"speculation": "ngram"},
    "draft_model": {"speculation": "draft"},
}


def agent_requests(tokenizer: Any, sessions: int, seed: int) -> list[list[tuple[list[int], int]]]:
    """Per session, the ``(prompt ids, max_tokens)`` of each turn."""

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    trace = aw.generate(sessions, None, seed, count)
    prompts = aw.render_prompts(trace, tokenizer)
    return [
        [
            (tokenizer(text, add_special_tokens=False)["input_ids"], 2 * turn.max_tokens)
            for text, turn in zip(prompts[session.session_id], session.turns, strict=True)
        ]
        for session in trace.sessions
    ]


def sharegpt_requests(tokenizer: Any, count: int) -> list[list[tuple[list[int], int]]]:
    """Single-turn chats from ShareGPT, as one-turn "sessions"."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset")
    conversations = json.loads(Path(path).read_text())
    requests = []
    for conversation in conversations:
        turns = conversation.get("conversations") or []
        if not turns or turns[0].get("from") != "human":
            continue
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": turns[0]["value"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if 8 <= len(ids) <= 1024:
            requests.append([(ids, 256)])
        if len(requests) == count:
            break
    return requests


def run(engine: Engine, sessions: list[list[tuple[list[int], int]]]) -> dict[str, Any]:
    """Run every session closed-loop; returns outputs and timing."""
    owner: dict[int, tuple[int, int]] = {}
    first: dict[int, float] = {}
    last: dict[int, float] = {}
    outputs = [[[] for _ in turns] for turns in sessions]

    def submit(session: int, turn: int) -> None:
        prompt, max_tokens = sessions[session][turn]
        owner[engine.add_request(prompt, SamplingParams(max_tokens=max_tokens))] = (session, turn)

    stats = engine.stats
    steps, proposed, accepted = stats.steps, stats.spec_proposed, stats.spec_accepted
    for session in range(len(sessions)):
        submit(session, 0)
    start = time.perf_counter()
    while engine.has_work:
        produced = engine.step()
        now = time.perf_counter()
        for out in produced:
            session, turn = owner[out.seq_id]
            outputs[session][turn].append(out.token_id)
            first.setdefault(out.seq_id, now)
            last[out.seq_id] = now
            if out.finish_reason is not None and turn + 1 < len(sessions[session]):
                submit(session, turn + 1)
    seconds = time.perf_counter() - start
    tokens = sum(len(o) for turns in outputs for o in turns)
    counts = {seq_id: len(outputs[s][t]) for seq_id, (s, t) in owner.items()}
    tpot = [(last[i] - first[i]) / (counts[i] - 1) for i in counts if counts[i] > 1]
    proposed, accepted = stats.spec_proposed - proposed, stats.spec_accepted - accepted
    return {
        "outputs": outputs,
        "requests": len(owner),
        "generated_tokens": tokens,
        "wall_seconds": seconds,
        "output_tokens_per_second": tokens / seconds,
        "tpot_ms_p50": float(np.percentile(tpot, 50) * 1e3),
        "tpot_ms_p99": float(np.percentile(tpot, 99) * 1e3),
        "engine_steps": stats.steps - steps,
        "tokens_per_step": tokens / (stats.steps - steps),
        "draft_tokens_proposed": proposed,
        "draft_tokens_accepted": accepted,
        "acceptance_rate": accepted / proposed if proposed else None,
    }


def compare(reference: Any, sessions: list, base: list, other: list) -> dict[str, Any]:
    """How ``other`` differs from ``base``; first differences judged in float32."""
    identical = diverged = unexplained = 0
    margins = []
    for turns, a_turns, b_turns in zip(sessions, base, other, strict=True):
        for (prompt, _), a, b in zip(turns, a_turns, b_turns, strict=True):
            if a == b:
                identical += 1
                continue
            diverged += 1
            at = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
            if at is None:
                unexplained += 1  # one output is a strict prefix of the other
                continue
            margin = reference_margin(reference, prompt + a[:at], b[at], a[at])
            margins.append(margin["fp32_top2_margin"])
            unexplained += not margin["both_in_fp32_top2"]
    return {
        "identical_outputs": identical,
        "diverged_outputs": diverged,
        "divergences_not_between_fp32_top2": unexplained,
        "max_fp32_margin_at_divergence": max(margins, default=None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("bench/results/speculation.json"))
    parser.add_argument("--agent-sessions", type=int, default=32)
    parser.add_argument("--sharegpt-requests", type=int, default=256)
    parser.add_argument("--num-draft-tokens", type=int, default=4)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    workloads = {
        "agent": agent_requests(tokenizer, args.agent_sessions, seed=0),
        "sharegpt": sharegpt_requests(tokenizer, args.sharegpt_requests),
    }
    warmup = agent_requests(tokenizer, 4, seed=99)
    results: dict[str, Any] = {
        "model": MODEL,
        "draft_model": EngineConfig().draft_model,
        "num_draft_tokens": args.num_draft_tokens,
        "ngram_size": EngineConfig().ngram_size,
        "decoding": "greedy; requests stop at EOS or their token limit",
        "workloads": {name: {} for name in workloads},
    }
    reference = load_model(MODEL, dtype=torch.float32)
    baseline: dict[str, list] = {}
    for setting, overrides in SETTINGS.items():
        config = EngineConfig(
            num_blocks=40_000, num_draft_tokens=args.num_draft_tokens, **overrides
        )
        engine = Engine.from_pretrained(MODEL, config)
        run(engine, warmup)
        for name, sessions in workloads.items():
            if engine.prefix_cache is not None:
                engine.prefix_cache.clear()  # every setting starts each workload cold
            measured = run(engine, sessions)
            outputs = measured.pop("outputs")
            if setting == "off":
                baseline[name] = outputs
            else:
                measured["versus_off"] = compare(reference, sessions, baseline[name], outputs)
                off = results["workloads"][name]["off"]
                measured["throughput_vs_off"] = (
                    measured["output_tokens_per_second"] / off["output_tokens_per_second"]
                )
            results["workloads"][name][setting] = measured
            print(name, setting, json.dumps(measured))
        del engine
        torch.cuda.empty_cache()
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
