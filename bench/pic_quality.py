"""What position-independent KV reuse costs in output quality, on real trajectories.

    .venv/bin/python -m bench.pic_quality

Reusing KV that was computed at another position, after other tokens, is an approximation:
only the first layer's keys and values are a function of the token alone. This script
measures how far the model's predictions move, as a function of how much of each reused run
is recomputed in its new context.

Setup, on the real trace (``bench/real_traces.py``): the first calls of a few sessions are
served to fill the cache; then, for each of a set of *other* sessions, the first call —
whose long system prompt sits behind a different opening paragraph than any cached one,
the shifted-duplicate case of DESIGN.md M-021 — is prefilled once by an engine without
reuse and once per recompute ratio by an engine with it. After each prefill the recorded
reply is fed token by token (teacher forcing) through the same model on top of the KV the
engine left in its cache, and the next-token distributions are compared position by
position:

* top-1 agreement with the exact prefill, and with the recorded token;
* KL divergence from the exact prefill's distribution (float32 log-softmax);
* the share of the prompt that was actually computed.

A second exact engine with a different prefill chunking gives the noise floor: how much two
*exact* bf16 evaluations of the same prompt differ. Writes
``bench/results/pic_quality_7b.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from bench import models
from bench import real_traces as rt
from minfer.attention.paged import SequenceSpan, build_paged_batch
from minfer.engine import Engine, EngineConfig
from minfer.model import load_model
from minfer.sampling import SamplingParams

RATIOS = (0.0, 0.05, 0.15, 0.3, 0.5)
NUM_BLOCKS = 12_000  # 192k tokens of KV per engine


def prefill(engine: Engine, prompt: list[int]) -> int:
    """Serve ``prompt`` for one token; return the prompt tokens the engine did not compute."""
    engine.add_request(prompt, SamplingParams(max_tokens=1))
    cached = 0
    while engine.has_work:
        for out in engine.step():
            cached = out.cached_tokens
    return cached


@torch.inference_mode()
def teacher_forced(engine: Engine, prompt: list[int], reply: list[int]) -> torch.Tensor:
    """Float32 log-probabilities of the next token at every position of ``reply``.

    Uses whatever KV the engine has cached for ``prompt`` (exact or reused) and computes
    the uncached tail of the prompt and the reply exactly on top of it.
    """
    cache, blocks, size = engine.prefix_cache, engine.blocks, engine.config.block_size
    canonical = cache.sync(prompt, [], (len(prompt) - 1) // size)
    blocks.adopt("probe", canonical)
    cached = len(canonical) * size
    tokens = [*prompt, *reply]
    blocks.ensure_capacity("probe", len(tokens))
    span = SequenceSpan(tokens[cached:], cached, blocks.table("probe"))
    batch = build_paged_batch([span], size, engine.device)
    hidden = engine.model(batch.token_ids, batch.positions, engine.backend.prefill(batch))
    first = len(prompt) - 1 - cached
    logits = engine.model.logits(hidden[first : first + len(reply)]).float()
    blocks.free("probe")
    return logits.log_softmax(dim=-1)


def compare(exact: torch.Tensor, other: torch.Tensor, reply: list[int]) -> dict[str, Any]:
    """Position-wise agreement between two sets of next-token log-probabilities."""
    recorded = torch.tensor(reply, device=exact.device)
    divergence = (exact.exp() * (exact - other)).sum(dim=-1)
    return {
        "positions": len(reply),
        "top1_same_as_exact": int((exact.argmax(-1) == other.argmax(-1)).sum()),
        "top1_is_recorded_token": int((other.argmax(-1) == recorded).sum()),
        "kl_sum": float(divergence.sum()),
        "kl_max": float(divergence.max()),
        "recorded_token_logprob_sum": float(other.gather(-1, recorded[:, None]).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=models.PRIMARY, help="model id, or 7b / 1.5b")
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument(
        "--warm-sessions", type=int, default=16, help="sessions that fill the cache"
    )
    parser.add_argument("--sessions", type=int, default=48, help="sessions measured")
    parser.add_argument("--reply-tokens", type=int, default=128, help="reply positions compared")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    model_id = models.resolve(args.model)
    tag = models.tag(model_id)
    out = args.out or models.results_path("bench/results/pic_quality.json", model_id)
    trace = rt.RealTrace.load(
        args.trace or Path(f"bench/traces/real_{tag}.jsonl"), args.warm_sessions + args.sessions
    )
    tokenizer = rt._tokenizer(model_id)

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    calls = [(session, session.calls[0]) for session in trace.sessions]
    prompts = [encode(rt.render(session, call, tokenizer)) for session, call in calls]
    replies = [
        encode(rt.completion(session, call, tokenizer))[: args.reply_tokens]
        for session, call in calls
    ]
    warm, measured = range(args.warm_sessions), range(args.warm_sessions, len(calls))

    model = load_model(model_id)
    base = {"num_blocks": NUM_BLOCKS, "prefill_chunk_tokens": None, "max_batched_tokens": 32_768}

    def run(config: EngineConfig) -> tuple[list[torch.Tensor], list[int]]:
        """Fill the cache, then prefill and probe every measured call, one after another."""
        engine = Engine(model, config)
        for index in warm:
            prefill(engine, prompts[index])
        probabilities, computed = [], []
        for index in measured:
            cached = prefill(engine, prompts[index])
            computed.append(len(prompts[index]) - cached)
            probabilities.append(teacher_forced(engine, prompts[index], replies[index]))
        del engine
        torch.cuda.empty_cache()
        return probabilities, computed

    exact, exact_computed = run(EngineConfig(**base))
    variants: dict[str, tuple[list[torch.Tensor], list[int]]] = {
        "exact, prefill in 1,024-token chunks (noise floor)": run(
            EngineConfig(**{**base, "max_batched_tokens": 1024})
        )
    }
    for ratio in RATIOS:
        config = EngineConfig(**base, position_independent_reuse=True, reuse_recompute_ratio=ratio)
        variants[f"reuse, recompute {ratio:.0%} of each run"] = run(config)

    prompt_tokens = sum(len(prompts[index]) for index in measured)
    rows = []
    for label, (probabilities, computed) in variants.items():
        per_call = [
            compare(reference, other, replies[index])
            for reference, other, index in zip(exact, probabilities, measured, strict=True)
        ]
        positions = sum(c["positions"] for c in per_call)
        row = {
            "variant": label,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_computed": sum(computed),
            "computed_fraction": sum(computed) / prompt_tokens,
            "positions": positions,
            "top1_same_as_exact": sum(c["top1_same_as_exact"] for c in per_call) / positions,
            "top1_is_recorded_token": sum(c["top1_is_recorded_token"] for c in per_call)
            / positions,
            "mean_kl": sum(c["kl_sum"] for c in per_call) / positions,
            "max_kl": max(c["kl_max"] for c in per_call),
            "mean_recorded_token_logprob": sum(c["recorded_token_logprob_sum"] for c in per_call)
            / positions,
            "calls_with_any_top1_change": sum(
                c["top1_same_as_exact"] < c["positions"] for c in per_call
            ),
        }
        rows.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))
    recorded = [
        compare(reference, reference, replies[index])
        for reference, index in zip(exact, measured, strict=True)
    ]
    positions = sum(c["positions"] for c in recorded)
    result = {
        "model": model_id,
        "trace_sha256": trace.sha256,
        "warm_sessions": args.warm_sessions,
        "measured_calls": len(measured),
        "reply_tokens_compared_per_call": args.reply_tokens,
        "exact": {
            "prompt_tokens_computed": sum(exact_computed),
            "computed_fraction": sum(exact_computed) / prompt_tokens,
            "top1_is_recorded_token": sum(c["top1_is_recorded_token"] for c in recorded)
            / positions,
            "mean_recorded_token_logprob": sum(c["recorded_token_logprob_sum"] for c in recorded)
            / positions,
        },
        "variants": rows,
    }
    out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
