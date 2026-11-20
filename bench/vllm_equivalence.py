"""Compare minfer's greedy outputs with the vLLM fixture, token for token.

    .venv/bin/python -m bench.vllm_equivalence

Three views of the same question, written to ``bench/results/vllm_greedy_equivalence.json``:

* **Free-running.** minfer generates from each prompt (one request at a time, and all 20
  batched). For every output that differs from vLLM's, the first differing position is
  examined with a float32 cache-free forward pass over the common prefix: the divergence is
  *explained* when the two engines picked the float32 model's two best tokens and those
  are a near-tie, i.e. the bf16 engines disagree about a coin flip.
* **vLLM against itself.** The fixture holds vLLM's outputs for the same prompts run one at
  a time and batched; their agreement is the noise floor for any bf16 engine.
* **Teacher-forced.** Free-running comparison stops being informative after the first
  flip. Here minfer's prefill + decode path is fed vLLM's own tokens, so every position is
  compared: does minfer's bf16 argmax equal vLLM's token, and how far are minfer's bf16
  logits from the float32 reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from minfer.attention import DenseCausalAttention, SequenceSpan, build_paged_batch
from minfer.engine import Engine, EngineConfig
from minfer.model import Qwen2Model, load_model
from minfer.sampling import SamplingParams

FIXTURE = Path("tests/fixtures/vllm_greedy.json")
RESULTS = Path("bench/results/vllm_greedy_equivalence.json")


@torch.inference_mode()
def reference_margin(model: Qwen2Model, tokens: list[int], a: int, b: int) -> dict[str, Any]:
    """Float32 view of a disagreement between tokens ``a`` and ``b`` after ``tokens``."""
    ids = torch.tensor(tokens, device=model.device)
    positions = torch.arange(len(tokens), device=model.device)
    hidden = model(ids, positions, DenseCausalAttention(1, len(tokens)))
    logits = model.logits(hidden[-1:])[0].double()
    top = logits.topk(2)
    return {
        "fp32_top2_tokens": top.indices.tolist(),
        "fp32_top2_margin": float(top.values[0] - top.values[1]),
        "fp32_logit_minfer_token": float(logits[a]),
        "fp32_logit_vllm_token": float(logits[b]),
        "both_in_fp32_top2": {a, b} == set(top.indices.tolist()),
    }


def compare(
    engine: Engine, reference: Qwen2Model, fixture: dict[str, Any], batched: bool
) -> list[dict[str, Any]]:
    """Run the fixture's prompts through ``engine`` and describe every case."""
    cases = fixture["cases"]
    prompts = [case["prompt_token_ids"] for case in cases]
    params = SamplingParams(max_tokens=fixture["max_tokens"])
    if batched:
        outputs = engine.generate(prompts, params)
    else:
        outputs = [engine.generate([prompt], params)[0] for prompt in prompts]
    report = []
    for case, prompt, ours in zip(cases, prompts, outputs, strict=True):
        theirs = case["output_token_ids"]
        entry: dict[str, Any] = {
            "prompt": case["prompt"],
            "vllm_tokens": len(theirs),
            "minfer_tokens": len(ours),
            "identical": ours == theirs,
        }
        if ours != theirs:
            first = next(
                (i for i, (x, y) in enumerate(zip(ours, theirs, strict=False)) if x != y),
                min(len(ours), len(theirs)),
            )
            entry["first_divergence"] = first
            if first < min(len(ours), len(theirs)):
                entry["minfer_token"] = ours[first]
                entry["vllm_token"] = theirs[first]
                entry.update(
                    reference_margin(reference, prompt + ours[:first], ours[first], theirs[first])
                )
        report.append(entry)
    return report


@torch.inference_mode()
def engine_path_logits(engine: Engine, tokens: list[int], prompt_len: int) -> torch.Tensor:
    """Next-token logits at positions ``prompt_len - 1 ..`` via the engine's own kernels.

    The prompt goes through one prefill pass and each later token through a decode pass,
    exactly as in generation, except that the tokens are given rather than sampled.
    """
    blocks, size, model = engine.blocks, engine.config.block_size, engine.model
    seq_id = "teacher-forced"
    blocks.ensure_capacity(seq_id, len(tokens))
    table = blocks.table(seq_id)
    batch = build_paged_batch([SequenceSpan(tokens[:prompt_len], 0, table)], size, engine.device)
    hidden = model(batch.token_ids, batch.positions, engine.backend.prefill(batch))
    rows = [model.logits(hidden[-1:])]
    for position in range(prompt_len, len(tokens)):
        span = SequenceSpan(tokens[position : position + 1], position, table)
        batch = build_paged_batch([span], size, engine.device)
        hidden = model(batch.token_ids, batch.positions, engine.backend.decode(batch))
        rows.append(model.logits(hidden))
    blocks.free(seq_id)
    return torch.cat(rows).float()


@torch.inference_mode()
def teacher_forced(engine: Engine, reference: Qwen2Model, fixture: dict[str, Any]) -> dict:
    """Per-position agreement with vLLM's tokens when minfer is fed vLLM's sequence."""
    positions = agree = fp32_agree = ours_fp32_agree = 0
    abs_error_sum = 0.0
    abs_error_max = 0.0
    disagreements = []
    for index, case in enumerate(fixture["cases"]):
        prompt, output = case["prompt_token_ids"], case["output_token_ids"]
        tokens = prompt + output[:-1]
        ours = engine_path_logits(engine, tokens, len(prompt))
        ids = torch.tensor(tokens, device=reference.device)
        hidden = reference(
            ids, torch.arange(len(tokens), device=ids.device), DenseCausalAttention(1, len(tokens))
        )
        exact = reference.logits(hidden[len(prompt) - 1 :])
        target = torch.tensor(output, device=ids.device)
        error = (ours - exact).abs()
        abs_error_sum += error.mean(dim=-1).sum().item()
        abs_error_max = max(abs_error_max, error.max().item())
        positions += len(output)
        fp32_agree += int((exact.argmax(-1) == target).sum())
        picked = ours.argmax(-1)
        agree += int((picked == target).sum())
        ours_fp32_agree += int((picked == exact.argmax(-1)).sum())
        for step in (picked != target).nonzero().flatten().tolist():
            a, b = int(picked[step]), output[step]
            top = exact[step].topk(2)
            disagreements.append(
                {
                    "case": index,
                    "position": step,
                    "minfer_token": a,
                    "vllm_token": b,
                    "fp32_margin_between_them": abs(float(exact[step, a] - exact[step, b])),
                    "both_in_fp32_top2": {a, b} == set(top.indices.tolist()),
                    "fp32_prefers": "minfer" if exact[step, a] > exact[step, b] else "vllm",
                }
            )
    return {
        "positions": positions,
        "minfer_bf16_argmax_equals_vllm_token": agree,
        "fp32_reference_argmax_equals_vllm_token": fp32_agree,
        "minfer_bf16_argmax_equals_fp32_argmax": ours_fp32_agree,
        "minfer_bf16_vs_fp32_mean_abs_logit_error": abs_error_sum / positions,
        "minfer_bf16_vs_fp32_max_abs_logit_error": abs_error_max,
        "max_fp32_margin_at_disagreement": max(
            (d["fp32_margin_between_them"] for d in disagreements), default=None
        ),
        "disagreements": disagreements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text())
    engine = Engine.from_pretrained(fixture["model"], EngineConfig(num_blocks=8192))
    reference = load_model(fixture["model"], dtype=torch.float32)
    results: dict[str, Any] = {
        "model": fixture["model"],
        "vllm_version": fixture["vllm_version"],
        "dtype": "bfloat16",
        "max_tokens": fixture["max_tokens"],
        "num_prompts": len(fixture["cases"]),
    }
    for mode, batched in (("one_at_a_time", False), ("batched_20", True)):
        report = compare(engine, reference, fixture, batched)
        diverged = [r for r in report if not r["identical"]]
        results[mode] = {
            "identical": len(report) - len(diverged),
            "diverged": len(diverged),
            "tokens_compared": sum(r["vllm_tokens"] for r in report),
            "max_fp32_margin_at_divergence": max(
                (r["fp32_top2_margin"] for r in diverged if "fp32_top2_margin" in r), default=None
            ),
            "cases": report,
        }
        print(f"{mode}: {len(report) - len(diverged)}/{len(report)} identical")
    serial_vs_batched = sum(
        case["output_token_ids"] == case["output_token_ids_batched"] for case in fixture["cases"]
    )
    results["vllm_one_at_a_time_vs_vllm_batched_identical"] = serial_vs_batched
    print(f"vLLM vs itself (serial vs batched): {serial_vs_batched}/{len(fixture['cases'])}")
    forced = teacher_forced(engine, reference, fixture)
    results["teacher_forced"] = forced
    print("teacher-forced:", {k: v for k, v in forced.items() if k != "disagreements"})
    for d in forced["disagreements"]:
        print("   ", d)
    args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
