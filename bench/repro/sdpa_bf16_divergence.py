"""Reproduction for DESIGN.md B-001: bf16 SDPA results that differ for identical inputs.

minfer's dense reference path and Hugging Face transformers both call
``F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)``. With both
models loaded in one process, the two call sites sometimes return different bf16 outputs
for bit-identical ``q``/``k``/``v`` at layer 0, while each call site is stable when called
again immediately. This script quantifies how often that happens.

Each trial is a fresh process that runs both models over the five model-test prompts and
hashes the layer-0 attention inputs and outputs.

Usage: python bench/repro/sdpa_bf16_divergence.py [--trials 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def child() -> None:
    """Run one trial and print a JSON record per prompt."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(ROOT / "tests"))
    from prompts import model_test_prompts

    from minfer.attention import DenseCausalAttention
    from minfer.model import load_model

    def digest(tensor: torch.Tensor) -> str:
        return hashlib.md5(tensor.float().contiguous().cpu().numpy().tobytes()).hexdigest()

    original = F.scaled_dot_product_attention
    calls: list[dict[str, str | bool]] = []

    def recording_sdpa(*args, **kwargs):  # noqa: ANN202
        out = original(*args, **kwargs)
        again = original(*args, **kwargs)
        calls.append(
            {
                "inputs": "".join(digest(a) for a in args[:3]),
                "output": digest(out),
                "stable_on_repeat": bool(torch.equal(out, again)),
            }
        )
        return out

    F.scaled_dot_product_attention = recording_sdpa

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ours = load_model(MODEL, "cuda", torch.bfloat16)
    theirs = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    records = {}
    for name, ids in model_test_prompts(tokenizer).items():
        tokens = torch.tensor(ids, device="cuda")
        with torch.inference_mode():
            calls.clear()
            ref = theirs(tokens[None]).logits[0].float()
            hf_calls = list(calls)
            calls.clear()
            hidden = ours(
                tokens, torch.arange(len(ids), device="cuda"), DenseCausalAttention(1, len(ids))
            )
            out = ours.logits(hidden).float()
            our_calls = list(calls)
        records[name] = {
            "logits_max_abs_diff": (out - ref).abs().max().item(),
            "layer0_inputs_identical": hf_calls[0]["inputs"] == our_calls[0]["inputs"],
            "layer0_outputs_identical": hf_calls[0]["output"] == our_calls[0]["output"],
            "all_calls_stable_on_repeat": all(c["stable_on_repeat"] for c in hf_calls + our_calls),
        }
    print("RESULT " + json.dumps(records))


def main() -> None:
    """Run the trials and write bench/results/repro_sdpa_bf16_divergence.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child()
        return

    trials = []
    for index in range(args.trials):
        proc = subprocess.run(
            [sys.executable, __file__, "--child"], capture_output=True, text=True, check=True
        )
        line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT "))
        record = json.loads(line.removeprefix("RESULT "))
        trials.append(record)
        diverged = {k: round(v["logits_max_abs_diff"], 3) for k, v in record.items()}
        print(f"trial {index}: logits max abs diff per prompt = {diverged}")

    flat = [r for trial in trials for r in trial.values()]
    diverged = [r for r in flat if r["logits_max_abs_diff"] > 0]
    summary = {
        "trials": args.trials,
        "prompt_runs": len(flat),
        "prompt_runs_diverged": len(diverged),
        "trials_with_any_divergence": sum(
            any(r["logits_max_abs_diff"] > 0 for r in trial.values()) for trial in trials
        ),
        "max_logit_abs_diff": max(r["logits_max_abs_diff"] for r in flat),
        "diverged_runs_with_identical_layer0_inputs": sum(
            r["layer0_inputs_identical"] for r in diverged
        ),
        "diverged_runs_with_different_layer0_outputs": sum(
            not r["layer0_outputs_identical"] for r in diverged
        ),
        "runs_unstable_on_immediate_repeat": sum(not r["all_calls_stable_on_repeat"] for r in flat),
    }
    out = ROOT / "bench/results/repro_sdpa_bf16_divergence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "trials": trials}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
