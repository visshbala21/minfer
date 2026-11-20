"""Measure how closely minfer's Qwen2 forward pass tracks Hugging Face transformers.

For each of the five model-test prompts this records, in fp32 and bf16:
  * minfer vs transformers at the same dtype (max abs logit difference), and
  * each bf16 implementation vs the transformers fp32 reference (max/mean abs logit
    difference, top-1 agreement, KL divergence of the next-token distribution).

It also counts how many RoPE table entries differ between minfer's CPU-computed tables and
the same formula evaluated on the GPU (which is what transformers does per forward pass).

The bf16 tolerances asserted in tests/test_model.py are derived from this file's output.

Usage: python bench/model_numerics.py [--model Qwen/Qwen2.5-1.5B-Instruct]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from prompts import model_test_prompts  # noqa: E402

from minfer.attention import DenseCausalAttention  # noqa: E402
from minfer.model import ModelConfig, load_model, resolve_model_dir  # noqa: E402
from minfer.model.qwen2 import RotaryEmbedding  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def compare(logits: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Distance metrics between two ``[T, vocab]`` logit tensors."""
    diff = (logits - reference).abs()
    log_p = torch.log_softmax(reference.double(), dim=-1)
    log_q = torch.log_softmax(logits.double(), dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "top1_agreement": (logits.argmax(-1) == reference.argmax(-1)).float().mean().item(),
        "kl_mean": kl.mean().item(),
        "kl_max": kl.max().item(),
    }


def collect_logits(model_name: str, prompts: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    """Logits of both implementations at both dtypes, keyed ``impl/dtype/prompt``."""
    out: dict[str, torch.Tensor] = {}
    for label, dtype in DTYPES.items():
        ours = load_model(model_name, "cuda", dtype)
        theirs = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).cuda().eval()
        for name, ids in prompts.items():
            tokens = torch.tensor(ids, device="cuda")
            positions = torch.arange(len(ids), device="cuda")
            with torch.inference_mode():
                hidden = ours(tokens, positions, DenseCausalAttention(1, len(ids)))
                out[f"minfer/{label}/{name}"] = ours.logits(hidden).float()
                out[f"hf/{label}/{name}"] = theirs(tokens[None]).logits[0].float()
        del ours, theirs
        torch.cuda.empty_cache()
    return out


def rope_table_device_gap(model_name: str) -> dict[str, int]:
    """Count RoPE table entries that differ between CPU and GPU evaluation of cos/sin."""
    config = ModelConfig.from_pretrained(resolve_model_dir(model_name))
    head_dim, max_pos = config.head_dim, config.max_position_embeddings
    cpu = RotaryEmbedding(head_dim, config.rope_theta, max_pos)
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv_freq = (1.0 / (config.rope_theta**exponents)).cuda()
    angles = torch.arange(max_pos, device="cuda")[:, None].float() * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    out: dict[str, int] = {"entries_per_table": cpu.cos.numel()}
    for name, table, gpu in (("cos", cpu.cos, angles.cos()), ("sin", cpu.sin, angles.sin())):
        table = table.cuda()
        differs_bf16 = table.bfloat16() != gpu.bfloat16()
        out[f"{name}_fp32_entries_differing"] = int((table != gpu).sum())
        out[f"{name}_bf16_entries_differing"] = int(differs_bf16.sum())
        rows = differs_bf16.any(dim=-1).nonzero()
        out[f"{name}_bf16_first_differing_position"] = int(rows[0]) if len(rows) else -1
    return out


def main() -> None:
    """Run the comparison and write bench/results/model_numerics.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", default=str(ROOT / "bench/results/model_numerics.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = model_test_prompts(tokenizer)
    logits = collect_logits(args.model, prompts)

    rows = {}
    for name, ids in prompts.items():
        reference = logits[f"hf/fp32/{name}"]
        top2 = reference.topk(2, dim=-1).values
        rows[name] = {
            "num_tokens": len(ids),
            "fp32_minfer_vs_hf_max_abs": compare(logits[f"minfer/fp32/{name}"], reference)[
                "max_abs"
            ],
            "bf16_minfer_vs_hf_bf16_max_abs": compare(
                logits[f"minfer/bf16/{name}"], logits[f"hf/bf16/{name}"]
            )["max_abs"],
            "bf16_minfer_vs_hf_fp32": compare(logits[f"minfer/bf16/{name}"], reference),
            "bf16_hf_vs_hf_fp32": compare(logits[f"hf/bf16/{name}"], reference),
            "fp32_reference_logit_std": reference.std().item(),
            "fp32_reference_median_top1_margin": (top2[:, 0] - top2[:, 1]).median().item(),
        }
    result = {
        "model": args.model,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "prompts": rows,
        "rope_tables_cpu_vs_gpu": rope_table_device_gap(args.model),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    header = (
        f"{'prompt':11s}{'T':>4s} | {'fp32 max':>9s} | {'bf16 vs hf-bf16':>15s} | "
        f"{'minfer-bf16 vs fp32: max  mean   top1    KL':>44s} | "
        f"{'hf-bf16 vs fp32: max  mean   top1    KL':>40s}"
    )
    print(header)
    for name, row in rows.items():
        a, b = row["bf16_minfer_vs_hf_fp32"], row["bf16_hf_vs_hf_fp32"]
        print(
            f"{name:11s}{row['num_tokens']:4d} | {row['fp32_minfer_vs_hf_max_abs']:9.2e} | "
            f"{row['bf16_minfer_vs_hf_bf16_max_abs']:15.3f} | "
            f"{a['max_abs']:25.3f} {a['mean_abs']:5.3f} {a['top1_agreement']:6.4f} "
            f"{a['kl_mean']:.1e} | "
            f"{b['max_abs']:21.3f} {b['mean_abs']:5.3f} {b['top1_agreement']:6.4f} "
            f"{b['kl_mean']:.1e}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
