"""Phase 1 acceptance: minfer's Qwen2 forward pass matches Hugging Face transformers.

* fp32: logits agree to ``atol=1e-3`` on five diverse prompts (measured: <= 8.1e-5).
* bf16: there is no meaningful elementwise tolerance, because bf16 itself sits up to ~3
  logits away from the fp32 result at isolated positions (for transformers' own bf16 run
  just as much as for ours). The bf16 tests therefore bound the error against the fp32
  reference with aggregate metrics, and require minfer to be no further from that reference
  than transformers' bf16 run is. Thresholds come from bench/results/model_numerics.json
  with headroom; see DESIGN.md (M-001, B-001).
"""

from __future__ import annotations

import pytest
import torch
from conftest import PRIMARY_MODEL
from prompts import model_test_prompts

from minfer.attention import DenseCausalAttention
from minfer.model import ModelConfig, Qwen2Model, load_model

PROMPT_NAMES = ["english", "code", "chinese", "chat", "structured"]

# bf16-vs-fp32 budgets. Measured worst cases over the five prompts (model_numerics.json):
# mean abs 0.085, mean KL 3.5e-3 nats, top-1 agreement 0.939, max abs 3.05.
BF16_MEAN_ABS = 0.15
BF16_MEAN_KL = 1e-2
BF16_MIN_TOP1 = 0.90
BF16_MAX_ABS = 5.0


def dense_logits(model: Qwen2Model, token_ids: list[int]) -> torch.Tensor:
    """Full-sequence logits ``[T, vocab]`` through the cache-free reference path."""
    tokens = torch.tensor(token_ids, device=model.device)
    positions = torch.arange(len(token_ids), device=model.device)
    with torch.inference_mode():
        hidden = model(tokens, positions, DenseCausalAttention(1, len(token_ids)))
        return model.logits(hidden).float()


def kl_divergence(reference: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Per-position KL(reference || logits) in nats."""
    log_p = torch.log_softmax(reference.double(), dim=-1)
    log_q = torch.log_softmax(logits.double(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


# --------------------------------------------------------------------------------------
# CPU: tiny randomly initialized models, no downloads
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_tiny_random_model_matches_hf(tmp_path, tied):
    """A checkpoint written by transformers loads and reproduces its logits on CPU."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=199,
        hidden_size=64,
        intermediate_size=160,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=96,
        rope_theta=10000.0,
        tie_word_embeddings=tied,
        eos_token_id=3,
        bos_token_id=1,
    )
    torch.manual_seed(0)
    reference_model = Qwen2ForCausalLM(config).eval()
    reference_model.save_pretrained(tmp_path)

    model = load_model(tmp_path, device="cpu", dtype=torch.float32)
    assert (model.lm_head.weight is model.embed_tokens.weight) == tied

    token_ids = torch.randint(
        0, config.vocab_size, (2, 37), generator=torch.Generator().manual_seed(1)
    )
    with torch.inference_mode():
        reference = reference_model(token_ids).logits
        positions = torch.arange(37).repeat(2)
        hidden = model(token_ids.reshape(-1), positions, DenseCausalAttention(2, 37))
        logits = model.logits(hidden).view(2, 37, -1)
    torch.testing.assert_close(logits, reference, atol=1e-5, rtol=0)


def test_load_rejects_mismatched_checkpoint(tmp_path):
    """A checkpoint with the wrong set of tensors is refused, not silently half-loaded."""
    import json

    from safetensors.torch import save_file

    raw = {
        "model_type": "qwen2",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 16,
        "tie_word_embeddings": True,
        "eos_token_id": 0,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    state = Qwen2Model(ModelConfig.from_dict(raw)).state_dict()
    state.pop("lm_head.weight")
    state.pop("layers.0.mlp.up_proj.weight")
    state["layers.0.mlp.sideways_proj.weight"] = torch.zeros(1)
    save_file({f"model.{k}": v.contiguous() for k, v in state.items()}, tmp_path / "m.safetensors")
    with pytest.raises(KeyError, match="up_proj"):
        load_model(tmp_path, device="cpu")


def test_config_accepts_both_hf_layouts():
    """The shipped Qwen2.5 config layout and the current transformers layout both parse."""
    common = {
        "model_type": "qwen2",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 16,
        "eos_token_id": 0,
    }
    old = ModelConfig.from_dict({**common, "rope_theta": 1e6, "torch_dtype": "bfloat16"})
    new = ModelConfig.from_dict(
        {
            **common,
            "rope_parameters": {"rope_theta": 1e6, "rope_type": "default"},
            "dtype": "bfloat16",
        }
    )
    assert old == new
    assert old.head_dim == 4 and old.gqa_group_size == 2 and old.torch_dtype == torch.bfloat16


# --------------------------------------------------------------------------------------
# GPU: real Qwen2.5-1.5B weights against transformers
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def logits(tokenizer) -> dict[tuple[str, str, str], torch.Tensor]:
    """Logits for (implementation, dtype, prompt), computed once for the module."""
    from transformers import AutoModelForCausalLM

    prompts = model_test_prompts(tokenizer)
    out: dict[tuple[str, str, str], torch.Tensor] = {}
    for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        ours = load_model(PRIMARY_MODEL, "cuda", dtype)
        theirs = AutoModelForCausalLM.from_pretrained(PRIMARY_MODEL, dtype=dtype).cuda().eval()
        for name, token_ids in prompts.items():
            out["minfer", label, name] = dense_logits(ours, token_ids)
            with torch.inference_mode():
                tokens = torch.tensor([token_ids], device="cuda")
                out["hf", label, name] = theirs(tokens).logits[0].float()
        del ours, theirs
        torch.cuda.empty_cache()
    return out


@pytest.mark.gpu
@pytest.mark.parametrize("prompt", PROMPT_NAMES)
def test_fp32_logits_match_hf(logits, prompt):
    """Acceptance: fp32 logits agree with transformers to atol 1e-3 at every position."""
    torch.testing.assert_close(
        logits["minfer", "fp32", prompt], logits["hf", "fp32", prompt], atol=1e-3, rtol=0
    )


@pytest.mark.gpu
@pytest.mark.parametrize("prompt", PROMPT_NAMES)
def test_bf16_logits_within_documented_tolerance(logits, prompt):
    """bf16 logits stay within the documented budget of the fp32 reference."""
    reference = logits["hf", "fp32", prompt]
    ours = logits["minfer", "bf16", prompt]
    diff = (ours - reference).abs()
    assert diff.mean().item() <= BF16_MEAN_ABS
    assert diff.max().item() <= BF16_MAX_ABS
    assert kl_divergence(reference, ours).mean().item() <= BF16_MEAN_KL
    top1 = (ours.argmax(-1) == reference.argmax(-1)).float().mean().item()
    assert top1 >= BF16_MIN_TOP1


@pytest.mark.gpu
@pytest.mark.parametrize("prompt", PROMPT_NAMES)
def test_bf16_no_worse_than_hf_bf16(logits, prompt):
    """minfer's bf16 error against fp32 is not larger than transformers' own bf16 error."""
    reference = logits["hf", "fp32", prompt]
    ours = (logits["minfer", "bf16", prompt] - reference).abs().mean().item()
    theirs = (logits["hf", "bf16", prompt] - reference).abs().mean().item()
    assert ours <= 1.25 * theirs + 1e-3
