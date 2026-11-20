"""Phase 2 acceptance on the GPU: the real model through FlashInfer paged attention.

* 64 concurrent requests complete (with ample memory, and under memory pressure).
* Greedy outputs are compared token-for-token with vLLM on a 20-prompt fixture
  (``tests/fixtures/vllm_greedy.json``, produced by ``bench/vllm_greedy_fixture.py``).
  bf16 inference is not bit-reproducible across engines — or across batch compositions of
  one engine: vLLM's own serial and batched outputs in the fixture differ for about half
  of the prompts. The assertion is therefore that every divergence is a float32 near-tie
  between the float32 model's two best tokens. See DESIGN.md M-002.
"""

from __future__ import annotations

import json
import random

import pytest
import torch
from conftest import PRIMARY_MODEL, TESTS_DIR

from bench.vllm_equivalence import compare, teacher_forced
from minfer.engine import Engine, EngineConfig
from minfer.model import load_model
from minfer.sampling import SamplingParams

pytestmark = pytest.mark.gpu

# Largest float32 logit gap between two tokens that bf16 arithmetic is allowed to flip.
# Measured worst case on the fixture is 0.33 (bench/results/vllm_greedy_equivalence.json);
# bf16 logits are individually up to ~1.2 away from float32 on the same data.
NEAR_TIE_MARGIN = 0.5


@pytest.fixture(scope="module")
def engine():
    return Engine.from_pretrained(PRIMARY_MODEL, EngineConfig(num_blocks=8192))


@pytest.fixture(scope="module")
def reference():
    return load_model(PRIMARY_MODEL, dtype=torch.float32)


@pytest.fixture(scope="module")
def fixture():
    return json.loads((TESTS_DIR / "fixtures" / "vllm_greedy.json").read_text())


def chat_prompts(tokenizer, count: int) -> list[list[int]]:
    topics = ["rivers", "prime numbers", "volcanoes", "TCP", "jazz", "enzymes", "chess", "tides"]
    prompts = []
    for i in range(count):
        text = f"Request {i}: write {1 + i % 4} sentences about {topics[i % len(topics)]}."
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        prompts.append(tokenizer(rendered, add_special_tokens=False)["input_ids"])
    return prompts


def mixed_params(count: int) -> list[SamplingParams]:
    rng = random.Random(0)
    params = []
    for i in range(count):
        max_tokens = rng.randint(8, 96)
        if i % 3 == 0:
            params.append(SamplingParams(max_tokens=max_tokens))
        else:
            params.append(
                SamplingParams(max_tokens=max_tokens, temperature=0.8, top_p=0.95, top_k=40, seed=i)
            )
    return params


def check_outputs(engine: Engine, outputs: list[list[int]], params: list[SamplingParams]) -> None:
    vocab = engine.model.config.vocab_size
    for out, p in zip(outputs, params, strict=True):
        assert 1 <= len(out) <= p.max_tokens
        assert all(0 <= t < vocab for t in out)
        assert len(out) == p.max_tokens or out[-1] in engine.eos_token_ids
        assert not any(t in engine.eos_token_ids for t in out[:-1])
    assert not engine.has_work
    assert engine.blocks.num_free == engine.blocks.num_usable
    engine.blocks.check_invariants(external_refs={})


def test_64_concurrent_requests_complete(engine, tokenizer):
    prompts = chat_prompts(tokenizer, 64)
    params = mixed_params(64)
    finished_before = engine.stats.requests_finished
    outputs = engine.generate(prompts, params)
    check_outputs(engine, outputs, params)
    assert engine.stats.requests_finished - finished_before == 64
    assert engine.scheduler.num_preemptions == 0
    # The model produces language, not noise: outputs decode to mostly-ASCII prose.
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert len(text.split()) >= 3 and text.isascii()


def test_64_concurrent_requests_complete_under_memory_pressure(tokenizer):
    # 200 blocks of 16 tokens hold 3,200 tokens; the 64 requests need about 5,500 at peak.
    small = Engine.from_pretrained(PRIMARY_MODEL, EngineConfig(num_blocks=201))
    prompts = chat_prompts(tokenizer, 64)
    params = mixed_params(64)
    outputs = small.generate(prompts, params)
    check_outputs(small, outputs, params)
    assert small.scheduler.num_preemptions > 0


def test_chunked_prefill_and_serial_execution_complete(tokenizer):
    tight = Engine.from_pretrained(
        PRIMARY_MODEL, EngineConfig(num_blocks=2048, max_batched_tokens=48, max_num_seqs=3)
    )
    prompts = chat_prompts(tokenizer, 12)
    params = mixed_params(12)
    check_outputs(tight, tight.generate(prompts, params), params)


def test_engine_logits_track_the_float32_reference(engine, reference, fixture):
    """FlashInfer prefill + decode, fed vLLM's tokens, against cache-free float32."""
    report = teacher_forced(engine, reference, fixture)
    positions = report["positions"]
    assert positions >= 1500
    assert report["minfer_bf16_vs_fp32_mean_abs_logit_error"] < 0.15
    assert report["minfer_bf16_argmax_equals_fp32_argmax"] / positions > 0.97
    assert report["minfer_bf16_argmax_equals_vllm_token"] / positions > 0.97
    for d in report["disagreements"]:
        assert d["both_in_fp32_top2"], d
        assert d["fp32_margin_between_them"] < NEAR_TIE_MARGIN, d


@pytest.mark.parametrize("batched", [False, True], ids=["one-at-a-time", "batched"])
def test_greedy_matches_vllm_up_to_float32_near_ties(engine, reference, fixture, batched):
    assert len(fixture["cases"]) == 20
    report = compare(engine, reference, fixture, batched)
    for case in report:
        if case["identical"]:
            continue
        # Never a length-only difference: there is always a first differing token...
        assert "minfer_token" in case, case
        # ...and at that position both engines chose among float32's two best tokens,
        # which float32 itself separates by less than bf16 can resolve.
        assert case["both_in_fp32_top2"], case
        assert case["fp32_top2_margin"] < NEAR_TIE_MARGIN, case
    assert any(case["identical"] for case in report)
