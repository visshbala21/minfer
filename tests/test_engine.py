"""The engine end to end on the CPU: a tiny float32 model and the reference backend.

The oracle is greedy generation that recomputes the whole sequence, cache-free, at every
step. Whatever the engine does to get there — batching, chunked prefill, preemption and
recompute — its greedy outputs must be the same tokens.
"""

from __future__ import annotations

import random

import pytest
import torch
from tiny import dense_greedy, dense_logits, make_tiny_model

from minfer.engine import Engine, EngineConfig
from minfer.sampling import SamplingParams


@pytest.fixture(scope="module")
def model():
    return make_tiny_model()


def make_engine(model, **config) -> Engine:
    config.setdefault("num_blocks", 400)
    config.setdefault("block_size", 4)
    eos = config.pop("eos", None)
    engine_config = EngineConfig(attention_backend="reference", **config)
    calibration = None
    if engine_config.kv_cache_dtype == "fp8":
        rng = random.Random(99)
        calibration = [[rng.randrange(4, 199) for _ in range(40)] for _ in range(4)]
    return Engine(
        model,
        engine_config,
        eos_token_ids=() if eos is None else (eos,),
        calibration_prompts=calibration,
    )


def assert_all_released(engine: Engine) -> None:
    """No sequence holds a block; whatever is still allocated belongs to the prefix cache."""
    blocks, cache = engine.blocks, engine.prefix_cache
    assert not engine.has_work
    assert blocks.num_free + blocks.num_evictable == blocks.num_usable
    if cache is None:
        assert blocks.num_free == blocks.num_usable
        blocks.check_invariants(external_refs={})
    else:
        cache.check_invariants()
        blocks.check_invariants(external_refs=cache.block_refs())
        assert cache.num_blocks == blocks.num_evictable


def make_prompts(count: int, seed: int = 0, max_len: int = 40) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(4, 199) for _ in range(rng.randint(1, max_len))] for _ in range(count)]


MAX_TOKENS = 24


@pytest.fixture(scope="module")
def oracle(model):
    """Prompts, an EOS token, and the oracle's greedy outputs.

    The random model has no meaningful EOS, so one is chosen from what it generates: the
    token that ends the largest share of sequences early while leaving others to run to
    the length limit, so both finish reasons are exercised.
    """
    prompts = make_prompts(24)
    free_running = [dense_greedy(model, p, MAX_TOKENS) for p in prompts]
    candidates = sorted({t for out in free_running for t in out})
    half = len(prompts) / 2
    eos = min(candidates, key=lambda t: abs(sum(t in out for out in free_running) - half))
    expected = [out[: out.index(eos) + 1] if eos in out else out for out in free_running]
    stopped = sum(out[-1] == eos for out in expected)
    assert 0 < stopped < len(expected)
    assert len({tuple(out) for out in expected}) > len(expected) // 2
    return prompts, eos, expected


def test_greedy_matches_dense_oracle(model, oracle):
    prompts, eos, expected = oracle
    engine = make_engine(model, eos=eos)
    assert engine.generate(prompts, SamplingParams(max_tokens=24)) == expected
    assert_all_released(engine)
    assert engine.stats.requests_finished == len(prompts)
    # Every token except each request's last sampled one went through the model exactly once.
    stats = engine.stats
    processed = stats.prefill_tokens + stats.decode_tokens
    assert processed == stats.prompt_tokens + stats.generated_tokens - len(prompts)


@pytest.mark.parametrize(
    "config",
    [
        {"max_num_seqs": 1},
        {"max_num_seqs": 5},
        {"max_batched_tokens": 7},
        {"prefill_chunk_tokens": 5},
        {"block_size": 1},
        {"block_size": 16, "num_blocks": 64},
        {"prefix_cache": False},
        {"prefix_cache": False, "max_batched_tokens": 7},
        {"unified_batches": True},
        {"unified_batches": True, "prefill_chunk_tokens": 5},
    ],
    ids=[
        "serial",
        "five-at-a-time",
        "token-budget",
        "chunked-prefill",
        "block-1",
        "block-16",
        "no-cache",
        "no-cache-token-budget",
        "unified-passes",
        "unified-passes-chunked-prefill",
    ],
)
def test_greedy_is_invariant_to_batching(model, oracle, config):
    prompts, eos, expected = oracle
    engine = make_engine(model, eos=eos, **config)
    assert engine.generate(prompts, SamplingParams(max_tokens=24)) == expected
    assert_all_released(engine)


def test_a_mixed_step_is_one_pass_when_unified(model, oracle):
    """Prefill chunks and decoding sequences of one step share a single forward pass."""
    prompts, eos, expected = oracle
    passes = {}
    for unified in (True, False):
        engine = make_engine(model, eos=eos, prefill_chunk_tokens=5, unified_batches=unified)
        counts = {"prefill": 0, "decode": 0, "mixed": 0}

        def counted(kind, run, counts=counts):
            def call(*args):
                counts[kind] += 1
                return run(*args)

            return call

        engine._run_prefill = counted("prefill", engine._run_prefill)
        engine._run_decode = counted("decode", engine._run_decode)
        engine._run_joint = counted("mixed", engine._run_joint)
        assert engine.generate(prompts, SamplingParams(max_tokens=24)) == expected
        assert_all_released(engine)
        stats = engine.stats
        processed = stats.prefill_tokens + stats.decode_tokens
        assert processed == stats.prompt_tokens + stats.generated_tokens - len(prompts)
        passes[unified] = (counts, engine.stats.steps)
    (unified, steps), (split, split_steps) = passes[True], passes[False]
    assert steps == split_steps  # the schedule is the same; only the passes differ
    assert unified["mixed"] > 0 and split["mixed"] == 0
    # Unified: one pass per step, whatever the step holds.
    assert sum(unified.values()) == steps < sum(split.values())


def test_preemption_recomputes_to_the_same_output(model, oracle):
    prompts, eos, expected = oracle
    # 24 usable blocks of 4 tokens: far less than the 24 sequences need at once.
    engine = make_engine(model, eos=eos, num_blocks=25, watermark=0.0)
    assert engine.generate(prompts, SamplingParams(max_tokens=24)) == expected
    assert engine.scheduler.num_preemptions > 0
    assert_all_released(engine)
    # Recomputed tokens show up as extra work.
    stats = engine.stats
    processed = stats.prefill_tokens + stats.decode_tokens
    assert processed > stats.prompt_tokens + stats.generated_tokens - len(prompts)


def test_requests_arriving_mid_flight(model, oracle):
    prompts, eos, expected = oracle
    engine = make_engine(model, eos=eos, max_num_seqs=6)
    outputs: dict[int, list[int]] = {}
    order = []
    pending = list(prompts)
    while pending or engine.has_work:
        if pending:
            seq_id = engine.add_request(pending.pop(0), SamplingParams(max_tokens=24))
            outputs[seq_id] = []
            order.append(seq_id)
        for out in engine.step():
            outputs[out.seq_id].append(out.token_id)
    assert [outputs[i] for i in order] == expected


def test_finish_reasons_and_stop_tokens(model, oracle):
    prompts, eos, expected = oracle
    index = next(i for i, out in enumerate(expected) if len(out) >= 6 and out[-1] != eos)
    prompt, full = prompts[index], expected[index]
    engine = make_engine(model, eos=eos)

    seq_id = engine.add_request(prompt, SamplingParams(max_tokens=4))
    outs = []
    while engine.has_work:
        outs.extend(engine.step())
    assert [o.seq_id for o in outs] == [seq_id] * 4
    assert [o.token_id for o in outs] == full[:4]
    assert [o.finish_reason for o in outs] == [None, None, None, "length"]

    stop = full[3]
    first = full.index(stop)
    (got,) = engine.generate([prompt], SamplingParams(max_tokens=24, stop_token_ids=(stop,)))
    assert got == full[: first + 1]


def test_ignore_eos_generates_past_it(model, oracle):
    prompts, eos, expected = oracle
    index = next(i for i, out in enumerate(expected) if out[-1] == eos and len(out) < 20)
    engine = make_engine(model, eos=eos)
    (got,) = engine.generate([prompts[index]], SamplingParams(max_tokens=24, ignore_eos=True))
    assert len(got) == 24 and got[: len(expected[index])] == expected[index]


def test_seeded_sampling_is_independent_of_batch_composition(model):
    prompts = make_prompts(12, seed=1)
    params = [
        SamplingParams(max_tokens=16, temperature=0.9, top_p=0.95, top_k=50, seed=100 + i)
        for i in range(len(prompts))
    ]
    together = make_engine(model).generate(prompts, params)
    serial = make_engine(model, max_num_seqs=1).generate(prompts, params)
    squeezed = make_engine(model, num_blocks=25, watermark=0.0).generate(prompts, params)
    assert together == serial == squeezed
    other_seed = make_engine(model).generate(
        prompts, [SamplingParams(max_tokens=16, temperature=0.9, seed=7)] * len(prompts)
    )
    assert other_seed != together


def test_mixed_greedy_and_sampled_batch(model, oracle):
    prompts, eos, expected = oracle
    params = [
        SamplingParams(max_tokens=24)
        if i % 2 == 0
        else SamplingParams(max_tokens=24, temperature=1.2, seed=i)
        for i in range(len(prompts))
    ]
    outputs = make_engine(model, eos=eos).generate(prompts, params)
    assert outputs[0::2] == expected[0::2]
    assert outputs[1::2] != expected[1::2]


def test_sampling_follows_the_model_distribution(model):
    """Temperature-1 samples of the first token match the oracle's softmax."""
    prompt = [11, 42, 97, 5]
    probs = torch.softmax(dense_logits(model, prompt).double(), dim=-1)
    engine = make_engine(model, num_blocks=3000, max_num_seqs=512)
    n = 4000
    outs = engine.generate(
        [prompt] * n, SamplingParams(max_tokens=1, temperature=1.0, ignore_eos=True)
    )
    counts = torch.bincount(torch.tensor([o[0] for o in outs]), minlength=probs.shape[0])
    top = probs.topk(5).indices
    assert ((counts[top].double() / n) - probs[top]).abs().max().item() < 0.03


def test_max_tokens_is_clamped_to_the_context_limit(model):
    engine = make_engine(model, max_model_len=20)
    prompt = make_prompts(1, seed=3, max_len=12)[0]
    (out,) = engine.generate([prompt], SamplingParams(max_tokens=500, ignore_eos=True))
    assert len(prompt) + len(out) == 20


def test_rejects_bad_prompts(model):
    engine = make_engine(model, max_model_len=20)
    with pytest.raises(ValueError):
        engine.add_request([])
    with pytest.raises(ValueError):
        engine.add_request(list(range(4, 24)))
    assert not engine.has_work


def test_abort_releases_blocks(model):
    engine = make_engine(model)
    prompts = make_prompts(4, seed=5)
    ids = [engine.add_request(p, SamplingParams(max_tokens=50, ignore_eos=True)) for p in prompts]
    for _ in range(3):
        engine.step()
    for seq_id in ids:
        engine.abort(seq_id)
    engine.abort(ids[0])  # idempotent
    assert_all_released(engine)
