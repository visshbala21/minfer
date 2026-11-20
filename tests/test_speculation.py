"""Speculative decoding in the engine (CPU, tiny float32 model, reference backend).

Acceptance for Phase 7: greedy outputs are identical with speculation on and off. Beyond
that: rejected tokens leave no trace in the KV cache or the prefix cache, stop conditions
inside an accepted draft are honoured, and speculative *sampling* draws from the target
model's distribution.
"""

from __future__ import annotations

import random

import pytest
import torch
from test_engine import assert_all_released, make_engine
from tiny import dense_greedy, dense_logits, make_tiny_model

from minfer.sampling import SamplingParams
from minfer.spec import Draft
from minfer.spec.draft_model import DraftModelDrafter

MAX_TOKENS = 20


@pytest.fixture(scope="module")
def model():
    return make_tiny_model()


@pytest.fixture(scope="module")
def cases(model):
    rng = random.Random(11)
    prompts = [[rng.randrange(4, 199) for _ in range(rng.randint(1, 30))] for _ in range(16)]
    return prompts, [dense_greedy(model, p, MAX_TOKENS) for p in prompts]


class ScriptedDrafter:
    """Proposes the true continuation, corrupted with probability ``error_rate`` per token.

    Knows each prompt's oracle output, so it produces fully accepted drafts, drafts
    rejected at the first token, and everything in between — deterministically.
    """

    def __init__(self, prompts, outputs, error_rate: float, seed: int = 0) -> None:
        self.truth = {tuple(p): o for p, o in zip(prompts, outputs, strict=True)}
        self.error_rate = error_rate
        self.rng = random.Random(seed)
        self.released: list[int] = []

    def propose(self, seqs, limits):
        drafts = []
        for seq, limit in zip(seqs, limits, strict=True):
            future = self.truth[tuple(seq.tokens[: seq.prompt_len])][seq.num_generated :]
            tokens = [
                (t + 1) % 199 if self.rng.random() < self.error_rate else t for t in future[:limit]
            ]
            drafts.append(Draft(tokens))
        return drafts

    def release(self, seq_id):
        self.released.append(seq_id)


def spec_engine(model, drafter, **config):
    engine = make_engine(model, speculation="ngram", **config)
    engine.drafter = drafter
    return engine


@pytest.mark.parametrize("error_rate", [0.0, 0.3, 1.0], ids=["all-right", "mixed", "all-wrong"])
@pytest.mark.parametrize("num_draft", [1, 4, 7])
def test_greedy_output_is_identical_with_speculation(model, cases, error_rate, num_draft):
    prompts, expected = cases
    drafter = ScriptedDrafter(prompts, expected, error_rate)
    engine = spec_engine(model, drafter, num_draft_tokens=num_draft)
    assert engine.generate(prompts, SamplingParams(max_tokens=MAX_TOKENS)) == expected
    assert_all_released(engine)
    stats = engine.stats
    assert stats.spec_rounds > 0 and stats.spec_proposed >= stats.spec_accepted
    if error_rate == 0.0:
        assert stats.spec_accepted == stats.spec_proposed
        # Each round emits its accepted tokens plus one: far fewer steps than tokens.
        assert stats.steps < MAX_TOKENS
    if error_rate == 1.0:
        assert stats.spec_accepted == 0
    assert sorted(drafter.released) == list(range(len(prompts)))


@pytest.mark.parametrize(
    "config",
    [
        {"max_num_seqs": 1},
        {"prefill_chunk_tokens": 5, "max_batched_tokens": 9},
        {"num_blocks": 30, "watermark": 0.0},
        {"prefix_cache": False},
        {"block_size": 1},
    ],
    ids=["serial", "chunked", "memory-pressure", "no-prefix-cache", "block-1"],
)
def test_speculation_is_invariant_to_scheduling(model, cases, config):
    prompts, expected = cases
    engine = spec_engine(model, ScriptedDrafter(prompts, expected, 0.3), **config)
    assert engine.generate(prompts, SamplingParams(max_tokens=MAX_TOKENS)) == expected
    assert_all_released(engine)


def test_rejected_tokens_do_not_pollute_the_prefix_cache(model, cases):
    """After a run full of rejections, a cache-hitting rerun still matches the oracle."""
    prompts, expected = cases
    engine = spec_engine(model, ScriptedDrafter(prompts, expected, 0.5), block_size=2)
    params = SamplingParams(max_tokens=MAX_TOKENS)
    assert engine.generate(prompts, params) == expected
    assert engine.stats.spec_accepted < engine.stats.spec_proposed
    # Second pass: prompts and, block by block, the generated tokens come from the cache.
    extended = [p + o[:10] for p, o in zip(prompts, expected, strict=True)]
    oracle = [dense_greedy(model, p, 6) for p in extended]
    engine.drafter = ScriptedDrafter(extended, oracle, 0.5)
    assert engine.generate(extended, SamplingParams(max_tokens=6)) == oracle
    assert engine.prefix_cache.stats.hit_tokens > 0
    assert_all_released(engine)


def test_stop_token_inside_an_accepted_draft_ends_the_request(model, cases):
    prompts, expected = cases
    index = max(range(len(expected)), key=lambda i: len(set(expected[i])))
    prompt, full = prompts[index], expected[index]
    stop = full[6]
    cut = full.index(stop) + 1
    engine = spec_engine(model, ScriptedDrafter([prompt], [full], 0.0), num_draft_tokens=8)
    seq_id = engine.add_request(
        prompt, SamplingParams(max_tokens=MAX_TOKENS, stop_token_ids=(stop,))
    )
    outputs = []
    while engine.has_work:
        outputs += engine.step()
    assert [o.seq_id for o in outputs] == [seq_id] * cut
    assert [o.token_id for o in outputs] == full[:cut]
    assert [o.finish_reason for o in outputs] == [None] * (cut - 1) + ["stop"]
    assert_all_released(engine)


def test_draft_never_overshoots_max_tokens(model, cases):
    prompts, expected = cases
    for limit in (1, 2, 5):
        engine = spec_engine(model, ScriptedDrafter(prompts, expected, 0.0), num_draft_tokens=8)
        outputs = engine.generate(prompts, SamplingParams(max_tokens=limit))
        assert outputs == [o[:limit] for o in expected]


def test_draft_length_adapts_per_request(model, cases):
    prompts, expected = cases
    good = spec_engine(model, ScriptedDrafter(prompts[:1], expected[:1], 0.0), num_draft_tokens=8)
    good.generate(prompts[:1], SamplingParams(max_tokens=MAX_TOKENS))
    bad = spec_engine(model, ScriptedDrafter(prompts[:1], expected[:1], 1.0), num_draft_tokens=8)
    bad.generate(prompts[:1], SamplingParams(max_tokens=MAX_TOKENS))
    # Always-wrong drafts shrink to one token per round; always-right ones stay long.
    assert bad.stats.spec_proposed / bad.stats.spec_rounds < 1.5
    assert good.stats.spec_proposed / good.stats.spec_rounds > 4


def test_prompt_lookup_speculation_end_to_end(model):
    """The real n-gram drafter on prompts made of repeated material."""
    rng = random.Random(5)
    motif = [rng.randrange(4, 199) for _ in range(9)]
    prompts = [motif * rng.randint(2, 5) + motif[: rng.randint(1, 8)] for _ in range(10)]
    expected = [dense_greedy(model, p, MAX_TOKENS) for p in prompts]
    engine = make_engine(model, speculation="ngram", ngram_size=2, num_draft_tokens=4)
    assert engine.generate(prompts, SamplingParams(max_tokens=MAX_TOKENS)) == expected
    assert engine.stats.spec_rounds > 0  # the repeated motif produced drafts
    assert_all_released(engine)


class FixedDrafter:
    """Always proposes the same tokens (a deterministic, usually wrong, proposal)."""

    def __init__(self, tokens):
        self.tokens = tokens

    def propose(self, seqs, limits):
        return [Draft(self.tokens[:limit]) for limit in limits]

    def release(self, seq_id):
        pass


def test_speculative_sampling_preserves_the_target_distribution(model):
    """Temperature-1 sampling with speculation matches the exact conditional distributions.

    The first token comes from the prefill pass. The second and third are produced by one
    speculative round: the drafter proposes the likeliest second token and an arbitrary
    third one, so the round exercises acceptance, rejection with residual resampling, and
    the bonus token after a fully accepted draft.
    """
    prompt = [17, 60, 123, 8]

    def dist(tokens):
        return torch.softmax(dense_logits(model, tokens).double(), dim=-1)

    t1 = int(dist(prompt).argmax())
    second = dist([*prompt, t1])
    d1 = int(second.argmax())
    third = dist([*prompt, t1, d1])
    arbitrary = int(third.topk(4).indices[3])
    engine = spec_engine(model, FixedDrafter([d1, arbitrary]), num_blocks=9000, max_num_seqs=512)
    n = 12_000
    outs = engine.generate(
        [prompt] * n, SamplingParams(max_tokens=3, temperature=1.0, ignore_eos=True)
    )
    stats = engine.stats
    assert 0 < stats.spec_accepted < stats.spec_proposed

    runs = [o for o in outs if o[0] == t1]
    assert len(runs) > 500
    seconds = torch.tensor([o[1] for o in runs])
    for token in second.topk(4).indices.tolist():
        measured = (seconds == token).double().mean().item()
        assert abs(measured - second[token].item()) < 0.05, (token, measured, second[token])
    thirds = torch.tensor([o[2] for o in runs if o[1] == d1])
    assert len(thirds) > 100
    for token in {*third.topk(3).indices.tolist(), arbitrary}:
        measured = (thirds == token).double().mean().item()
        assert abs(measured - third[token].item()) < 0.1, (token, measured, third[token])


def test_seeded_speculative_sampling_is_reproducible(model, cases):
    prompts, expected = cases
    params = [
        SamplingParams(max_tokens=12, temperature=0.8, top_p=0.9, seed=i)
        for i in range(len(prompts))
    ]

    def run(**config):
        engine = spec_engine(model, ScriptedDrafter(prompts, expected, 0.2, seed=3), **config)
        return engine.generate(prompts, params)

    assert run() == run()


# ------------------------------------------------------------------- draft model


def draft_engine(model, draft_seed: int, draft_blocks: int = 400, **config):
    engine = make_engine(model, **config)
    engine.drafter = DraftModelDrafter(
        make_tiny_model(draft_seed),
        draft_blocks,
        block_size=engine.config.block_size,
        attention_backend="reference",
        max_model_len=engine.max_model_len,
    )
    return engine


def assert_drafter_released(drafter: DraftModelDrafter) -> None:
    blocks = drafter.blocks
    assert blocks.num_free + blocks.num_evictable == blocks.num_usable
    assert not drafter._confirmed and not drafter._tail
    drafter.cache.check_invariants()
    blocks.check_invariants(external_refs=drafter.cache.block_refs())


@pytest.mark.parametrize("draft_seed", [0, 1], ids=["draft-equals-target", "unrelated-draft"])
@pytest.mark.parametrize("num_draft", [1, 3, 6])
def test_draft_model_greedy_output_is_identical(model, cases, draft_seed, num_draft):
    prompts, expected = cases
    engine = draft_engine(model, draft_seed, num_draft_tokens=num_draft)
    assert engine.generate(prompts, SamplingParams(max_tokens=MAX_TOKENS)) == expected
    stats = engine.stats
    assert stats.spec_rounds > 0
    if draft_seed == 0:  # the draft *is* the target: everything it proposes is accepted
        assert stats.spec_accepted == stats.spec_proposed
    else:
        assert stats.spec_accepted < 0.5 * stats.spec_proposed
    assert_all_released(engine)
    assert_drafter_released(engine.drafter)


@pytest.mark.parametrize(
    "config",
    [
        {"max_num_seqs": 2},
        {"prefill_chunk_tokens": 5, "max_batched_tokens": 9},
        {"num_blocks": 30, "watermark": 0.0},
        {"draft_blocks": 12},
        {"block_size": 1, "draft_blocks": 1200},
    ],
    ids=["two-at-a-time", "chunked", "target-memory-pressure", "draft-memory-pressure", "block-1"],
)
def test_draft_model_speculation_is_invariant_to_scheduling(model, cases, config):
    prompts, expected = cases
    engine = draft_engine(model, 0, **config)
    assert engine.generate(prompts, SamplingParams(max_tokens=MAX_TOKENS)) == expected
    assert_all_released(engine)
    assert_drafter_released(engine.drafter)


def test_draft_model_reuses_shared_prefixes(model, cases):
    prompts, expected = cases
    engine = draft_engine(model, 0, block_size=2)
    params = SamplingParams(max_tokens=MAX_TOKENS)
    assert engine.generate(prompts, params) == expected
    assert engine.generate(prompts, params) == expected  # second pass: both caches are warm
    assert engine.drafter.cache.num_blocks > 0
    assert_drafter_released(engine.drafter)


def test_draft_model_speculative_sampling_preserves_the_target_distribution(model):
    """Drafts sampled from an unrelated model; outputs must still follow the target."""
    prompt = [17, 60, 123, 8]

    def dist(tokens):
        return torch.softmax(dense_logits(model, tokens).double(), dim=-1)

    t1 = int(dist(prompt).argmax())
    second = dist([*prompt, t1])
    engine = draft_engine(model, 1, draft_blocks=9000, num_blocks=9000, max_num_seqs=512)
    n = 12_000
    outs = engine.generate(
        [prompt] * n, SamplingParams(max_tokens=3, temperature=1.0, ignore_eos=True)
    )
    stats = engine.stats
    assert 0 < stats.spec_accepted < stats.spec_proposed
    runs = [o for o in outs if o[0] == t1]
    assert len(runs) > 500
    seconds = torch.tensor([o[1] for o in runs])
    for token in second.topk(5).indices.tolist():
        measured = (seconds == token).double().mean().item()
        assert abs(measured - second[token].item()) < 0.05, (token, measured, second[token])
    t2 = int(second.argmax())
    third = dist([*prompt, t1, t2])
    thirds = torch.tensor([o[2] for o in runs if o[1] == t2])
    assert len(thirds) > 60
    for token in third.topk(3).indices.tolist():
        measured = (thirds == token).double().mean().item()
        assert abs(measured - third[token].item()) < 0.1, (token, measured, third[token])
