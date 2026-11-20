"""The prefix cache inside the engine (CPU, tiny float32 model, reference backend).

Acceptance for Phase 4: outputs with the cache on are identical to outputs with it off (and
to the cache-free oracle), across multi-turn sessions that share prefixes, concurrent
requests that share prefixes, eviction, and preemption; and the hit rate is reported.
"""

from __future__ import annotations

import random

import pytest
from test_engine import assert_all_released, make_engine
from tiny import dense_greedy, make_tiny_model

from minfer.sampling import SamplingParams

MAX_TOKENS = 12


@pytest.fixture(scope="module")
def model():
    return make_tiny_model()


@pytest.fixture(scope="module")
def sessions(model):
    """Agent-like sessions: a shared system prompt per agent type, history growing per turn.

    Returns ``turns[session][turn] = (prompt, expected_output)`` where each prompt is the
    previous prompt + the previous (oracle) output + new "user" tokens.
    """
    rng = random.Random(7)

    def fresh(count: int) -> list[int]:
        return [rng.randrange(4, 199) for _ in range(count)]

    systems = [fresh(37), fresh(52)]
    turns = []
    for session in range(6):
        prompt = systems[session % 2] + fresh(rng.randint(3, 9))
        history = []
        for _ in range(4):
            output = dense_greedy(model, prompt, MAX_TOKENS)
            history.append((prompt, output))
            prompt = prompt + output + fresh(rng.randint(2, 11))
        turns.append(history)
    return turns


def run_sessions(engine, sessions) -> list[list[list[int]]]:
    """Drive all sessions concurrently; a session's next turn is sent when the last ends."""
    params = SamplingParams(max_tokens=MAX_TOKENS)
    outputs = [[[] for _ in history] for history in sessions]
    owner = {}
    for index, history in enumerate(sessions):
        owner[engine.add_request(history[0][0], params)] = (index, 0)
    while engine.has_work:
        for out in engine.step():
            index, turn = owner[out.seq_id]
            outputs[index][turn].append(out.token_id)
            if out.finish_reason and turn + 1 < len(sessions[index]):
                owner[engine.add_request(sessions[index][turn + 1][0], params)] = (index, turn + 1)
    return outputs


def expected(sessions) -> list[list[list[int]]]:
    return [[output for _, output in history] for history in sessions]


def total_prompt_tokens(sessions) -> int:
    return sum(len(prompt) for history in sessions for prompt, _ in history)


def test_sessions_cache_on_equals_cache_off_equals_oracle(model, sessions):
    on = make_engine(model, prefix_cache=True)
    off = make_engine(model, prefix_cache=False)
    got_on, got_off = run_sessions(on, sessions), run_sessions(off, sessions)
    assert got_on == got_off == expected(sessions)
    assert_all_released(on)
    assert_all_released(off)

    stats = on.prefix_cache.stats
    prompt_tokens = total_prompt_tokens(sessions)
    assert stats.lookups == sum(map(len, sessions))
    assert stats.lookup_tokens == prompt_tokens - stats.lookups  # one token is never cacheable
    # Later turns resend the whole previous turn, and sessions share system prompts.
    assert stats.hit_rate > 0.6
    assert stats.prefill_tokens_saved == stats.hit_tokens + stats.catchup_tokens
    processed_on = on.stats.prefill_tokens + on.stats.decode_tokens
    processed_off = off.stats.prefill_tokens + off.stats.decode_tokens
    assert processed_off - processed_on == stats.prefill_tokens_saved
    assert off.stats.prefill_tokens + off.stats.decode_tokens >= prompt_tokens


@pytest.mark.parametrize(
    "config",
    [
        {"max_num_seqs": 1},
        {"max_num_seqs": 3, "cache_aware_scheduling": False},
        {"prefill_chunk_tokens": 6},
        {"max_batched_tokens": 9},
        {"block_size": 1},
        {"block_size": 16},
    ],
    ids=["serial", "fcfs-3", "chunked", "token-budget", "block-1", "block-16"],
)
def test_sessions_are_invariant_to_scheduling_with_the_cache_on(model, sessions, config):
    engine = make_engine(model, **config)
    assert run_sessions(engine, sessions) == expected(sessions)
    assert engine.prefix_cache.stats.hit_tokens > 0
    assert_all_released(engine)


def test_sessions_under_eviction_and_preemption(model, sessions):
    # The six sessions' final turns alone need ~120 blocks of 4 tokens; give the pool 60.
    engine = make_engine(model, num_blocks=61, watermark=0.0)
    assert run_sessions(engine, sessions) == expected(sessions)
    assert engine.prefix_cache.stats.evicted_blocks > 0
    assert_all_released(engine)


def test_sessions_with_a_pool_too_small_to_cache_anything_useful(model, sessions):
    longest = max(len(p) + MAX_TOKENS for history in sessions for p, _ in history)
    engine = make_engine(model, num_blocks=longest // 4 + 3, watermark=0.0)
    assert run_sessions(engine, sessions) == expected(sessions)
    assert engine.scheduler.num_preemptions > 0
    assert_all_released(engine)


def test_identical_concurrent_requests_compute_the_shared_prefix_once(model, sessions):
    """Catch-up: followers adopt the leader's blocks chunk by chunk instead of recomputing."""
    prompt, output = sessions[1][3]  # a long prompt
    count = 6
    engine = make_engine(model, max_batched_tokens=16, prefill_chunk_tokens=16)
    got = engine.generate([prompt] * count, SamplingParams(max_tokens=MAX_TOKENS))
    assert got == [output] * count
    stats = engine.prefix_cache.stats
    assert stats.catchup_tokens > 0
    assert_all_released(engine)

    baseline = make_engine(model, max_batched_tokens=16, prefix_cache=False)
    assert baseline.generate([prompt] * count, SamplingParams(max_tokens=MAX_TOKENS)) == got

    def processed(e) -> int:
        return e.stats.prefill_tokens + e.stats.decode_tokens

    generated = count * (len(output) - 1)  # every output token but the last is fed back
    assert processed(baseline) == count * len(prompt) + generated
    # With the cache, about one copy of the prompt is computed instead of six.
    assert processed(engine) - generated < 2 * len(prompt)
    assert processed(baseline) - processed(engine) == stats.prefill_tokens_saved


def test_duplicate_blocks_computed_in_the_same_step_are_merged(model, sessions):
    prompt, output = sessions[0][2]
    engine = make_engine(model, prefix_cache=True)
    got = engine.generate([prompt] * 4, SamplingParams(max_tokens=MAX_TOKENS))
    assert got == [output] * 4
    # All four prefilled in one step (nothing to catch up with), then collapsed onto one copy.
    assert engine.stats.prefill_tokens == 4 * len(prompt)
    full_blocks = (len(prompt) + len(output) - 1) // engine.config.block_size
    assert engine.prefix_cache.num_blocks == full_blocks
    assert_all_released(engine)


def test_repeated_request_hits_and_seeded_sampling_is_unchanged(model, sessions):
    prompt, _ = sessions[2][3]
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.9, top_p=0.95, seed=11)
    cold = make_engine(model, prefix_cache=False).generate([prompt], params)
    engine = make_engine(model)
    first = engine.generate([prompt], params)
    second = engine.generate([prompt], params)
    assert first == second == cold
    stats = engine.prefix_cache.stats
    assert stats.lookups == 2 and stats.hits == 1
    size = engine.config.block_size
    assert stats.hit_tokens == (len(prompt) - 1) // size * size


def test_abort_mid_prefill_leaves_the_cache_consistent(model, sessions):
    prompt, output = sessions[3][3]
    engine = make_engine(model, max_batched_tokens=8, prefill_chunk_tokens=8)
    doomed = engine.add_request(prompt, SamplingParams(max_tokens=MAX_TOKENS))
    for _ in range(3):
        engine.step()
    engine.abort(doomed)
    assert_all_released(engine)
    # What the aborted request computed is still usable.
    assert engine.generate([prompt], SamplingParams(max_tokens=MAX_TOKENS)) == [output]
    assert engine.prefix_cache.stats.hit_tokens > 0
