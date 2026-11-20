"""Session-aware retention: the predictor, the eviction order, and the engine (CPU only)."""

from __future__ import annotations

import random

import pytest
from test_engine import assert_all_released
from tiny import dense_greedy, make_tiny_model

from minfer.cache.block_manager import BlockManager
from minfer.cache.radix_tree import PrefixCache
from minfer.cache.retention import (
    NEVER,
    ReturnPredictor,
    SessionHint,
    SessionRetention,
    tool_key,
)
from minfer.engine import Engine, EngineConfig
from minfer.sampling import SamplingParams
from minfer.server.app import called_tools


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# ------------------------------------------------------------------------- predictor


def test_predictor_without_history_orders_like_lru():
    predictor = ReturnPredictor()
    assert predictor.expected_remaining("Bash", 3.0) == 3.0
    assert predictor.expected_remaining("Bash", 7.0) > predictor.expected_remaining("Bash", 3.0)


def test_predictor_conditions_on_the_time_already_elapsed():
    predictor = ReturnPredictor(min_samples=4)
    for pause in [0.1] * 90 + [30.0] * 10:  # heavy tail: mostly instant, sometimes 30 s
        predictor.observe("Bash", pause)
    fresh = predictor.expected_remaining("Bash", 0.0)
    assert fresh == pytest.approx(0.1 * 0.9 + 30.0 * 0.1)
    # One second in, the short pauses are ruled out: it is one of the long ones.
    assert predictor.expected_remaining("Bash", 1.0) == pytest.approx(29.0)
    # Longer than anything seen: fall back to "as long again".
    assert predictor.expected_remaining("Bash", 40.0) == 40.0


def test_predictor_is_per_tool_with_a_pooled_fallback():
    predictor = ReturnPredictor(min_samples=4)
    for _ in range(10):
        predictor.observe("Read", 0.05)
        predictor.observe("Bash", 20.0)
    assert predictor.expected_remaining("Read", 0.0) == pytest.approx(0.05)
    assert predictor.expected_remaining("Bash", 0.0) == pytest.approx(20.0)
    # A tool never seen uses everything observed.
    assert predictor.expected_remaining("WebSearch", 0.0) == pytest.approx(10.025)
    assert tool_key(["Read", "Bash", "Read"]) == "Bash+Read" and tool_key([]) == "*"


def test_retention_scores():
    clock = Clock()
    retention = SessionRetention(clock, ReturnPredictor(min_samples=2))
    for _ in range(4):
        retention.predictor.observe("Bash", 5.0)
    paused = retention.begin("a")
    assert retention.score(paused, 0.0, 0.0) == 0.0  # no request has finished yet... active
    retention.end(paused, ["Bash"], False)
    done = retention.begin("b")
    retention.end(done, [], True)
    clock.now = 2.0
    assert retention.score(paused, 0.0, clock.now) == pytest.approx(3.0)
    assert retention.score(done, 0.0, clock.now) == NEVER
    assert retention.score(None, 0.5, clock.now) == 1.5  # no session: idle time
    # The session comes back: its pause is learned, and it is no longer final or paused.
    again = retention.begin("a")
    assert again is paused and again.active == 1
    assert retention.score(again, 0.0, clock.now) == 0.0
    assert list(retention.predictor._pauses["Bash"])[-1] == pytest.approx(2.0)
    back = retention.begin("b")
    assert not back.final


# ------------------------------------------------------------------------ radix tree


def cached_sequence(cache: PrefixCache, blocks: BlockManager, seq_id: str, tokens: list[int]):
    """Compute-and-publish stand-in: allocate blocks, share them, then drop the sequence."""
    blocks.ensure_capacity(seq_id, len(tokens))
    cache.sync_sequence(seq_id, [*tokens, 0], len(tokens))
    blocks.free(seq_id)


def test_eviction_takes_finished_sessions_first_and_soonest_returning_last():
    clock = Clock()
    retention = SessionRetention(clock, ReturnPredictor(min_samples=2))
    for _ in range(4):
        retention.predictor.observe("Read", 1.0)
        retention.predictor.observe("Bash", 60.0)
    blocks = BlockManager(64, block_size=4)
    cache = PrefixCache(blocks, retention, clock)
    paths = {name: [index * 10 + 1] * 16 for index, name in enumerate(("quick", "slow", "done"))}
    clock.now = 1.0
    cached_sequence(cache, blocks, "anonymous", [99] * 16)
    finished = {"quick": (1.0, ["Read"], False), "slow": (1.2, ["Bash"], False)}
    finished["done"] = (1.4, [], True)  # the most recently used of all
    for name, (when, tools, final) in finished.items():
        clock.now = when
        state = retention.begin(name)
        cached_sequence(cache, blocks, name, paths[name])
        retention.end(state, tools, final)
        cache.tag(paths[name], state)
    # Expected time to next use at t = 1.6: done never, slow ~59.6 s, anonymous 0.6 s (its
    # idle time), quick 0.4 s.
    clock.now = 1.6

    def cached_blocks(name: str) -> int:
        return cache.match_length([*paths[name], 0]) // 4

    assert cache.evict(4) == 4  # the finished session goes first although it is the newest
    assert (cached_blocks("done"), cached_blocks("slow"), cached_blocks("quick")) == (0, 4, 4)
    assert cache.evict(2) == 2  # then the session expected back in a minute, from its tail
    assert (cached_blocks("slow"), cached_blocks("quick")) == (2, 4)
    assert cache.evict(2) == 2
    assert cached_blocks("slow") == 0 and cache.match_length([99] * 17) == 16
    assert cache.evict(4) == 4  # anonymous KV is ordered by idle time, as under LRU ...
    # ... and the session expected back within half a second is the last to go.
    assert cache.match_length([99] * 17) == 0 and cached_blocks("quick") == 4
    assert cache.evict(100) == 4
    cache.check_invariants()
    blocks.check_invariants(external_refs=cache.block_refs())
    assert blocks.num_free == blocks.num_usable


def test_lru_ignores_sessions():
    clock = Clock()
    blocks = BlockManager(64, block_size=4)
    cache = PrefixCache(blocks)
    cached_sequence(cache, blocks, "old", [1] * 16)
    cached_sequence(cache, blocks, "new", [2] * 16)
    assert cache.evict(4) == 4
    assert cache.match_length([1] * 17) == 0 and cache.match_length([2] * 17) == 16
    assert clock.now == 0.0


# ---------------------------------------------------------------------------- engine


@pytest.fixture(scope="module")
def model():
    return make_tiny_model()


def run(engine: Engine, prompt: list[int], session: SessionHint | None, max_tokens: int = 8):
    """One request to completion: ``(output tokens, prompt tokens served from the cache)``."""
    engine.add_request(prompt, SamplingParams(max_tokens=max_tokens), session)
    tokens, cached = [], 0
    while engine.has_work:
        for out in engine.step():
            tokens.append(out.token_id)
            cached = out.cached_tokens
    return tokens, cached


@pytest.mark.parametrize("policy", ["lru", "session"])
def test_paused_session_keeps_its_kv_under_pressure(model, policy):
    """A tool-call pause followed by memory pressure: who loses their cached history?"""
    rng = random.Random(11)

    def fresh(count: int) -> list[int]:
        return [rng.randrange(4, 199) for _ in range(count)]

    clock = Clock()
    config = EngineConfig(
        attention_backend="reference", num_blocks=41, block_size=4, eviction_policy=policy
    )
    engine = Engine(model, config, eos_token_ids=(), clock=clock)
    agent, chat, filler = fresh(60), fresh(60), fresh(60)
    first = dense_greedy(model, agent, 8)

    # The agent's first call ends in a tool call; a second session then gives a final answer.
    assert run(engine, agent, SessionHint("agent", ("Bash",), False)) == (first, 0)
    clock.now = 1.0
    run(engine, chat, SessionHint("chat", (), True))
    clock.now = 2.0
    # Another request needs memory: 40 usable blocks hold only two of the three contexts.
    run(engine, filler, None)
    clock.now = 3.0
    # The tool returns and the agent sends its history again, with the tool result appended.
    followup = [*agent, *first, *fresh(10)]
    output, cached = run(engine, followup, SessionHint("agent", ("Bash",), False))
    assert output == dense_greedy(model, followup, 8)  # the policy never changes the output
    if policy == "session":
        assert cached >= 60  # the finished chat session was evicted; the agent's KV survived
    else:
        # Least recently used: the paused agent was the oldest entry and lost most of its
        # history (eviction trims a leaf from its tail, so a head of it can survive).
        assert cached < 32
    assert_all_released(engine)


def test_requests_without_sessions_behave_the_same_under_both_policies(model):
    rng = random.Random(5)
    prompts = [[rng.randrange(4, 199) for _ in range(50)] for _ in range(6)]
    results = {}
    for policy in ("lru", "session"):
        config = EngineConfig(
            attention_backend="reference", num_blocks=41, block_size=4, eviction_policy=policy
        )
        engine = Engine(model, config, eos_token_ids=(), clock=Clock())
        results[policy] = [run(engine, prompt, None) for prompt in [*prompts, *prompts[:3]]]
        assert_all_released(engine)
    assert results["lru"] == results["session"]


def test_abort_and_inferred_hints(model):
    clock = Clock()
    config = EngineConfig(attention_backend="reference", num_blocks=101, block_size=4)
    engine = Engine(model, config, eos_token_ids=(), clock=clock)
    seq_id = engine.add_request([5] * 30, SamplingParams(max_tokens=50), SessionHint("s"))
    engine.step()
    engine.abort(seq_id)
    state = engine.retention.sessions["s"]
    assert state.active == 0 and state.finished_at == 0.0 and not state.final
    engine.note_session("s", [], True)  # the server read a final answer off the output
    assert state.final
    engine.note_session("unknown", ["Bash"], False)  # ignored
    assert_all_released(engine)


def test_called_tools():
    text = (
        'Let me look.\n<tool_call>\n{"name": "Bash", "arguments": {"command": "ls"}}\n</tool_call>'
    )
    assert called_tools(text) == ["Bash"]
    assert called_tools(text + '\n<tool_call>{"name":"Read","arguments":{}}</tool_call>') == [
        "Bash",
        "Read",
    ]
    assert called_tools("The bug is fixed.") == []
