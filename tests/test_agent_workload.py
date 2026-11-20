"""The agent workload generator: determinism, structure, and trace round-trips (CPU only)."""

from __future__ import annotations

import json

import pytest

from bench import agent_workload as aw


def count(text: str) -> int:
    """A stand-in tokenizer: roughly four characters per token."""
    return max(1, len(text) // 4)


@pytest.fixture(scope="module")
def trace():
    return aw.generate(40, 4.0, seed=3, count_tokens=count)


def test_generation_is_deterministic(trace):
    again = aw.generate(40, 4.0, seed=3, count_tokens=count)
    assert again.sha256 == trace.sha256
    assert aw.generate(40, 4.0, seed=4, count_tokens=count).sha256 != trace.sha256


def test_session_contents_do_not_depend_on_the_rate(trace):
    other = aw.generate(40, 16.0, seed=3, count_tokens=count)
    assert [s.turns for s in other.sessions] != []
    for a, b in zip(trace.sessions, other.sessions, strict=True):
        assert (a.agent, a.task) == (b.agent, b.task)
        assert [(t.assistant, t.tool, t.max_tokens) for t in a.turns] == [
            (t.assistant, t.tool, t.max_tokens) for t in b.turns
        ]
    assert other.sessions[-1].start < trace.sessions[-1].start  # same sessions, arriving faster


def test_session_structure(trace):
    assert {s.agent for s in trace.sessions} == set(trace.agents)
    for session in trace.sessions:
        assert aw.MIN_TURNS <= len(session.turns) <= aw.MAX_TURNS
        assert all(turn.tool is not None for turn in session.turns[:-1])
        assert session.turns[-1].tool is None  # the final answer ends the session
        assert all(turn.max_tokens == count(turn.assistant) > 0 for turn in session.turns)
        assert all(turn.think >= 0 for turn in session.turns)
        assert all("<tool_call>" in turn.assistant for turn in session.turns[:-1])
    assert trace.num_requests == sum(len(s.turns) for s in trace.sessions)
    assert len(list(aw.iter_requests(trace))) == trace.num_requests


def test_tool_calls_are_valid_json(trace):
    tools = {
        agent: {tool["function"]["name"] for tool in spec["tools"]}
        for agent, spec in trace.agents.items()
    }
    for session in trace.sessions:
        for turn in session.turns[:-1]:
            body = turn.assistant.split("<tool_call>\n")[1].split("\n</tool_call>")[0]
            call = json.loads(body)
            assert call["name"] in tools[session.agent]
            assert isinstance(call["arguments"], dict)


def test_history_grows_and_shares_the_system_prompt(trace):
    session = trace.sessions[0]
    system = trace.agents[session.agent]["system"]
    previous: list[dict[str, str]] = []
    for turn in range(len(session.turns)):
        messages = session.messages(turn, system)
        assert messages[0] == {"role": "system", "content": system}
        assert messages[1] == {"role": "user", "content": session.task}
        assert messages[: len(previous)] == previous  # every request extends the last one
        assert len(messages) == 2 + 2 * turn
        previous = messages
    roles = [m["role"] for m in previous[2:]]
    assert roles == ["assistant", "tool"] * (len(session.turns) - 1)


def test_system_prompts_hit_the_target_size_and_are_shared():
    for agent in ("coder", "analyst", "researcher"):
        prompt = aw.system_prompt(agent, count, 4000)
        tools = json.dumps(aw._AGENT_TOOLS[agent], indent=2)
        total = count(prompt) + count(tools)
        assert 4000 <= total < 4200
        assert prompt == aw.system_prompt(agent, count, 4000)
    small = aw.generate(3, None, seed=0, count_tokens=count, system_tokens=500)
    assert all(count(spec["system"]) < 600 for spec in small.agents.values())


def test_arrivals_follow_the_requested_rate():
    trace = aw.generate(2000, 8.0, seed=1, count_tokens=count, system_tokens=200)
    starts = [s.start for s in trace.sessions]
    assert starts == sorted(starts)
    session_rate = len(starts) / starts[-1]
    assert session_rate == pytest.approx(8.0 / aw.MEAN_TURNS, rel=0.1)
    thinks = [t.think for s in trace.sessions for t in s.turns[:-1]]
    assert sum(thinks) / len(thinks) == pytest.approx(1.0, rel=0.1)


def test_max_rate_starts_everything_at_once_without_think_time():
    trace = aw.generate(10, None, seed=0, count_tokens=count, system_tokens=200)
    assert all(s.start == 0.0 for s in trace.sessions)
    assert all(t.think == 0.0 for s in trace.sessions for t in s.turns)


def test_jsonl_round_trip(trace, tmp_path):
    path = tmp_path / "trace.jsonl"
    trace.save(path)
    lines = path.read_text().splitlines()
    assert json.loads(lines[0])["type"] == "header" and len(lines) == 1 + len(trace.sessions)
    loaded = aw.Trace.load(path)
    assert loaded.sha256 == trace.sha256
    assert loaded.sessions == trace.sessions
    path.write_text('{"type": "header", "version": 999}\n')
    with pytest.raises(ValueError):
        aw.Trace.load(path)


def test_summarize():
    def record(turn, sent, first, done, tokens, cached=None, error=None):
        r = aw.RequestRecord("s", "coder", turn, sent, first, done, 1000, tokens, cached, error)
        return r

    records = [
        record(0, 0.0, 0.5, 1.5, 11, cached=0),
        record(1, 2.0, 2.1, 2.6, 6, cached=900),
        record(2, 3.0, None, None, 0, error="HTTP 500"),
    ]
    summary = aw.summarize(records, duration=4.0)
    assert (summary["requests"], summary["completed"], summary["failed"]) == (3, 2, 1)
    assert summary["errors"] == ["HTTP 500"]
    assert summary["completion_tokens"] == 17
    assert summary["output_tokens_per_second"] == pytest.approx(17 / 4)
    assert summary["ttft_ms"]["p50"] == pytest.approx(300.0)
    assert summary["ttft_first_turn_ms"]["p50"] == pytest.approx(500.0)
    assert summary["ttft_later_turns_ms"]["p50"] == pytest.approx(100.0)
    assert summary["tpot_ms"]["max"] == pytest.approx(100.0)  # (1.5 - 0.5) / 10 and 0.5 / 5
    assert summary["e2e_ms"]["max"] == pytest.approx(1500.0)
    assert summary["client_reported_cached_token_fraction"] == pytest.approx(0.45)
