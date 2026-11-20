"""Real-trace tooling: pause sampling, compilation, reuse fingerprints (CPU only)."""

from __future__ import annotations

import itertools
import json
import random

import numpy as np
import pytest

from bench import real_traces as rt


class WordTokenizer:
    """A stand-in tokenizer: one token per whitespace-separated word, append-only template."""

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True):
        parts = [f"<tools> {len(tools or [])} "]
        for message in messages:
            calls = " ".join(
                f"call {call['function']['name']} {json.dumps(call['function']['arguments'])}"
                for call in message.get("tool_calls", ())
            )
            parts.append(f"<{message['role']}> {message['content']} {calls} <end> ")
        if add_generation_prompt:
            parts.append("<assistant> ")
        return "".join(parts)

    def __call__(self, texts, add_special_tokens=False):
        ids = [
            [self.vocabulary.setdefault(word, len(self.vocabulary)) for word in text.split()]
            for text in texts
        ]
        return {"input_ids": ids}


def conversation(turns: int, words_per_result: int = 20) -> list[dict]:
    messages = [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "fix the bug"},
    ]
    for turn in range(turns):
        call = {"type": "function", "function": {"name": "Bash", "arguments": {"n": turn}}}
        messages.append({"role": "assistant", "content": f"step {turn}", "tool_calls": [call]})
        result = " ".join(f"line{turn}_{i}" for i in range(words_per_result))
        messages.append({"role": "tool", "content": result})
    messages.append({"role": "assistant", "content": "done"})
    return messages


TABLE = {
    "tools": {
        "Bash": {"calls": 1000, "quantiles_s": np.linspace(1.0, 3.0, rt.NUM_QUANTILES).tolist()},
        "*": {"calls": 1000, "quantiles_s": np.linspace(10.0, 30.0, rt.NUM_QUANTILES).tolist()},
    }
}


def test_pause_sampler_follows_the_tool_and_respects_the_cap_and_scale():
    sampler = rt.PauseSampler(TABLE, max_pause=20.0, time_scale=0.5)
    rng = random.Random(0)
    bash = [sampler.sample(["Bash"], rng) for _ in range(200)]
    assert min(bash) >= 0.5 and max(bash) <= 1.5  # [1, 3] s, halved
    unknown = [sampler.sample(["Mystery"], rng) for _ in range(200)]
    assert min(unknown) >= 5.0 and max(unknown) == 10.0  # falls back to '*', capped at 20 s
    assert sampler.sample([], rng) == 0.0
    # Parallel calls wait for the slowest one.
    assert sampler.sample(["Bash", "Mystery"], random.Random(1)) >= 5.0
    assert [sampler.sample(["Bash"], random.Random(7)) for _ in range(3)] == [
        sampler.sample(["Bash"], random.Random(7)) for _ in range(3)
    ]


def test_the_committed_latency_table_is_well_formed():
    table = json.loads(rt.LATENCY_TABLE.read_text())
    assert {"Bash", "Read", "Edit", "*"} <= set(table["tools"])
    for entry in table["tools"].values():
        quantiles = entry["quantiles_s"]
        assert len(quantiles) == rt.NUM_QUANTILES and entry["calls"] >= rt.MIN_TOOL_SAMPLES
        assert all(a <= b for a, b in itertools.pairwise(quantiles))


def test_compile_session_keeps_the_calls_that_fit_the_context():
    tokenizer = WordTokenizer()
    sampler = rt.PauseSampler(TABLE, 60.0, 1.0)
    messages = conversation(6)
    full = rt.compile_session("s", messages, [], tokenizer, 10_000, sampler, random.Random(0))
    assert len(full.calls) == 7 and not full.truncated
    assert [call.message for call in full.calls] == [2, 4, 6, 8, 10, 12, 14]
    assert [call.tools for call in full.calls] == [["Bash"]] * 6 + [[]]
    assert all(call.pause > 0 for call in full.calls[:-1]) and full.calls[-1].pause == 0.0
    lengths = [call.prompt_tokens for call in full.calls]
    assert lengths == sorted(lengths) and len(set(lengths)) == len(lengths)
    # Each prompt is the conversation before the call; the output is the recorded message.
    first = rt.render(full, full.calls[0], tokenizer)
    assert first.endswith("<assistant> ") and "step 0" not in first
    assert full.calls[0].prompt_tokens == len(first.split())

    limit = full.calls[3].prompt_tokens + full.calls[3].max_tokens
    cut = rt.compile_session("s", messages, [], tokenizer, limit, sampler, random.Random(0))
    assert len(cut.calls) == 4 and cut.truncated
    assert cut.calls[-1].pause == 0.0  # the session ends there: nothing to wait for
    assert len(cut.messages) == cut.calls[-1].message + 1
    assert rt.compile_session("s", messages, [], tokenizer, 3, sampler, random.Random(0)) is None


def test_trace_round_trip(tmp_path):
    tokenizer = WordTokenizer()
    sampler = rt.PauseSampler(TABLE, 60.0, 1.0)
    trace = rt.RealTrace({"version": rt.TRACE_VERSION, "model": "tiny"})
    for index in range(3):
        session = rt.compile_session(
            f"s{index}", conversation(3 + index), [], tokenizer, 10_000, sampler, random.Random(0)
        )
        trace.sessions.append(session)
    path = tmp_path / "trace.jsonl"
    trace.save(path)
    loaded = rt.RealTrace.load(path)
    assert loaded.sha256 == trace.sha256 and loaded.num_requests == trace.num_requests == 15
    assert len(rt.RealTrace.load(path, max_sessions=2).sessions) == 2


def test_window_fingerprints_are_position_independent():
    rng = np.random.default_rng(0)
    run = rng.integers(0, 50_000, size=200)
    a = np.concatenate([rng.integers(0, 50_000, size=37), run])
    b = np.concatenate([rng.integers(0, 50_000, size=411), run, rng.integers(0, 50_000, size=9)])
    ha = rt._window_hashes(rt._prefix_sums(a), 32)
    hb = rt._window_hashes(rt._prefix_sums(b), 32)
    assert np.array_equal(ha[37 : 37 + 169], hb[411 : 411 + 169])
    assert len(np.intersect1d(ha[:37], hb)) == 0
    # A run of zeros must not hash like a shorter run of zeros.
    zeros = rt._prefix_sums(np.zeros(64, dtype=np.int64))
    assert zeros[16] != zeros[32]


def test_reuse_classifier():
    rng = np.random.default_rng(1)
    classifier = rt.ReuseClassifier(window=32, block=16)
    first = rng.integers(0, 50_000, size=400)
    assert classifier.classify(first) == (0, 0, 400)
    # The same prompt again: everything up to the last full block is a cached prefix.
    assert classifier.classify(first) == (400, 0, 0)
    # An append-only continuation: the old prompt is prefix, the new tokens are novel.
    longer = np.concatenate([first, rng.integers(0, 50_000, size=100)])
    assert classifier.classify(longer) == (400, 0, 100)
    # Known content behind a different head: no prefix, but a position-independent match.
    shifted = np.concatenate([rng.integers(0, 50_000, size=50), first[100:300]])
    prefix, duplicate, novel = classifier.classify(shifted)
    assert (prefix, duplicate, novel) == (0, 200, 50)
    # A duplicate run shorter than the window is not detected.
    short = np.concatenate([rng.integers(0, 50_000, size=64), first[:20]])
    assert classifier.classify(short) == (0, 0, 84)


def test_trace_statistics_on_a_tiny_trace():
    tokenizer = WordTokenizer()
    sampler = rt.PauseSampler(TABLE, 60.0, 1.0)
    trace = rt.RealTrace({"version": rt.TRACE_VERSION, "model": "tiny"})
    for index in range(2):
        trace.sessions.append(
            rt.compile_session(
                f"s{index}", conversation(5, 60), [], tokenizer, 10_000, sampler, random.Random(0)
            )
        )
    stats = rt.trace_statistics(trace, tokenizer)
    assert stats["sessions"] == 2 and stats["model_calls"] == 12
    assert stats["tool_calls"] == {"Bash": 10}
    fractions = stats["token_fractions"]
    assert fractions["prefix"] + fractions["shifted_duplicate"] + fractions["novel"] == (
        pytest.approx(1.0)
    )
    # The second session repeats the first exactly, and each call extends the one before:
    # most tokens are a cached prefix (short prompts lose up to a block each to alignment).
    assert fractions["prefix"] > 0.75
    assert stats["total_prompt_tokens"] == sum(
        c.prompt_tokens for s in trace.sessions for c in s.calls
    )
