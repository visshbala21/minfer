"""The engine process and its asyncio front end (CPU, tiny model), and stop strings."""

from __future__ import annotations

import asyncio
import functools
import time

import pytest
from tiny import dense_greedy, make_tiny_engine, make_tiny_model

from minfer.sampling import SamplingParams
from minfer.server.app import StopStrings
from minfer.server.async_engine import AsyncEngine, EngineDeadError


def run(coroutine):
    return asyncio.run(coroutine)


async def started(**config) -> AsyncEngine:
    engine = AsyncEngine(functools.partial(make_tiny_engine, **config))
    await engine.start()
    return engine


async def collect(engine: AsyncEngine, prompt: list[int], **params) -> tuple[list[int], str]:
    tokens, reason = [], None
    async for out in engine.generate(prompt, SamplingParams(**params)):
        tokens.append(out.token_id)
        reason = out.finish_reason
    return tokens, reason


PROMPTS = [[5 + i, 40 + 2 * i, 77, 120 - i, 9 + 3 * i] * (1 + i % 3) for i in range(12)]


def test_commands_sent_while_results_are_backed_up_do_not_deadlock():
    """A front end that falls behind on results must still be able to submit requests.

    The engine blocks writing results once the 64 KB pipe is full. When the event loop
    wrote commands itself, a command larger than what was left of the *command* pipe then
    blocked the loop too — and a blocked loop reads no results: both ends waited on each
    other forever (DESIGN.md B-007, found by 1,000 simultaneous requests at 7B).
    """
    prompts = [[4 + (i * 7 + j) % 190 for j in range(8)] for i in range(164)]
    # Stop tokens outside the vocabulary: they never fire, they only make each command
    # several kilobytes, so the second wave is far more than a pipe holds.
    bulk = tuple(range(1000, 3000))

    async def scenario():
        engine = await started(num_blocks=4000)
        try:
            first = [
                asyncio.ensure_future(collect(engine, p, max_tokens=200, ignore_eos=True))
                for p in prompts[:64]
            ]
            await asyncio.sleep(0.2)
            # The loop is busy elsewhere for 8 s. The first wave produces ~130 KB of results at
            # ~60 ms a step, so well before that the results pipe is full and the engine blocked.
            time.sleep(8.0)
            second = [
                collect(engine, p, max_tokens=4, ignore_eos=True, stop_token_ids=bulk)
                for p in prompts[64:]
            ]
            return await asyncio.wait_for(asyncio.gather(*first, *second), timeout=120)
        finally:
            await engine.stop()

    results = run(scenario())
    assert [len(tokens) for tokens, _ in results] == [200] * 64 + [4] * 100
    assert all(reason == "length" for _, reason in results)


def test_concurrent_requests_match_the_oracle():
    model = make_tiny_model()
    expected = [dense_greedy(model, p, 10) for p in PROMPTS]

    async def scenario():
        engine = await started()
        try:
            assert engine.healthy and engine.info["max_model_len"] == 256
            results = await asyncio.gather(*(collect(engine, p, max_tokens=10) for p in PROMPTS))
            await asyncio.sleep(0.4)  # let the idle engine publish its final metrics
            return results, dict(engine.metrics)
        finally:
            await engine.stop()

    results, metrics = run(scenario())
    assert [tokens for tokens, _ in results] == expected
    assert all(reason == "length" for _, reason in results)
    assert metrics["requests_finished_total"] == len(PROMPTS)
    assert metrics["generation_tokens_total"] == 10 * len(PROMPTS)
    assert metrics["requests_running"] == 0 and metrics["requests_waiting"] == 0
    assert 0.0 <= metrics["kv_usage_ratio"] <= 1.0
    assert "prefix_cache_hit_rate" in metrics


def test_rejected_request_raises_and_engine_keeps_serving():
    async def scenario():
        engine = await started(max_model_len=16)
        try:
            with pytest.raises(ValueError, match="at most"):
                await collect(engine, list(range(4, 30)), max_tokens=4)
            return await collect(engine, [7, 8, 9], max_tokens=3)
        finally:
            await engine.stop()

    tokens, reason = run(scenario())
    assert len(tokens) == 3 and reason == "length"


def test_abandoning_a_stream_aborts_the_request():
    async def scenario():
        engine = await started()
        try:
            stream = engine.generate([11, 12, 13], SamplingParams(max_tokens=200, ignore_eos=True))
            seen = 0
            async for _ in stream:
                seen += 1
                if seen == 3:
                    break
            await stream.aclose()
            other = await collect(engine, [21, 22], max_tokens=5)
            await asyncio.sleep(0.4)
            return other, dict(engine.metrics)
        finally:
            await engine.stop()

    (tokens, _), metrics = run(scenario())
    assert len(tokens) == 5
    # The abandoned request left the engine without finishing normally.
    assert metrics["requests_total"] == 2 and metrics["requests_finished_total"] == 1
    assert metrics["requests_running"] == 0


def _broken_factory():
    raise RuntimeError("no model here")


def test_start_failure_is_reported():
    async def scenario():
        engine = AsyncEngine(_broken_factory)
        with pytest.raises(EngineDeadError, match="no model here"):
            await engine.start()
        assert not engine.healthy
        with pytest.raises(EngineDeadError):
            await collect(engine, [1, 2], max_tokens=2)

    run(scenario())


def test_engine_process_death_fails_in_flight_requests():
    async def scenario():
        engine = await started()
        params = SamplingParams(max_tokens=10_000, ignore_eos=True)
        stream = engine.generate([31, 32, 33], params)
        await stream.__anext__()  # the request is running
        engine._process.kill()
        with pytest.raises(EngineDeadError):
            async for _ in stream:
                pass
        assert not engine.healthy
        with pytest.raises(EngineDeadError):
            await collect(engine, [1, 2], max_tokens=2)

    run(scenario())


# ------------------------------------------------------------------- stop strings


def feed_all(stops: list[str], pieces: list[str]) -> tuple[str, bool]:
    checker = StopStrings(stops)
    out = ""
    for piece in pieces:
        out += checker.feed(piece)
        if checker.stopped:
            return out, True
    return out + checker.flush(), False


def test_stop_strings_passthrough_without_stops():
    assert feed_all([], ["ab", "cd"]) == ("abcd", False)


def test_stop_string_inside_one_piece():
    assert feed_all(["STOP"], ["hello STOP world"]) == ("hello ", True)


def test_stop_string_split_across_pieces_is_never_leaked():
    checker = StopStrings(["\n\nUser:"])
    released = [checker.feed(p) for p in ["Answer.", "\n", "\nUs", "er: next"]]
    assert "".join(released) == "Answer."
    assert checker.stopped
    # Nothing that could be the start of the stop string was sent early.
    assert all("\n" not in r for r in released)


def test_false_start_is_released_later():
    assert feed_all(["abc"], ["xa", "b", "d", "e"]) == ("xabde", False)


def test_earliest_of_several_stops_wins():
    assert feed_all(["zzz", "b"], ["aabzzz"]) == ("aa", True)


# ---------------------------------------------------------------- batch tokenizer


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class _FakeTokenizer:
    def __init__(self):
        self.batches = []

    def encode_batch_fast(self, texts, add_special_tokens=False):
        self.batches.append((list(texts), add_special_tokens))
        if any(text == "boom" for text in texts):
            raise RuntimeError("cannot tokenize")
        offset = 1000 if add_special_tokens else 0
        return [_FakeEncoding([offset + len(text)]) for text in texts]


def test_batch_tokenizer_groups_concurrent_requests():
    from minfer.server.app import BatchTokenizer

    fake = _FakeTokenizer()

    async def scenario():
        tokenizer = BatchTokenizer(fake)
        texts = ["a" * n for n in range(1, 21)]
        results = await asyncio.gather(*(tokenizer.encode(text) for text in texts))
        special = await tokenizer.encode("abc", add_special_tokens=True)
        return results, special

    results, special = run(scenario())
    assert results == [[n] for n in range(1, 21)]  # each caller gets its own result
    assert special == [1003]
    assert sum(len(texts) for texts, _ in fake.batches) == 21
    assert len(fake.batches) < 21  # concurrent requests shared batches


def test_batch_tokenizer_reports_errors_and_keeps_working():
    from minfer.server.app import BatchTokenizer

    async def scenario():
        tokenizer = BatchTokenizer(_FakeTokenizer())
        with pytest.raises(RuntimeError, match="cannot tokenize"):
            await tokenizer.encode("boom")
        return await tokenizer.encode("fine")

    assert run(scenario()) == [4]
