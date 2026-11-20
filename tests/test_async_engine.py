"""The engine process and its asyncio front end (CPU, tiny model), and stop strings."""

from __future__ import annotations

import asyncio
import functools

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
        task = asyncio.ensure_future(
            collect(engine, [31, 32, 33], max_tokens=10_000, ignore_eos=True)
        )
        await asyncio.sleep(0.5)
        engine._process.kill()
        with pytest.raises(EngineDeadError):
            await asyncio.wait_for(task, timeout=10)
        assert not engine.healthy

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
