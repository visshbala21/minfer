"""The OpenAI-compatible server end to end: a real ``minfer serve`` process on the GPU.

Phase 5 acceptance: the official ``openai`` client works, with and without streaming.
(``vllm bench serve`` against this server is run by ``bench/serve_compat.sh``.)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest
from conftest import PRIMARY_MODEL, TESTS_DIR

pytestmark = pytest.mark.gpu


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    port = free_port()
    command = [sys.executable, "-m", "minfer.cli", "serve", "--port", str(port)]
    command += ["--host", "127.0.0.1", "--num-blocks", "8192", "--log-level", "warning"]
    process = subprocess.Popen(command, cwd=TESTS_DIR.parent, env=os.environ.copy())
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 180
        while True:
            try:
                if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert process.poll() is None, "server exited during start-up"
            assert time.time() < deadline, "server did not become healthy"
            time.sleep(0.5)
        yield url
    finally:
        process.terminate()
        process.wait(timeout=30)


@pytest.fixture(scope="module")
def client(base_url):
    from openai import OpenAI

    return OpenAI(base_url=f"{base_url}/v1", api_key="not-needed")


def metrics(base_url: str) -> dict[str, float]:
    text = httpx.get(f"{base_url}/metrics").text
    rows = (line.split() for line in text.splitlines() if line and not line.startswith("#"))
    return {name: float(value) for name, value in rows}


def test_health_and_models(base_url, client):
    assert httpx.get(f"{base_url}/health").json() == {"status": "ok"}
    models = client.models.list()
    assert [m.id for m in models.data] == [PRIMARY_MODEL]


def test_openai_client_chat_completion(client):
    response = client.chat.completions.create(
        model=PRIMARY_MODEL,
        messages=[{"role": "user", "content": "What is the capital of France? One word."}],
        max_tokens=16,
        temperature=0,
    )
    choice = response.choices[0]
    assert "Paris" in choice.message.content
    assert choice.message.role == "assistant" and choice.finish_reason == "stop"
    usage = response.usage
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.completion_tokens >= 1


def test_chat_accepts_multi_part_text_content(base_url, client):
    def ask(content):
        response = client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=12,
            temperature=0,
        )
        return response.choices[0].message.content

    parts = [
        {"type": "text", "text": "Name the largest planet. "},
        {"type": "text", "text": "One word."},
    ]
    assert ask(parts) == ask("Name the largest planet. One word.")
    image = [{"type": "image_url", "image_url": {"url": "http://example.invalid/x.png"}}]
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": image}]},
    )
    assert response.status_code == 400 and "image_url" in response.json()["error"]["message"]


def test_openai_client_chat_streaming_matches_non_streaming(client):
    request = {
        "model": PRIMARY_MODEL,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Count from one to eight in words, comma separated."},
        ],
        "max_tokens": 48,
        "temperature": 0,
    }
    whole = client.chat.completions.create(**request)
    stream = client.chat.completions.create(
        **request, stream=True, stream_options={"include_usage": True}
    )
    chunks = list(stream)
    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    reasons = [c.choices[0].finish_reason for c in chunks if c.choices]
    assert reasons[-1] == whole.choices[0].finish_reason and set(reasons[:-1]) == {None}
    assert text == whole.choices[0].message.content
    assert len([c for c in chunks if c.choices]) > 3  # it really arrived in pieces
    usage = chunks[-1].usage
    assert chunks[-1].choices == [] and usage.completion_tokens == whole.usage.completion_tokens
    assert usage.prompt_tokens == whole.usage.prompt_tokens


def test_openai_client_completions_streaming(client):
    stream = client.completions.create(
        model=PRIMARY_MODEL,
        prompt="The first five prime numbers are 2, 3,",
        max_tokens=12,
        temperature=0,
        stream=True,
    )
    chunks = list(stream)
    text = "".join(c.choices[0].text for c in chunks)
    assert "5" in text and "7" in text
    assert chunks[-1].choices[0].finish_reason == "length"
    whole = client.completions.create(
        model=PRIMARY_MODEL,
        prompt="The first five prime numbers are 2, 3,",
        max_tokens=12,
        temperature=0,
    )
    assert whole.choices[0].text == text and whole.usage.completion_tokens == 12


def test_completions_prompt_forms(base_url, client):
    tokens = httpx.post(f"{base_url}/tokenize", json={"prompt": "Hello there, world"}).json()
    assert tokens["count"] == len(tokens["tokens"]) > 0
    assert tokens["max_model_len"] == 32768
    back = httpx.post(f"{base_url}/detokenize", json={"tokens": tokens["tokens"]}).json()
    assert back["prompt"] == "Hello there, world"

    from_text = client.completions.create(
        model=PRIMARY_MODEL, prompt="Hello there, world", max_tokens=8, temperature=0
    )
    from_ids = client.completions.create(
        model=PRIMARY_MODEL, prompt=tokens["tokens"], max_tokens=8, temperature=0
    )
    assert from_text.choices[0].text == from_ids.choices[0].text
    batch = client.completions.create(
        model=PRIMARY_MODEL, prompt=["One, two, three,", "A, B, C,"], max_tokens=6, temperature=0
    )
    assert sorted(c.index for c in batch.choices) == [0, 1]
    assert batch.usage.completion_tokens == 12


def test_stop_string_and_seeded_sampling(client):
    stopped = client.completions.create(
        model=PRIMARY_MODEL, prompt="1 2 3 4 5 6 7 8", max_tokens=40, temperature=0, stop=[" 12"]
    )
    assert stopped.choices[0].text == " 9 10 11"
    assert stopped.choices[0].finish_reason == "stop"

    def sample(seed: int) -> str:
        response = client.completions.create(
            model=PRIMARY_MODEL,
            prompt="Once upon a time",
            max_tokens=24,
            temperature=1.0,
            top_p=0.95,
            seed=seed,
        )
        return response.choices[0].text

    assert sample(1) == sample(1)
    assert sample(1) != sample(2)


def test_errors_use_the_openai_envelope(base_url, client):
    import openai

    with pytest.raises(openai.NotFoundError):
        client.completions.create(model="no-such-model", prompt="x")
    with pytest.raises(openai.BadRequestError, match="n=1"):
        client.completions.create(model=PRIMARY_MODEL, prompt="x", n=3)
    with pytest.raises(openai.BadRequestError, match="context"):
        client.completions.create(model=PRIMARY_MODEL, prompt=[1] * 40_000, max_tokens=1)
    response = httpx.post(f"{base_url}/v1/chat/completions", json={"messages": []})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert httpx.get(f"{base_url}/health").status_code == 200  # still serving


def test_prefix_cache_is_reported_in_usage_and_metrics(base_url, client):
    system = "You are a meticulous assistant. " + "Always cite the relevant rule. " * 60
    before = metrics(base_url)

    def ask(question: str):
        return client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
            max_tokens=8,
            temperature=0,
        )

    first, second = ask("What is rule one?"), ask("What is rule two?")
    assert first.usage.prompt_tokens > 300
    cached = second.usage.prompt_tokens_details.cached_tokens
    assert cached >= 0.8 * second.usage.prompt_tokens  # the shared system prompt
    time.sleep(0.6)  # metrics snapshots are published every 250 ms
    after = metrics(base_url)
    assert (
        after["minfer_prefix_cache_hit_tokens_total"]
        - before["minfer_prefix_cache_hit_tokens_total"]
        >= cached
    )
    assert after["minfer_requests_finished_total"] - before["minfer_requests_finished_total"] == 2
    assert 0.0 < after["minfer_prefix_cache_hit_rate"] <= 1.0
    for name in ("requests_waiting", "kv_usage_ratio", "generation_tokens_per_second"):
        assert f"minfer_{name}" in after


def test_sse_wire_format(base_url):
    body = {"prompt": "Hello", "max_tokens": 3, "temperature": 0, "stream": True}
    with httpx.stream("POST", f"{base_url}/v1/completions", json=body) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line]
    assert all(line.startswith("data: ") for line in lines)
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(line[6:]) for line in lines[:-1]]
    assert all(event["object"] == "text_completion" for event in events)
    assert events[-1]["choices"][0]["finish_reason"] == "length"
