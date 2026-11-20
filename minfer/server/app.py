"""OpenAI-compatible HTTP API over the engine.

Routes: ``/v1/completions``, ``/v1/chat/completions`` (chat template applied, tools passed
to the template), ``/v1/models``, ``/health``, ``/metrics`` (Prometheus text format), and
vLLM-compatible ``/tokenize`` / ``/detokenize``. Streaming uses server-sent events.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import orjson
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response, StreamingResponse
from tokenizers.decoders import DecodeStream

from minfer.engine import Engine
from minfer.sampling import SamplingParams
from minfer.server.async_engine import AsyncEngine, EngineDeadError
from minfer.server.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    DetokenizeRequest,
    TokenizeRequest,
    _GenerationRequest,
)

_SSE_DONE = b"data: [DONE]\n\n"

_METRIC_HELP = {
    "requests_running": "Requests holding KV and being generated",
    "requests_waiting": "Requests queued for admission (queue depth)",
    "requests_total": "Requests accepted",
    "requests_finished_total": "Requests completed",
    "prompt_tokens_total": "Prompt tokens received",
    "generation_tokens_total": "Tokens generated",
    "generation_tokens_per_second": "Output throughput since the previous scrape",
    "prefill_tokens_total": "Tokens run through prefill passes",
    "decode_tokens_total": "Tokens run through decode passes",
    "verify_tokens_total": "Tokens run through speculative verification",
    "spec_rounds_total": "Speculative verification rounds",
    "spec_proposed_tokens_total": "Draft tokens proposed",
    "spec_accepted_tokens_total": "Draft tokens accepted",
    "engine_steps_total": "Engine iterations",
    "engine_step_seconds_total": "Time spent inside engine steps",
    "preemptions_total": "Sequences preempted for memory",
    "kv_blocks": "KV blocks in the pool",
    "kv_blocks_used": "KV blocks referenced by sequences or the prefix cache",
    "kv_blocks_evictable": "KV blocks held only by the prefix cache",
    "kv_usage_ratio": "Fraction of the KV pool in use",
    "prefix_cache_hit_rate": "Looked-up prompt tokens served from the cache (lifetime)",
    "prefix_cache_lookup_tokens_total": "Prompt tokens looked up at admission",
    "prefix_cache_hit_tokens_total": "Prompt tokens served from the cache at admission",
    "prefix_cache_catchup_tokens_total": "Tokens adopted from the cache during prefill",
    "prefill_tokens_saved_total": "Tokens whose KV was reused instead of computed",
    "prefix_cache_requests_hit_total": "Requests that found a cached prefix",
    "prefix_cache_blocks": "KV blocks in the radix tree",
    "prefix_cache_nodes": "Nodes in the radix tree",
    "prefix_cache_evicted_blocks_total": "Blocks evicted from the cache",
}


class APIError(Exception):
    """An error reported to the client in OpenAI's error envelope."""

    def __init__(self, status: int, message: str, kind: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind


def _json(payload: dict[str, Any], status: int = 200) -> Response:
    return Response(orjson.dumps(payload), status_code=status, media_type="application/json")


def _error(status: int, message: str, kind: str) -> Response:
    return _json({"error": {"message": message, "type": kind, "code": status}}, status)


def _sse(payload: dict[str, Any]) -> bytes:
    return b"data: " + orjson.dumps(payload) + b"\n\n"


def _text_message(message: dict[str, Any]) -> dict[str, Any]:
    """A chat message with multi-part content flattened to the string the template expects.

    OpenAI clients may send ``content`` as a list of typed parts. Text parts are
    concatenated; any other part type (images, audio) is rejected: the model is text-only.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return message
    texts = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            kind = part.get("type") if isinstance(part, dict) else type(part).__name__
            raise APIError(400, f"unsupported content part type {kind!r}; only text is supported")
        texts.append(part.get("text", ""))
    return {**message, "content": "".join(texts)}


class BatchTokenizer:
    """Tokenizes concurrent requests together, off the event loop.

    Agent prompts are tens of kilobytes. Encoding one takes ~7 ms and holds the GIL, so
    handing each request to a thread pool makes concurrent requests *slower* (measured:
    25-29 ms each with 4-16 threads). ``encode_batch_fast`` releases the GIL and
    parallelizes in Rust (0.6 ms per prompt in a batch of 64). Requests that arrive while
    a batch is being encoded simply form the next batch, so batching adds no waiting.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self._pending: dict[bool, list[tuple[str, asyncio.Future]]] = {False: [], True: []}
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="minfer-tokenize")
        self._drain_task: asyncio.Task | None = None

    async def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Token ids of ``text``."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[add_special_tokens].append((text, future))
        if self._drain_task is None:
            self._drain_task = loop.create_task(self._drain())
        return await future

    async def _drain(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while any(self._pending.values()):
                for special, batch in self._pending.items():
                    if not batch:
                        continue
                    self._pending[special] = []
                    texts = [text for text, _ in batch]
                    try:
                        encodings = await loop.run_in_executor(
                            self._worker,
                            lambda t=texts, s=special: self.tokenizer.encode_batch_fast(
                                t, add_special_tokens=s
                            ),
                        )
                    except Exception as error:
                        for _, future in batch:
                            if not future.done():
                                future.set_exception(error)
                        continue
                    for (_, future), encoding in zip(batch, encodings, strict=True):
                        if not future.done():
                            future.set_result(encoding.ids)
        finally:
            self._drain_task = None


class StopStrings:
    """Incrementally applies stop strings to streamed text.

    Text that could still turn out to be the beginning of a stop string is held back, so
    nothing after (or inside) a stop string is ever sent to the client.
    """

    def __init__(self, stops: list[str]) -> None:
        self.stops = stops
        self.holdback = max((len(s) for s in stops), default=1) - 1
        self.buffer = ""
        self.stopped = False

    def feed(self, text: str) -> str:
        """Add generated text; return what can be released to the client."""
        if not self.stops:
            return text
        self.buffer += text
        cut = min((i for i in (self.buffer.find(s) for s in self.stops) if i >= 0), default=-1)
        if cut >= 0:
            self.stopped = True
            released, self.buffer = self.buffer[:cut], ""
            return released
        keep = min(self.holdback, len(self.buffer))
        released = self.buffer[: len(self.buffer) - keep]
        self.buffer = self.buffer[len(self.buffer) - keep :]
        return released

    def flush(self) -> str:
        """Release held-back text once generation ended without a stop string."""
        released, self.buffer = self.buffer, ""
        return released


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int = 0
    cached_tokens: int = 0

    def payload(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": self.cached_tokens},
        }


class Server:
    """Request handling shared by the routes; owns the tokenizer and the async engine."""

    def __init__(
        self, engine_factory: Callable[[], Engine], tokenizer: Any, model_name: str
    ) -> None:
        self.async_engine = AsyncEngine(engine_factory)
        self.tokenizer = tokenizer
        self.fast_tokenizer = tokenizer.backend_tokenizer
        self.batch_tokenizer = BatchTokenizer(self.fast_tokenizer)
        self.model_name = model_name
        self.created = int(time.time())
        self._rate_mark = (time.monotonic(), 0.0)

    @property
    def max_model_len(self) -> int:
        """The engine's context limit (known once the engine process is ready)."""
        return self.async_engine.info["max_model_len"]

    # ------------------------------------------------------------- helpers

    def check_model(self, name: str | None) -> None:
        """Reject requests addressed to a model this server does not serve."""
        if name is not None and name != self.model_name:
            raise APIError(404, f"model {name!r} does not exist", "model_not_found")

    async def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Tokenize ``text`` without blocking the event loop."""
        return await self.batch_tokenizer.encode(text, add_special_tokens)

    def sampling(
        self, request: _GenerationRequest, prompt_len: int, limit: int | None
    ) -> SamplingParams:
        """Engine parameters for a request, with the output limit clamped to the context."""
        room = self.max_model_len - prompt_len
        if room < 1:
            raise APIError(
                400,
                f"the prompt has {prompt_len} tokens; the model's context is "
                f"{self.max_model_len} tokens",
            )
        return request.sampling_params(min(limit, room) if limit is not None else room)

    async def stream_text(
        self,
        prompt_ids: list[int],
        params: SamplingParams,
        request: _GenerationRequest,
        usage: _Usage,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Generate and yield ``(text_delta, finish_reason)``; the last item has a reason.

        Detokenization is incremental; stop strings end the request early (the engine
        request is aborted when this generator is closed before exhaustion).
        """
        decoder = DecodeStream(skip_special_tokens=request.skip_special_tokens)
        stops = StopStrings(request.stop_strings)
        tokenizer = self.fast_tokenizer
        async for out in self.async_engine.generate(prompt_ids, params):
            usage.completion_tokens += 1
            piece = decoder.step(tokenizer, out.token_id) or ""
            text = stops.feed(piece)
            if stops.stopped:
                yield text, "stop"
                return
            if out.finish_reason is not None:
                usage.cached_tokens = out.cached_tokens
                yield text + stops.flush(), out.finish_reason
                return
            if text:
                yield text, None

    async def collect_text(
        self,
        prompt_ids: list[int],
        params: SamplingParams,
        request: _GenerationRequest,
        usage: _Usage,
    ) -> tuple[str, str | None]:
        """Generate to completion; return the full text and the finish reason."""
        parts: list[str] = []
        finish_reason = None
        async for text, reason in self.stream_text(prompt_ids, params, request, usage):
            parts.append(text)
            finish_reason = reason
        return "".join(parts), finish_reason

    # ----------------------------------------------------------- completions

    async def completions(self, request: CompletionRequest) -> Response:
        """Handle ``POST /v1/completions``."""
        self.check_model(request.model)
        prompt = request.prompt
        if isinstance(prompt, str):
            prompts = [await self.encode(prompt)]
        elif prompt and isinstance(prompt[0], int):
            prompts = [prompt]
        elif prompt and isinstance(prompt[0], str):
            prompts = [await self.encode(text) for text in prompt]
        else:
            prompts = list(prompt)
        if not prompts or any(len(p) == 0 for p in prompts):
            raise APIError(400, "prompt must not be empty")
        params = [self.sampling(request, len(p), request.max_tokens) for p in prompts]
        usages = [_Usage(len(p)) for p in prompts]
        request_id = f"cmpl-{uuid.uuid4().hex}"
        base = {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_name,
        }

        def total_usage() -> dict[str, Any]:
            total = _Usage(
                sum(u.prompt_tokens for u in usages),
                sum(u.completion_tokens for u in usages),
                sum(u.cached_tokens for u in usages),
            )
            return total.payload()

        if request.stream:
            return self._sse_response(
                self._stream_completions(request, prompts, params, usages, base, total_usage)
            )

        async def collect(index: int) -> dict[str, Any]:
            text, reason = await self.collect_text(
                prompts[index], params[index], request, usages[index]
            )
            return {"index": index, "text": text, "logprobs": None, "finish_reason": reason}

        choices = await asyncio.gather(*(collect(i) for i in range(len(prompts))))
        return _json({**base, "choices": list(choices), "usage": total_usage()})

    async def _stream_completions(
        self,
        request: CompletionRequest,
        prompts: list[list[int]],
        params: list[SamplingParams],
        usages: list[_Usage],
        base: dict[str, Any],
        total_usage: Callable[[], dict[str, Any]],
    ) -> AsyncIterator[bytes]:
        async def one(index: int) -> AsyncIterator[bytes]:
            async for text, reason in self.stream_text(
                prompts[index], params[index], request, usages[index]
            ):
                choice = {"index": index, "text": text, "logprobs": None, "finish_reason": reason}
                yield _sse({**base, "choices": [choice]})

        if len(prompts) == 1:
            async for chunk in one(0):
                yield chunk
        else:
            async for chunk in _merge([one(i) for i in range(len(prompts))]):
                yield chunk
        if request.stream_options and request.stream_options.include_usage:
            yield _sse({**base, "choices": [], "usage": total_usage()})
        yield _SSE_DONE

    # ------------------------------------------------------------------ chat

    async def chat_completions(self, request: ChatCompletionRequest) -> Response:
        """Handle ``POST /v1/chat/completions``."""
        self.check_model(request.model)
        try:
            text = self.tokenizer.apply_chat_template(
                [_text_message(message) for message in request.messages],
                tools=request.tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except APIError:
            raise
        except Exception as error:
            raise APIError(400, f"could not apply the chat template: {error}") from error
        prompt = await self.encode(text)
        params = self.sampling(request, len(prompt), request.output_limit)
        usage = _Usage(len(prompt))
        base = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk" if request.stream else "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
        }
        if request.stream:
            return self._sse_response(self._stream_chat(request, prompt, params, usage, base))
        text, reason = await self.collect_text(prompt, params, request, usage)
        message = {"role": "assistant", "content": text}
        choice = {"index": 0, "message": message, "logprobs": None, "finish_reason": reason}
        return _json({**base, "choices": [choice], "usage": usage.payload()})

    async def _stream_chat(
        self,
        request: ChatCompletionRequest,
        prompt: list[int],
        params: SamplingParams,
        usage: _Usage,
        base: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        def chunk(delta: dict[str, Any], reason: str | None) -> bytes:
            choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": reason}
            return _sse({**base, "choices": [choice]})

        yield chunk({"role": "assistant", "content": ""}, None)
        async for text, reason in self.stream_text(prompt, params, request, usage):
            if text:
                yield chunk({"content": text}, None)
            if reason is not None:
                yield chunk({}, reason)
        if request.stream_options and request.stream_options.include_usage:
            yield _sse({**base, "choices": [], "usage": usage.payload()})
        yield _SSE_DONE

    def _sse_response(self, body: AsyncIterator[bytes]) -> StreamingResponse:
        async def guarded() -> AsyncIterator[bytes]:
            try:
                async for chunk in body:
                    yield chunk
            except (ValueError, EngineDeadError) as error:
                # Headers are already sent; report in-band, as OpenAI-compatible servers do.
                yield _sse({"error": {"message": str(error), "type": "server_error"}})
                yield _SSE_DONE

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(guarded(), media_type="text/event-stream", headers=headers)

    # --------------------------------------------------------------- metrics

    def metrics(self) -> str:
        """The engine's latest snapshot in Prometheus text exposition format."""
        snapshot = dict(self.async_engine.metrics)
        now, generated = time.monotonic(), snapshot.get("generation_tokens_total", 0)
        then, generated_then = self._rate_mark
        self._rate_mark = (now, generated)
        snapshot["generation_tokens_per_second"] = (generated - generated_then) / (now - then)
        lines = []
        for name, value in snapshot.items():
            kind = "counter" if name.endswith("_total") else "gauge"
            lines += [
                f"# HELP minfer_{name} {_METRIC_HELP.get(name, name)}",
                f"# TYPE minfer_{name} {kind}",
                f"minfer_{name} {value}",
            ]
        return "\n".join(lines) + "\n"


async def _merge(streams: list[AsyncIterator[bytes]]) -> AsyncIterator[bytes]:
    """Interleave several async iterators as their items become available."""
    merged: asyncio.Queue[tuple[bool, bytes | BaseException | None]] = asyncio.Queue()

    async def pump(stream: AsyncIterator[bytes]) -> None:
        try:
            async for item in stream:
                merged.put_nowait((False, item))
        except Exception as error:
            merged.put_nowait((True, error))
        else:
            merged.put_nowait((True, None))

    tasks = [asyncio.ensure_future(pump(stream)) for stream in streams]
    try:
        remaining = len(tasks)
        while remaining:
            done, item = await merged.get()
            if not done:
                yield item
            elif item is not None:
                raise item
            else:
                remaining -= 1
    finally:
        for task in tasks:
            task.cancel()


def create_app(engine_factory: Callable[[], Engine], tokenizer: Any, model_name: str) -> FastAPI:
    """Build the FastAPI application.

    Args:
        engine_factory: Picklable callable that builds the engine; it is run in the engine
            process when the application starts.
        tokenizer: The model's tokenizer (chat template, encoding, incremental decoding).
        model_name: The name clients address the model by.
    """
    server = Server(engine_factory, tokenizer, model_name)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await server.async_engine.start()
        yield
        await server.async_engine.stop()

    app = FastAPI(title="minfer", lifespan=lifespan)
    app.state.server = server

    @app.exception_handler(APIError)
    async def _api_error(_: Request, error: APIError) -> Response:
        return _error(error.status, error.message, error.kind)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, error: RequestValidationError) -> Response:
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}"
            for e in error.errors()
        )
        return _error(400, detail, "invalid_request_error")

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, error: ValueError) -> Response:
        return _error(400, str(error), "invalid_request_error")

    @app.exception_handler(EngineDeadError)
    async def _engine_dead(_: Request, error: EngineDeadError) -> Response:
        return _error(500, str(error), "server_error")

    @app.get("/health")
    async def health() -> Response:
        if not server.async_engine.healthy:
            return _error(503, "engine is not running", "server_error")
        return _json({"status": "ok"})

    @app.get("/v1/models")
    async def models() -> Response:
        card = {
            "id": model_name,
            "object": "model",
            "created": server.created,
            "owned_by": "minfer",
            "max_model_len": server.max_model_len,
        }
        return _json({"object": "list", "data": [card]})

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(server.metrics(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest) -> Response:
        return await server.completions(request)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Response:
        return await server.chat_completions(request)

    @app.post("/tokenize")
    async def tokenize(request: TokenizeRequest) -> Response:
        server.check_model(request.model)
        tokens = await server.encode(request.prompt, request.add_special_tokens)
        return _json(
            {"tokens": tokens, "count": len(tokens), "max_model_len": server.max_model_len}
        )

    @app.post("/detokenize")
    async def detokenize(request: DetokenizeRequest) -> Response:
        server.check_model(request.model)
        return _json(
            {"prompt": server.fast_tokenizer.decode(request.tokens, skip_special_tokens=False)}
        )

    return app
