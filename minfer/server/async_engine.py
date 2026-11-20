"""Run the engine in its own process and expose it to asyncio.

The engine loop is CPU-bound Python between GPU calls, and so is an HTTP server streaming
thousands of chunks per second. In one process they share the interpreter lock: measured
with 300 requests at 30 req/s, an engine step took 26 ms next to the event loop versus
8.5 ms alone (DESIGN.md B-002). The engine therefore gets a process — and an interpreter —
of its own, as in vLLM and SGLang.

The two processes talk over a pair of pipes. Requests travel as token ids; results come
back as one message per engine step carrying every token that step produced, so the
cross-process cost is per step, not per token.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import time
from collections.abc import AsyncIterator, Callable, Sequence
from multiprocessing.connection import Connection
from typing import Any

from minfer.engine import Engine, StepOutput
from minfer.sampling import SamplingParams

_METRICS_INTERVAL = 0.25


class EngineDeadError(RuntimeError):
    """The engine process failed or exited; the server cannot generate anymore."""


def engine_main(factory: Callable[[], Engine], commands: Connection, results: Connection) -> None:
    """Entry point of the engine process: build the engine, then serve commands forever.

    Messages sent on ``results``: ``("ready", info)`` once, ``("tokens", [(request_id,
    token_id, finish_reason, cached_tokens), ...])`` per step, ``("error", request_id,
    message)`` for a rejected request, ``("metrics", snapshot)`` periodically, and
    ``("dead", message)`` before exiting on a failure.
    """
    try:
        engine = factory()
    except Exception as error:
        results.send(("dead", f"engine failed to start: {error!r}"))
        raise
    results.send(("ready", {"max_model_len": engine.max_model_len}))
    request_of: dict[int, int] = {}
    seq_of: dict[int, int] = {}
    last_metrics = 0.0
    try:
        while True:
            if not engine.has_work:
                results.send(("metrics", engine.metrics()))
                commands.poll(None)
            while commands.poll():
                command = commands.recv()
                kind = command[0]
                if kind == "add":
                    _, request_id, prompt_ids, params = command
                    try:
                        seq_id = engine.add_request(prompt_ids, params)
                    except ValueError as error:
                        results.send(("error", request_id, str(error)))
                    else:
                        request_of[seq_id] = request_id
                        seq_of[request_id] = seq_id
                elif kind == "abort":
                    seq_id = seq_of.pop(command[1], None)
                    if seq_id is not None:
                        del request_of[seq_id]
                        engine.abort(seq_id)
                else:  # "stop"
                    return
            if not engine.has_work:
                continue
            outputs = engine.step()
            if outputs:
                batch = []
                for out in outputs:
                    request_id = request_of[out.seq_id]
                    if out.finish_reason is not None:
                        del request_of[out.seq_id], seq_of[request_id]
                    batch.append((request_id, out.token_id, out.finish_reason, out.cached_tokens))
                results.send(("tokens", batch))
            now = time.monotonic()
            if now - last_metrics >= _METRICS_INTERVAL:
                last_metrics = now
                results.send(("metrics", engine.metrics()))
    except Exception as error:
        results.send(("dead", f"engine step failed: {error!r}"))
        raise


class AsyncEngine:
    """Asyncio front end of an :class:`Engine` running in a child process.

    Args:
        factory: Picklable zero-argument callable that builds the engine; it runs in the
            child, so the model is loaded there and never in the server process.
    """

    def __init__(self, factory: Callable[[], Engine]) -> None:
        self._factory = factory
        self.error: str | None = None
        self.info: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self._queues: dict[int, asyncio.Queue] = {}
        self._next_id = 0
        self._process: multiprocessing.Process | None = None
        self._commands: Connection | None = None
        self._results: Connection | None = None

    async def start(self) -> None:
        """Spawn the engine process and wait until the model is loaded."""
        context = multiprocessing.get_context("spawn")  # CUDA does not survive a fork
        command_reader, self._commands = context.Pipe(duplex=False)
        self._results, result_writer = context.Pipe(duplex=False)
        self._process = context.Process(
            target=engine_main,
            args=(self._factory, command_reader, result_writer),
            name="minfer-engine",
            daemon=True,
        )
        self._process.start()
        command_reader.close()
        result_writer.close()
        loop = asyncio.get_running_loop()
        try:
            kind, payload = await loop.run_in_executor(None, self._results.recv)
        except EOFError:
            kind, payload = "dead", "engine process exited during start-up"
        if kind != "ready":
            self.error = payload
            raise EngineDeadError(payload)
        self.info = payload
        loop.add_reader(self._results.fileno(), self._on_results)

    async def stop(self) -> None:
        """Ask the engine process to exit and wait for it."""
        if self._process is None:
            return
        loop = asyncio.get_running_loop()
        loop.remove_reader(self._results.fileno())
        if self._process.is_alive():
            self._commands.send(("stop",))
            await loop.run_in_executor(None, self._process.join, 10)
            if self._process.is_alive():
                self._process.terminate()

    @property
    def healthy(self) -> bool:
        """Whether the engine process is running and has not reported a failure."""
        return self.error is None and self._process is not None and self._process.is_alive()

    async def generate(
        self, prompt_ids: Sequence[int], params: SamplingParams
    ) -> AsyncIterator[StepOutput]:
        """Submit a request and yield its tokens as the engine produces them.

        ``StepOutput.seq_id`` carries this front end's request id.

        Raises:
            ValueError: If the engine rejects the request (e.g. the prompt is too long).
            EngineDeadError: If the engine process has failed.

        If the consumer stops iterating early (client disconnect, stop string), the request
        is aborted in the engine.
        """
        if not self.healthy:
            raise EngineDeadError(self.error or "engine process is not running")
        request_id = self._next_id
        self._next_id += 1
        tokens: asyncio.Queue[StepOutput | BaseException] = asyncio.Queue()
        self._queues[request_id] = tokens
        self._commands.send(("add", request_id, list(prompt_ids), params))
        try:
            while True:
                item = await tokens.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item.finish_reason is not None:
                    return
        finally:
            # Still registered means the engine has not finished the request: cancel it.
            if self._queues.pop(request_id, None) is not None and self.healthy:
                self._commands.send(("abort", request_id))

    def _on_results(self) -> None:
        """Event-loop callback: the results pipe is readable."""
        results, queues = self._results, self._queues
        try:
            while results.poll():
                message = results.recv()
                kind = message[0]
                if kind == "tokens":
                    for request_id, token_id, reason, cached in message[1]:
                        tokens = queues.get(request_id)
                        if tokens is None:
                            continue  # aborted by the client while this step was running
                        if reason is not None:
                            del queues[request_id]
                        tokens.put_nowait(StepOutput(request_id, token_id, reason, cached))
                elif kind == "metrics":
                    self.metrics = message[1]
                elif kind == "error":
                    tokens = queues.pop(message[1], None)
                    if tokens is not None:
                        tokens.put_nowait(ValueError(message[2]))
                else:  # "dead"
                    self._fail(message[1])
        except (EOFError, OSError):
            self._fail("engine process exited unexpectedly")

    def _fail(self, reason: str) -> None:
        """Mark the engine dead and fail every in-flight request."""
        self.error = reason
        asyncio.get_running_loop().remove_reader(self._results.fileno())
        queues, self._queues = self._queues, {}
        for tokens in queues.values():
            tokens.put_nowait(EngineDeadError(reason))
