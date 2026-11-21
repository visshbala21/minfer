"""Real coding-agent traces: token sequences from Open-SWE-Traces, pauses from TraceLab.

    .venv/bin/python -m bench.real_traces latency --tracelab syfi_coding_trace.jsonl.gz
    .venv/bin/python -m bench.real_traces compile --out bench/traces/real_7b.jsonl
    .venv/bin/python -m bench.real_traces stats bench/traces/real_7b.jsonl
    .venv/bin/python -m bench.real_traces replay bench/traces/real_7b.jsonl --concurrency 8

No public dataset has both halves of a real agent workload, so two are combined:

* **Content and turn structure** come from ``nvidia/Open-SWE-Traces`` (CC BY 4.0): complete
  trajectories of coding agents solving repository issues — system prompt, tool
  definitions, every assistant message with its tool calls, every tool result. The
  ``claudecode`` split is used: its tools (Bash, Read, Edit) are Claude Code's. The
  dataset has no timestamps.
* **Tool-call pause durations** come from TraceLab (github.com/uw-syfi/TraceLab,
  arXiv 2606.30560, data CC BY 4.0): timings of ~430k tool calls in real Claude Code and
  Codex sessions. Its release has no prompt text, only sizes and timings. The pause after
  a model call is sampled from TraceLab's empirical latency distribution of the *same
  tool name* (Claude Code sessions only); a step with several calls waits for the slowest.
  ``bench/data/tracelab_tool_latency.json`` is the committed quantile table derived from
  the release (the ``latency`` command regenerates it).

A model call of the trace is replayed exactly as the agent made it: the request is the
chat-templated conversation so far (system prompt, tool schemas, all earlier turns with
their real tool outputs), and the server is asked for as many tokens as the recorded
assistant message has. As in ``bench/agent_workload.py`` the history always contains the
*recorded* assistant messages, so every engine sees byte-identical prompts.

Trajectories are cut at the model's context limit: a session keeps the model calls whose
prompt and output fit, and ends there. Sessions run closed-loop at a fixed concurrency
(when one ends the next starts), which is the knob that sets KV memory pressure.

``stats`` classifies every prompt token of the trace as *prefix* (reusable by an exact
prefix cache of unlimited size), *shifted duplicate* (not in the cached prefix, but inside
a window of tokens seen before at another position — reusable only by position-independent
caching) or *novel*, using rolling-hash fingerprints.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import itertools
import json
import random
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bench import agent_workload as aw
from bench import models

TRACE_VERSION = 1
DATASET = "nvidia/Open-SWE-Traces"
DEFAULT_SHARD = "data/claudecode/qwen38_flash_next/scale-swe/train-00006-of-00007.parquet"
LATENCY_TABLE = Path(__file__).parent / "data" / "tracelab_tool_latency.json"
TRACELAB_RELEASE = (
    "https://github.com/uw-syfi/TraceLab/releases/download/v0.0.1/syfi_coding_trace.jsonl.gz"
)
NUM_QUANTILES = 201
MIN_TOOL_SAMPLES = 500
FINGERPRINT_WINDOW = 32
BLOCK = 16  # prefix granularity of the statistics: the engines' KV block size
_Q = np.uint64(0x9E3779B97F4A7C15)  # odd, so invertible modulo 2**64
_Q_INV = np.uint64(pow(int(_Q), -1, 1 << 64))


# ------------------------------------------------------------------- tool-call latencies


def build_latency_table(tracelab: Path) -> dict[str, Any]:
    """Per-tool quantiles of tool-call latency from the TraceLab release (Claude sessions).

    A call's latency is TraceLab's own "effective tool latency": the tool's internal
    latency when the harness reported one, otherwise the wall-clock time between the call
    being emitted and its result arriving.
    """
    samples: dict[str, list[float]] = defaultdict(list)
    digest = hashlib.sha256()
    with tracelab.open("rb") as raw:
        for chunk in iter(lambda: raw.read(1 << 20), b""):
            digest.update(chunk)
    with gzip.open(tracelab, "rt") as lines:
        for line in lines:
            row = json.loads(line)
            if row.get("provider") != "claude":
                continue
            for tool in row.get("tools") or ():
                latency = tool.get("tool_internal_latency_ms")
                if latency is None:
                    latency = tool.get("tool_wall_latency_ms")
                if latency is not None and latency >= 0:
                    samples[tool["tool_name"]].append(latency / 1e3)
    everything = np.concatenate([np.asarray(values) for values in samples.values()])
    grid = np.linspace(0.0, 1.0, NUM_QUANTILES)
    tools = {
        name: {"calls": len(values), "quantiles_s": np.quantile(values, grid).round(4).tolist()}
        for name, values in sorted(samples.items())
        if len(values) >= MIN_TOOL_SAMPLES
    }
    tools["*"] = {
        "calls": len(everything),
        "quantiles_s": np.quantile(everything, grid).round(4).tolist(),
    }
    return {
        "source": TRACELAB_RELEASE,
        "source_sha256": digest.hexdigest(),
        "license": "CC BY 4.0 (TraceLab dataset; Zhu et al., arXiv 2606.30560)",
        "what": (
            f"{NUM_QUANTILES} evenly spaced quantiles of effective tool-call latency in "
            "seconds, per tool name, Claude Code sessions; '*' pools every tool"
        ),
        "tools": tools,
    }


class PauseSampler:
    """Samples tool-call latencies from the committed TraceLab quantile table."""

    def __init__(self, table: dict[str, Any], max_pause: float, time_scale: float) -> None:
        self.quantiles = {
            name: np.asarray(entry["quantiles_s"]) for name, entry in table["tools"].items()
        }
        self.grid = np.linspace(0.0, 1.0, NUM_QUANTILES)
        self.max_pause = max_pause
        self.time_scale = time_scale

    def sample(self, tool_names: list[str], rng: random.Random) -> float:
        """Pause after a step that calls ``tool_names`` (run in parallel: the slowest wins)."""
        longest = 0.0
        for name in tool_names:
            quantiles = self.quantiles.get(name, self.quantiles["*"])
            longest = max(longest, float(np.interp(rng.random(), self.grid, quantiles)))
        return min(longest, self.max_pause) * self.time_scale


# ----------------------------------------------------------------------------- the trace


@dataclass(frozen=True)
class Call:
    """One model call of a real session.

    Attributes:
        message: Index of the assistant message this call produced; the request is the
            conversation before it.
        prompt_tokens: Tokens of the request.
        max_tokens: Tokens of the recorded assistant message (what the server is asked for).
        tools: Names of the tools the recorded message calls (empty: a final answer).
        pause: Seconds from this call's completion to the next call of the session.
    """

    message: int
    prompt_tokens: int
    max_tokens: int
    tools: list[str]
    pause: float


@dataclass(frozen=True)
class RealSession:
    """A recorded agent session: its conversation and the model calls replayed from it."""

    session_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    calls: list[Call]
    truncated: bool


@dataclass
class RealTrace:
    """Sessions compiled for one tokenizer and context limit."""

    header: dict[str, Any]
    sessions: list[RealSession] = field(default_factory=list)

    @property
    def num_requests(self) -> int:
        """Total model calls in the trace."""
        return sum(len(session.calls) for session in self.sessions)

    def to_jsonl(self) -> str:
        """Serialize: a header line, then one line per session."""
        lines = [json.dumps({"type": "header", **self.header}, ensure_ascii=False)]
        lines += [
            json.dumps({"type": "session", **asdict(session)}, ensure_ascii=False)
            for session in self.sessions
        ]
        return "\n".join(lines) + "\n"

    @property
    def sha256(self) -> str:
        """Content hash of the serialized trace (recorded with every result)."""
        return hashlib.sha256(self.to_jsonl().encode()).hexdigest()

    def save(self, path: Path) -> None:
        """Write the trace as JSONL."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_jsonl())

    @classmethod
    def load(cls, path: Path, max_sessions: int | None = None) -> RealTrace:
        """Read a trace written by :meth:`save` (optionally only its first sessions)."""
        sessions = []
        with path.open() as lines:
            header = json.loads(next(lines))
            if header.pop("type") != "header" or header.get("version") != TRACE_VERSION:
                raise ValueError(f"{path} is not a version-{TRACE_VERSION} real trace")
            for line in lines:
                if max_sessions is not None and len(sessions) >= max_sessions:
                    break
                row = json.loads(line)
                calls = [Call(**call) for call in row["calls"]]
                sessions.append(
                    RealSession(
                        row["session_id"], row["messages"], row["tools"], calls, row["truncated"]
                    )
                )
        return cls(header, sessions)


def convert_trajectory(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """An Open-SWE-Traces row as OpenAI-style chat messages and tool definitions."""
    tools = []
    for raw in row["tools"]:
        spec = json.loads(raw)
        function = {
            "name": spec["name"],
            "description": spec.get("description", ""),
            "parameters": spec.get("input_schema", {}),
        }
        tools.append({"type": "function", "function": function})
    messages = []
    for message in row["messages"]:
        converted: dict[str, Any] = {"role": message["role"], "content": message["content"] or ""}
        if message.get("tool_calls"):
            converted["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": _arguments(call["function"]["arguments"]),
                    },
                }
                for call in message["tool_calls"]
            ]
        messages.append(converted)
    return messages, tools


def _arguments(raw: str) -> Any:
    """Tool-call arguments as an object (the dataset stores a JSON string)."""
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {"raw": raw}


def render(session: RealSession, call: Call, tokenizer: Any) -> str:
    """The prompt text of ``call``: the chat-templated conversation before its message."""
    return tokenizer.apply_chat_template(
        session.messages[: call.message],
        tools=session.tools,
        tokenize=False,
        add_generation_prompt=True,
    )


def compile_session(
    session_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tokenizer: Any,
    max_context: int,
    sampler: PauseSampler,
    rng: random.Random,
) -> RealSession | None:
    """Turn one trajectory into replayable model calls that fit ``max_context``.

    Returns ``None`` for a trajectory whose very first call does not fit.
    """
    indices = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
    prompts, completions = [], []
    for index in indices:
        prompt = tokenizer.apply_chat_template(
            messages[:index], tools=tools, tokenize=False, add_generation_prompt=True
        )
        through = tokenizer.apply_chat_template(
            messages[: index + 1], tools=tools, tokenize=False, add_generation_prompt=False
        )
        if not through.startswith(prompt):
            raise ValueError(f"{session_id}: the chat template is not append-only")
        prompts.append(prompt)
        completions.append(through[len(prompt) :])
    lengths = [len(ids) for ids in tokenizer(prompts, add_special_tokens=False)["input_ids"]]
    outputs = [len(ids) for ids in tokenizer(completions, add_special_tokens=False)["input_ids"]]
    calls: list[Call] = []
    for index, prompt_tokens, output_tokens in zip(indices, lengths, outputs, strict=True):
        if prompt_tokens + output_tokens > max_context:
            break
        names = [call["function"]["name"] for call in messages[index].get("tool_calls", ())]
        calls.append(Call(index, prompt_tokens, max(output_tokens, 1), names, 0.0))
    if not calls:
        return None
    truncated = len(calls) < len(indices)
    # The pause follows a call only if the session goes on to make another one.
    paused = [
        Call(c.message, c.prompt_tokens, c.max_tokens, c.tools, sampler.sample(c.tools, rng))
        for c in calls[:-1]
    ]
    last = calls[-1]
    kept = messages[: last.message + 1]
    return RealSession(session_id, kept, tools, [*paused, last], truncated)


def compile_trace(
    parquet: Path,
    tokenizer: Any,
    model: str,
    max_context: int,
    max_sessions: int,
    min_calls: int,
    max_pause: float,
    time_scale: float,
    seed: int,
) -> RealTrace:
    """Compile the first ``max_sessions`` usable trajectories of a parquet shard."""
    import pyarrow.parquet as pq

    sampler = PauseSampler(json.loads(LATENCY_TABLE.read_text()), max_pause, time_scale)
    header = {
        "version": TRACE_VERSION,
        "dataset": DATASET,
        "shard": parquet.name,
        "model": model,
        "max_context": max_context,
        "min_calls": min_calls,
        "max_pause_s": max_pause,
        "time_scale": time_scale,
        "seed": seed,
        "pauses": "TraceLab v0.0.1 per-tool latency quantiles (bench/data)",
    }
    trace = RealTrace(header)
    file = pq.ParquetFile(parquet)
    for group in range(file.metadata.num_row_groups):
        for row in file.read_row_group(group).to_pylist():
            if len(trace.sessions) >= max_sessions:
                return trace
            messages, tools = convert_trajectory(row)
            rng = random.Random(f"{seed}-{row['trajectory_id']}")
            session = compile_session(
                row["trajectory_id"], messages, tools, tokenizer, max_context, sampler, rng
            )
            if session is not None and len(session.calls) >= min_calls:
                trace.sessions.append(session)
    return trace


# ----------------------------------------------------------------------------- statistics


def _prefix_sums(tokens: np.ndarray) -> np.ndarray:
    """``C[k] = Σ_{j<k} (t_j + 1) · q^j  (mod 2**64)``: a fingerprint of the first k tokens."""
    powers = np.ones(len(tokens), dtype=np.uint64)
    np.multiply.accumulate(np.full(len(tokens) - 1, _Q, dtype=np.uint64), out=powers[1:])
    sums = np.zeros(len(tokens) + 1, dtype=np.uint64)
    np.cumsum((tokens.astype(np.uint64) + np.uint64(1)) * powers, out=sums[1:])
    return sums


def _window_hashes(sums: np.ndarray, window: int) -> np.ndarray:
    """Position-independent fingerprints of every ``window``-token run."""
    count = len(sums) - window
    if count <= 0:
        return np.zeros(0, dtype=np.uint64)
    shift = np.ones(count, dtype=np.uint64)
    np.multiply.accumulate(np.full(count - 1, _Q_INV, dtype=np.uint64), out=shift[1:])
    return (sums[window:] - sums[:-window]) * shift


class ReuseClassifier:
    """Classifies prompt tokens as prefix / shifted duplicate / novel, in replay order.

    State is everything "seen so far": fingerprints of every block-aligned prefix (what an
    exact prefix cache of unlimited size could serve) and of every ``window``-token run
    (what a position-independent cache could additionally serve).
    """

    def __init__(self, window: int = FINGERPRINT_WINDOW, block: int = BLOCK) -> None:
        self.window = window
        self.block = block
        self.prefixes: set[int] = set()
        self.windows: set[int] = set()

    def classify(self, tokens: np.ndarray) -> tuple[int, int, int]:
        """``(prefix, shifted duplicate, novel)`` token counts of one prompt; then learn it."""
        total = len(tokens)
        sums = _prefix_sums(tokens)
        boundaries = sums[self.block :: self.block]
        cached_blocks = 0
        for value in boundaries.tolist():  # cached prefixes are downward closed
            if value not in self.prefixes:
                break
            cached_blocks += 1
        prefix = cached_blocks * self.block
        hashes = _window_hashes(sums, self.window)
        tail = hashes[prefix:].tolist()
        covered = np.zeros(total - prefix + 1, dtype=np.int64)
        for offset, value in enumerate(tail):
            if value in self.windows:
                covered[offset] += 1
                covered[offset + self.window] -= 1
        shifted = int((np.cumsum(covered[:-1]) > 0).sum())
        self.prefixes.update(boundaries[cached_blocks:].tolist())
        self.windows.update(tail)
        return prefix, shifted, total - prefix - shifted


def _percentiles(values: list[float], points: tuple[int, ...] = (50, 90, 99)) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    summary = {f"p{p}": float(np.percentile(array, p)) for p in points}
    return {"mean": float(array.mean()), **summary, "max": float(array.max())}


def trace_statistics(trace: RealTrace, tokenizer: Any) -> dict[str, Any]:
    """Shape of the trace and how much of it each kind of KV reuse could serve.

    Sessions are processed one after another (the order a single-stream replay would see
    them), so "seen before" includes earlier sessions: a shared system prompt counts as
    prefix from the second session on.
    """
    classifier = ReuseClassifier()
    totals = np.zeros(3, dtype=np.int64)
    first_call = np.zeros(3, dtype=np.int64)
    later_calls = np.zeros(3, dtype=np.int64)
    for session in trace.sessions:
        texts = [render(session, call, tokenizer) for call in session.calls]
        for turn, ids in enumerate(tokenizer(texts, add_special_tokens=False)["input_ids"]):
            counts = np.asarray(classifier.classify(np.asarray(ids, dtype=np.int64)))
            totals += counts
            (first_call if turn == 0 else later_calls)[:] += counts
    calls = [call for session in trace.sessions for call in session.calls]
    pauses = [call.pause for session in trace.sessions for call in session.calls[:-1]]
    tools: dict[str, int] = defaultdict(int)
    for call in calls:
        for name in call.tools:
            tools[name] += 1

    def fractions(counts: np.ndarray) -> dict[str, float]:
        share = counts / max(1, counts.sum())
        return {"prefix": share[0], "shifted_duplicate": share[1], "novel": share[2]}

    final = [session.calls[-1] for session in trace.sessions]
    return {
        "trace_sha256": trace.sha256,
        "sessions": len(trace.sessions),
        "sessions_truncated_at_context_limit": sum(s.truncated for s in trace.sessions),
        "model_calls": len(calls),
        "calls_per_session": _percentiles([len(s.calls) for s in trace.sessions]),
        "prompt_tokens": _percentiles([c.prompt_tokens for c in calls]),
        "final_context_tokens": _percentiles([c.prompt_tokens + c.max_tokens for c in final]),
        "appended_tokens_per_call": _percentiles(
            [
                later.prompt_tokens - earlier.prompt_tokens
                for s in trace.sessions
                for earlier, later in itertools.pairwise(s.calls)
            ]
        ),
        "output_tokens": _percentiles([c.max_tokens for c in calls]),
        "total_prompt_tokens": int(totals.sum()),
        "total_output_tokens": sum(c.max_tokens for c in calls),
        "tool_calls": dict(sorted(tools.items(), key=lambda item: -item[1])),
        "pause_seconds": _percentiles(pauses, (10, 25, 50, 75, 90, 99)),
        "pause_seconds_fraction_over": {
            f"{limit}s": float(np.mean(np.asarray(pauses) > limit)) for limit in (1, 2, 5, 10, 30)
        },
        "fingerprint_window_tokens": FINGERPRINT_WINDOW,
        "prefix_block_tokens": BLOCK,
        "token_fractions": fractions(totals),
        "token_fractions_first_call": fractions(first_call),
        "token_fractions_later_calls": fractions(later_calls),
    }


# --------------------------------------------------------------------------------- replay


@dataclass
class ReplayOptions:
    """How a trace is replayed.

    Attributes:
        concurrency: Sessions in flight at once (closed loop).
        session_hints: Send minfer's session fields with every request: the session id and,
            for a call the recording follows with tool calls, the tools about to run. A
            server that parses its own output knows the latter; the replay has to say it
            because the history is scripted and the output is not what is replayed.
        extra_body: Extra request fields for every request.
        max_sessions: Stop after this many sessions (``None``: the whole trace).
        duration: Stop sending requests this many seconds into the replay (requests in
            flight complete). ``None``: until the sessions run out.
        ramp: Seconds over which the workers' first sessions are started, evenly spaced,
            instead of all at once.
    """

    concurrency: int
    session_hints: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    max_sessions: int | None = None
    duration: float | None = None
    ramp: float = 0.0


async def replay(
    trace: RealTrace,
    tokenizer: Any,
    base_url: str,
    model: str,
    options: ReplayOptions,
    on_session_done: Callable[[int], None] | None = None,
) -> tuple[list[aw.RequestRecord], float]:
    """Replay ``trace`` closed-loop against an OpenAI-compatible server.

    Returns the per-request records and the wall-clock duration.
    """
    import aiohttp

    url = f"{base_url.rstrip('/')}/v1/completions"
    sessions = trace.sessions[: options.max_sessions]
    queue: Iterator[RealSession] = iter(sessions)
    records: list[aw.RequestRecord] = []
    origin = time.perf_counter()
    finished = 0

    def expired() -> bool:
        return options.duration is not None and time.perf_counter() - origin >= options.duration

    async def worker(index: int, http: aiohttp.ClientSession) -> None:
        nonlocal finished
        await asyncio.sleep(options.ramp * index / options.concurrency)
        for session in queue:
            for turn, call in enumerate(session.calls):
                if expired():
                    return
                body = {
                    "model": model,
                    "prompt": render(session, call, tokenizer),
                    "max_tokens": call.max_tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    **options.extra_body,
                }
                if options.session_hints:
                    last = turn == len(session.calls) - 1
                    body["session_id"] = session.session_id
                    body["session_tools"] = [] if last else (call.tools or ["*"])
                    body["session_final"] = last
                record = aw.RequestRecord(session.session_id, "real", turn, 0.0)
                records.append(record)
                await aw._one_request(http, url, body, record, origin)
                if record.error is not None:
                    break  # the rest of the session depends on this call
                await asyncio.sleep(call.pause)
            finished += 1
            if on_session_done is not None:
                on_session_done(finished)

    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http:
        await asyncio.gather(*(worker(index, http) for index in range(options.concurrency)))
    return records, time.perf_counter() - origin


def summarize(
    records: list[aw.RequestRecord], duration: float, warmup: float = 0.0
) -> dict[str, Any]:
    """The agent workload's summary plus what matters under memory pressure.

    Only requests sent at least ``warmup`` seconds into the replay are counted (rates are
    over the remaining time). ``prefill_tokens_computed`` is the prompt tokens the server
    reported as *not* served from its cache — the work its KV management failed to avoid —
    and ``cache_hit_rate`` the share it did serve; ``sessions`` counts distinct sessions
    that sent a counted request.
    """
    counted = [r for r in records if r.sent >= warmup]
    summary = aw.summarize(counted, duration - warmup)
    good = [r for r in counted if r.error is None and r.cached_tokens is not None]
    prompt = sum(r.prompt_tokens for r in good)
    computed = sum(r.prompt_tokens - r.cached_tokens for r in good)
    summary["prefill_tokens_computed"] = computed
    summary["cache_hit_rate"] = 1.0 - computed / prompt if prompt else None
    summary["sessions"] = len({r.session_id for r in counted})
    summary["warmup_s"] = warmup
    return summary


# ------------------------------------------------------------------------------------ CLI


def _tokenizer(model: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    latency = commands.add_parser("latency", help="rebuild the TraceLab tool-latency table")
    latency.add_argument("--tracelab", type=Path, required=True, help="syfi_coding_trace.jsonl.gz")
    latency.add_argument("--out", type=Path, default=LATENCY_TABLE)

    build = commands.add_parser("compile", help="compile a trace for a model")
    build.add_argument("--model", default=models.PRIMARY, help="model id, or 7b / 1.5b")
    build.add_argument("--parquet", type=Path, default=None, help="an Open-SWE-Traces shard")
    build.add_argument("--max-context", type=int, default=32_000)
    build.add_argument("--max-sessions", type=int, default=300)
    build.add_argument("--min-calls", type=int, default=3)
    build.add_argument("--max-pause", type=float, default=120.0, help="clip pauses (seconds)")
    build.add_argument("--time-scale", type=float, default=1.0, help="multiply every pause")
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--out", type=Path, required=True)

    stats = commands.add_parser("stats", help="trace statistics and reuse fractions")
    stats.add_argument("trace", type=Path)
    stats.add_argument("--out", type=Path, default=None)

    run = commands.add_parser("replay", help="replay a trace against an endpoint")
    run.add_argument("trace", type=Path)
    run.add_argument("--base-url", default="http://localhost:8000")
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--max-sessions", type=int, default=None)
    run.add_argument("--session-hints", action="store_true")
    run.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.command == "latency":
        table = build_latency_table(args.tracelab)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(table) + "\n")
        for name, entry in table["tools"].items():
            q = entry["quantiles_s"]
            print(f"{name:24s} {entry['calls']:7d} calls  p50 {q[100]:.2f}s  p90 {q[180]:.2f}s")
        return
    if args.command == "compile":
        model = models.resolve(args.model)
        parquet = args.parquet
        if parquet is None:
            from huggingface_hub import hf_hub_download

            parquet = Path(hf_hub_download(DATASET, DEFAULT_SHARD, repo_type="dataset"))
        trace = compile_trace(
            parquet,
            _tokenizer(model),
            model,
            args.max_context,
            args.max_sessions,
            args.min_calls,
            args.max_pause,
            args.time_scale,
            args.seed,
        )
        trace.save(args.out)
        print(f"{len(trace.sessions)} sessions, {trace.num_requests} model calls -> {args.out}")
        print(f"sha256 {trace.sha256}")
        return
    trace = RealTrace.load(args.trace, getattr(args, "max_sessions", None))
    tokenizer = _tokenizer(trace.header["model"])
    if args.command == "stats":
        result = trace_statistics(trace, tokenizer)
    else:
        options = ReplayOptions(args.concurrency, args.session_hints)
        records, duration = asyncio.run(
            replay(trace, tokenizer, args.base_url, trace.header["model"], options)
        )
        result = {"trace_sha256": trace.sha256, **summarize(records, duration)}
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
