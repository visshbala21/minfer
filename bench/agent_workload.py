"""Synthetic multi-turn agent workload: generator, JSONL traces, and a replay client.

    # write a trace
    python -m bench.agent_workload generate --sessions 100 --rate 4 --out trace.jsonl
    # replay it against any OpenAI-compatible endpoint
    python -m bench.agent_workload replay trace.jsonl --base-url http://localhost:8000

**What a session looks like.** An agent of some type (coding, data analysis, research) is
given a task. Every request carries the agent type's system prompt — instructions plus
JSON tool schemas, about 4k tokens, identical for every session of that type — then the
task, then the whole history so far. Each turn the agent emits a tool call or an answer,
the tool's result (file contents, test logs with stack traces, query results, search hits)
is appended, and the next request resends everything. Sessions last 5–15 turns, so a
session's later requests are mostly tokens the server has already seen.

**Determinism.** Everything is derived from a seed: session contents, session start times
(Poisson), and think times between turns. The assistant's messages in the history are
*scripted* — part of the trace — rather than whatever a server generated, and every request
asks for exactly the scripted number of output tokens (``ignore_eos``). Any two servers
replaying a trace therefore receive byte-identical prompts and produce equally long
outputs, whatever their sampling noise. The cost of that choice: a server cannot reuse the
KV of its *own* generated tokens on the next turn (the history contains the scripted text
instead), which understates what prefix caching gives a real agent loop — equally for
every server.

**Timing.** Sessions arrive open-loop (Poisson); within a session the next turn is sent
``think`` seconds after the previous response completes, like an agent executing a tool.
``rate`` is the nominal request rate: the session arrival rate is ``rate`` divided by the
mean number of turns. ``rate=None`` ("max") starts every session at once with no think
time, which measures saturated throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TRACE_VERSION = 1
MIN_TURNS, MAX_TURNS = 5, 15
MEAN_TURNS = (MIN_TURNS + MAX_TURNS) / 2
SYSTEM_PROMPT_TOKENS = 4000

# --------------------------------------------------------------------------- vocabulary

_MODULES = ["scheduler", "billing", "ingest", "auth", "cache", "router", "export", "notify"]
_NOUNS = ["retry", "batch", "session", "token", "queue", "shard", "index", "lease", "quota"]
_VERBS = ["flush", "merge", "resolve", "rotate", "validate", "rebalance", "expire", "compact"]
_ERRORS = ["KeyError", "TimeoutError", "ValueError", "AssertionError", "ConnectionResetError"]
_TABLES = ["orders", "customers", "events", "invoices", "sessions", "shipments"]
_COLUMNS = ["region", "plan", "channel", "status", "country", "device", "tier"]
_METRICS = ["revenue", "latency_ms", "error_rate", "active_users", "churn", "conversion"]
_TOPICS = [
    "connection pooling",
    "rate limiting",
    "schema migration",
    "cache invalidation",
    "idempotent retries",
    "backpressure",
    "leader election",
    "clock skew",
]
_SITES = ["docs.example.org", "engineering.example.com", "wiki.example.net", "forum.example.io"]
_FILLER = (
    "the service keeps a bounded queue per tenant and drains it in arrival order",
    "a request that exceeds its deadline is cancelled and reported to the caller",
    "configuration is reloaded atomically so that readers never observe a partial update",
    "each worker holds a lease that must be renewed before it expires",
    "writes are acknowledged only after they reach a quorum of replicas",
    "metrics are aggregated in one minute windows and retained for thirty days",
    "the fallback path is exercised by a canary on every deploy",
    "identifiers are opaque strings and must not be parsed by clients",
    "pagination uses a cursor that encodes the position of the last returned row",
    "a circuit breaker opens after five consecutive failures and half-opens after a delay",
)


def _sentences(rng: random.Random, count: int) -> str:
    picked = (rng.choice(_FILLER) for _ in range(count))
    return " ".join(sentence.capitalize() + "." for sentence in picked)


# -------------------------------------------------------------------------- agent types


def _tool(name: str, description: str, **properties: tuple[str, str]) -> dict[str, Any]:
    """A JSON tool schema in the OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": kind, "description": text}
                    for key, (kind, text) in properties.items()
                },
                "required": list(properties)[:1],
            },
        },
    }


_AGENT_TOOLS: dict[str, list[dict[str, Any]]] = {
    "coder": [
        _tool(
            "read_file",
            "Read a file from the repository and return its contents with line numbers.",
            path=("string", "Repository-relative path of the file."),
            start_line=("integer", "First line to return, 1-based."),
            end_line=("integer", "Last line to return, inclusive."),
        ),
        _tool(
            "search_code",
            "Search the repository with a regular expression and return matching lines.",
            pattern=("string", "Regular expression to search for."),
            glob=("string", "Only search files matching this glob."),
        ),
        _tool(
            "run_tests",
            "Run the test suite, or a subset of it, and return the log.",
            target=("string", "Test file, directory or node id to run."),
            verbose=("boolean", "Include captured output of passing tests."),
        ),
        _tool(
            "edit_file",
            "Apply a unified diff to a file in the repository.",
            path=("string", "Repository-relative path of the file to edit."),
            diff=("string", "Unified diff to apply."),
        ),
        _tool(
            "run_shell",
            "Run a shell command in the repository root and return stdout and stderr.",
            command=("string", "The command line to execute."),
            timeout_s=("integer", "Kill the command after this many seconds."),
        ),
    ],
    "analyst": [
        _tool(
            "run_sql",
            "Run a read-only SQL query against the warehouse and return the rows.",
            query=("string", "The SQL query."),
            limit=("integer", "Maximum number of rows to return."),
        ),
        _tool(
            "describe_table",
            "Return the columns, types and row count of a warehouse table.",
            table=("string", "Fully qualified table name."),
        ),
        _tool(
            "run_python",
            "Execute Python in a sandbox with pandas available and return what it prints.",
            code=("string", "Python source to execute."),
        ),
        _tool(
            "plot_chart",
            "Render a chart from a query result and return a link to the image.",
            kind=("string", "Chart type: line, bar or scatter."),
            x=("string", "Column for the x axis."),
            y=("string", "Column for the y axis."),
        ),
    ],
    "researcher": [
        _tool(
            "web_search",
            "Search the web and return the top results with titles, URLs and snippets.",
            query=("string", "The search query."),
            num_results=("integer", "How many results to return."),
        ),
        _tool(
            "open_url",
            "Fetch a web page and return its main text content.",
            url=("string", "The URL to fetch."),
        ),
        _tool(
            "search_notes",
            "Search the team's internal notes and return matching passages.",
            query=("string", "What to look for."),
        ),
        _tool(
            "save_note",
            "Save a finding to the internal notes for later reference.",
            title=("string", "Short title of the note."),
            body=("string", "The note's text."),
        ),
    ],
}

_AGENT_ROLES = {
    "coder": "a senior software engineer working in a large Python monorepo",
    "analyst": "a data analyst answering questions from a company data warehouse",
    "researcher": "a technical researcher who answers questions with cited sources",
}


def system_prompt(agent: str, count_tokens: Callable[[str], int], target: int) -> str:
    """The agent type's instructions, padded with numbered guidelines to ~``target`` tokens.

    Deterministic per agent type (its own fixed seed), so every session of a type shares it.
    """
    rng = random.Random(f"system-prompt-{agent}")
    lines = [
        f"You are {_AGENT_ROLES[agent]}. You work autonomously: plan, call tools, read their "
        "results carefully, and continue until the task is complete.",
        "",
        "# Operating guidelines",
    ]
    tools_tokens = count_tokens(json.dumps(_AGENT_TOOLS[agent], indent=2))
    number = 0
    while count_tokens("\n".join(lines)) + tools_tokens < target:
        number += 1
        lines.append(f"{number}. {_sentences(rng, rng.randint(2, 4))}")
    return "\n".join(lines)


# ------------------------------------------------------------------- content generators


def _python_source(rng: random.Random, lines: int) -> str:
    out = ["from __future__ import annotations", "", "import logging", "", ""]
    while len(out) < lines:
        verb, noun = rng.choice(_VERBS), rng.choice(_NOUNS)
        out += [
            f"def {verb}_{noun}({noun}s: list[dict], limit: int = {rng.randint(8, 512)}) -> int:",
            f'    """{_sentences(rng, 1)}"""',
            "    processed = 0",
            f"    for {noun} in {noun}s[:limit]:",
            f'        if {noun}.get("state") == "{rng.choice(["stale", "pending", "failed"])}":',
            f'            logging.debug("skipping %s", {noun}["id"])',
            "            continue",
            f'        {noun}["attempts"] = {noun}.get("attempts", 0) + 1',
            "        processed += 1",
            "    return processed",
            "",
            "",
        ]
    return "\n".join(f"{i + 1:4d}  {line}" for i, line in enumerate(out[:lines]))


def _test_log(rng: random.Random, failures: int) -> str:
    module = rng.choice(_MODULES)
    passed = rng.randint(20, 180)
    out = [f"collected {passed + failures} items", ""]
    for _ in range(failures):
        verb, noun, error = rng.choice(_VERBS), rng.choice(_NOUNS), rng.choice(_ERRORS)
        line = rng.randint(20, 400)
        out += [
            f"___________________ test_{verb}_{noun}_{rng.randint(1, 9)} ___________________",
            "Traceback (most recent call last):",
            f'  File "tests/{module}/test_{noun}.py", line {line}, in test_{verb}_{noun}',
            f"    result = {verb}_{noun}(fixture_{noun}s, limit={rng.randint(1, 64)})",
            f'  File "src/{module}/{noun}.py", line {rng.randint(10, 300)}, in {verb}_{noun}',
            f'    {noun}["attempts"] = {noun}.get("attempts", 0) + 1',
            f'  File "src/{module}/models.py", line {rng.randint(10, 300)}, in __setitem__',
            f'    raise {error}(f"{noun} {{key!r}} is read-only after {verb}")',
            f"{error}: {noun} 'attempts' is read-only after {verb}",
            "",
        ]
    out.append(
        f"=========== {failures} failed, {passed} passed in {rng.uniform(2, 40):.2f}s ==========="
    )
    return "\n".join(out)


def _grep_hits(rng: random.Random, count: int) -> str:
    rows = []
    for _ in range(count):
        module, noun, verb = rng.choice(_MODULES), rng.choice(_NOUNS), rng.choice(_VERBS)
        rows.append(
            f"src/{module}/{noun}.py:{rng.randint(1, 600)}:    "
            f"{noun}s = {verb}_{noun}({noun}s, limit=config.{noun}_limit)"
        )
    return "\n".join(rows)


def _sql_rows(rng: random.Random, count: int) -> str:
    dims = rng.sample(_COLUMNS, 2)
    metrics = rng.sample(_METRICS, 2)
    rows = [" | ".join([*dims, *metrics]), " | ".join("---" for _ in range(4))]
    for _ in range(count):
        rows.append(
            f"{rng.choice(['emea', 'amer', 'apac', 'latam'])} | "
            f"{rng.choice(['free', 'team', 'enterprise', 'trial'])} | "
            f"{rng.uniform(100, 90000):.2f} | {rng.uniform(0, 1):.4f}"
        )
    return "\n".join(rows) + f"\n({count} rows)"


def _search_hits(rng: random.Random, count: int) -> str:
    hits = []
    for rank in range(1, count + 1):
        topic = rng.choice(_TOPICS)
        slug = topic.replace(" ", "-")
        hits.append(
            f"[{rank}] {topic.title()}: lessons from production\n"
            f"    https://{rng.choice(_SITES)}/{slug}-{rng.randint(100, 999)}\n"
            f"    {_sentences(rng, 2)}"
        )
    return "\n".join(hits)


def _tool_call(name: str, **arguments: Any) -> str:
    call = json.dumps({"name": name, "arguments": arguments})
    return f"<tool_call>\n{call}\n</tool_call>"


def _coder_turn(rng: random.Random) -> tuple[str, str]:
    module, noun = rng.choice(_MODULES), rng.choice(_NOUNS)
    kind = rng.choice(["read", "search", "test", "edit", "shell"])
    thought = _sentences(rng, rng.randint(1, 2))
    if kind == "read":
        call = _tool_call("read_file", path=f"src/{module}/{noun}.py", start_line=1, end_line=120)
        return f"{thought}\n{call}", _python_source(rng, rng.randint(40, 110))
    if kind == "search":
        call = _tool_call(
            "search_code", pattern=f"{rng.choice(_VERBS)}_{noun}\\(", glob="src/**/*.py"
        )
        return f"{thought}\n{call}", _grep_hits(rng, rng.randint(6, 30))
    if kind == "test":
        call = _tool_call("run_tests", target=f"tests/{module}", verbose=False)
        return f"{thought}\n{call}", _test_log(rng, rng.randint(1, 5))
    if kind == "edit":
        diff = (
            f"--- a/src/{module}/{noun}.py\n+++ b/src/{module}/{noun}.py\n@@ -41,3 +41,5 @@\n"
            f'-        {noun}["attempts"] = {noun}.get("attempts", 0) + 1\n'
            f'+        attempts = {noun}.get("attempts", 0) + 1\n'
            f'+        {noun} = {{**{noun}, "attempts": attempts}}\n'
        )
        call = _tool_call("edit_file", path=f"src/{module}/{noun}.py", diff=diff)
        return f"{thought}\n{call}", f"applied 1 hunk to src/{module}/{noun}.py"
    call = _tool_call("run_shell", command=f"git log --oneline -n 12 -- src/{module}", timeout_s=30)
    log = "\n".join(
        f"{rng.getrandbits(28):07x} {rng.choice(_VERBS)} {rng.choice(_NOUNS)} handling in {module}"
        for _ in range(12)
    )
    return f"{thought}\n{call}", log


def _analyst_turn(rng: random.Random) -> tuple[str, str]:
    table, column, metric = rng.choice(_TABLES), rng.choice(_COLUMNS), rng.choice(_METRICS)
    kind = rng.choice(["sql", "sql", "describe", "python", "plot"])
    thought = _sentences(rng, rng.randint(1, 2))
    if kind == "sql":
        query = (
            f"SELECT {column}, plan, SUM({metric}) AS {metric}, AVG(error_rate) AS error_rate "
            f"FROM analytics.{table} WHERE day >= DATE '2026-0{rng.randint(1, 9)}-01' "
            f"GROUP BY 1, 2 ORDER BY 3 DESC"
        )
        call = _tool_call("run_sql", query=query, limit=200)
        return f"{thought}\n{call}", _sql_rows(rng, rng.randint(12, 70))
    if kind == "describe":
        call = _tool_call("describe_table", table=f"analytics.{table}")
        schema = "\n".join(
            f"{name:<18} {rng.choice(['VARCHAR', 'BIGINT', 'DOUBLE', 'DATE', 'BOOLEAN'])}"
            for name in [*_COLUMNS, *_METRICS, "day", "id"]
        )
        return f"{thought}\n{call}", f"{schema}\nrows: {rng.randint(10**5, 10**9):,}"
    if kind == "python":
        code = (
            f"import pandas as pd\ndf = pd.read_parquet('/data/{table}.parquet')\n"
            f"print(df.groupby('{column}')['{metric}'].describe())"
        )
        call = _tool_call("run_python", code=code)
        stats = "\n".join(
            f"{name:<12} {rng.randint(100, 9999):>6} {rng.uniform(1, 500):>10.3f} "
            f"{rng.uniform(1, 80):>10.3f} {rng.uniform(0, 5):>8.3f} {rng.uniform(500, 2000):>10.3f}"
            for name in ["emea", "amer", "apac", "latam", "other"]
        )
        return (
            f"{thought}\n{call}",
            f"{'':<12} {'count':>6} {'mean':>10} {'std':>10} {'min':>8} {'max':>10}\n{stats}",
        )
    call = _tool_call("plot_chart", kind=rng.choice(["line", "bar"]), x="day", y=metric)
    return (
        f"{thought}\n{call}",
        f"chart saved: https://charts.example.org/c/{rng.getrandbits(40):010x}.png",
    )


def _researcher_turn(rng: random.Random) -> tuple[str, str]:
    topic = rng.choice(_TOPICS)
    kind = rng.choice(["search", "open", "open", "notes", "save"])
    thought = _sentences(rng, rng.randint(1, 2))
    if kind == "search":
        call = _tool_call(
            "web_search", query=f"{topic} best practices failure modes", num_results=8
        )
        return f"{thought}\n{call}", _search_hits(rng, 8)
    if kind == "open":
        url = f"https://{rng.choice(_SITES)}/{topic.replace(' ', '-')}-{rng.randint(100, 999)}"
        call = _tool_call("open_url", url=url)
        page = "\n\n".join(_sentences(rng, rng.randint(4, 8)) for _ in range(rng.randint(4, 12)))
        return f"{thought}\n{call}", f"# {topic.title()}\n\n{page}"
    if kind == "notes":
        call = _tool_call("search_notes", query=topic)
        notes = "\n".join(f"- ({2020 + rng.randint(0, 6)}) {_sentences(rng, 2)}" for _ in range(6))
        return f"{thought}\n{call}", notes
    call = _tool_call("save_note", title=f"Findings on {topic}", body=_sentences(rng, 3))
    return f"{thought}\n{call}", "note saved"


_TURN_GENERATORS = {"coder": _coder_turn, "analyst": _analyst_turn, "researcher": _researcher_turn}


def _task(agent: str, rng: random.Random) -> str:
    if agent == "coder":
        return (
            f"The nightly build is red: tests under tests/{rng.choice(_MODULES)} fail with a "
            f"{rng.choice(_ERRORS)} in the {rng.choice(_NOUNS)} handling. Find the cause, fix it, "
            "and make sure the suite passes."
        )
    if agent == "analyst":
        return (
            f"Why did {rng.choice(_METRICS)} for the {rng.choice(_TABLES)} table change last "
            f"month? Break it down by {rng.choice(_COLUMNS)} and tell me what drove it."
        )
    return (
        f"Write me a short briefing on {rng.choice(_TOPICS)}: what goes wrong in practice, and "
        "what do well-run teams do about it? Cite your sources."
    )


# ------------------------------------------------------------------------------- traces


@dataclass(frozen=True)
class Turn:
    """One model call of a session.

    Attributes:
        assistant: The scripted assistant message that enters the history after the call.
        tool: The tool result that follows it (``None`` after the final answer).
        max_tokens: Output tokens to request: the token length of ``assistant``.
        think: Seconds between this call's completion and the next call.
    """

    assistant: str
    tool: str | None
    max_tokens: int
    think: float


@dataclass(frozen=True)
class Session:
    """A multi-turn agent session."""

    session_id: str
    agent: str
    start: float
    task: str
    turns: list[Turn]

    def messages(self, turn: int, system: str) -> list[dict[str, str]]:
        """The chat messages of the request for turn ``turn`` (0-based)."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": self.task}]
        for earlier in self.turns[:turn]:
            messages.append({"role": "assistant", "content": earlier.assistant})
            messages.append({"role": "tool", "content": earlier.tool})
        return messages


@dataclass
class Trace:
    """A generated workload: agent definitions plus sessions with their timing."""

    seed: int
    rate: float | None
    agents: dict[str, dict[str, Any]]
    sessions: list[Session] = field(default_factory=list)

    @property
    def num_requests(self) -> int:
        """Total model calls in the trace."""
        return sum(len(session.turns) for session in self.sessions)

    def to_jsonl(self) -> str:
        """Serialize: a header line, then one line per session."""
        header = {
            "type": "header",
            "version": TRACE_VERSION,
            "seed": self.seed,
            "rate": self.rate,
            "agents": self.agents,
        }
        lines = [json.dumps(header, ensure_ascii=False)]
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
    def load(cls, path: Path) -> Trace:
        """Read a trace written by :meth:`save`."""
        lines = [json.loads(line) for line in path.read_text().splitlines() if line]
        header, rows = lines[0], lines[1:]
        if header.get("type") != "header" or header.get("version") != TRACE_VERSION:
            raise ValueError(f"{path} is not a version-{TRACE_VERSION} agent trace")
        sessions = [
            Session(
                session_id=row["session_id"],
                agent=row["agent"],
                start=row["start"],
                task=row["task"],
                turns=[Turn(**turn) for turn in row["turns"]],
            )
            for row in rows
        ]
        return cls(header["seed"], header["rate"], header["agents"], sessions)


def generate(
    num_sessions: int,
    rate: float | None,
    seed: int,
    count_tokens: Callable[[str], int],
    system_tokens: int = SYSTEM_PROMPT_TOKENS,
    mean_think: float = 1.0,
) -> Trace:
    """Generate a trace.

    Args:
        num_sessions: Number of sessions.
        rate: Nominal request rate in requests/s; ``None`` starts everything at once with
            no think time.
        seed: Seed for contents and timing. Contents depend only on the seed and the
            session index, so traces for different rates contain the same sessions.
        count_tokens: Token counter used to size system prompts and outputs.
        system_tokens: Target size of each agent type's system prompt (with tool schemas).
        mean_think: Mean of the exponential think time between turns, in seconds.
    """
    agents = {
        name: {"system": system_prompt(name, count_tokens, system_tokens), "tools": tools}
        for name, tools in _AGENT_TOOLS.items()
    }
    trace = Trace(seed=seed, rate=rate, agents=agents)
    arrivals = random.Random(f"{seed}-arrivals")
    clock = 0.0
    names = list(agents)
    for index in range(num_sessions):
        rng = random.Random(f"{seed}-session-{index}")
        timing = random.Random(f"{seed}-timing-{index}")
        agent = names[rng.randrange(len(names))]
        make_turn = _TURN_GENERATORS[agent]
        num_turns = rng.randint(MIN_TURNS, MAX_TURNS)
        turns = []
        for turn in range(num_turns):
            last = turn == num_turns - 1
            if last:
                assistant, tool = _sentences(rng, rng.randint(6, 16)), None
            else:
                assistant, tool = make_turn(rng)
            think = 0.0 if rate is None or last else timing.expovariate(1.0 / mean_think)
            turns.append(Turn(assistant, tool, count_tokens(assistant), think))
        if rate is not None:
            clock += arrivals.expovariate(rate / MEAN_TURNS)
        trace.sessions.append(
            Session(
                f"s{index:05d}", agent, 0.0 if rate is None else clock, _task(agent, rng), turns
            )
        )
    return trace


# ------------------------------------------------------------------------------- replay


@dataclass
class RequestRecord:
    """Client-side measurements of one request (times in seconds since replay start)."""

    session_id: str
    agent: str
    turn: int
    sent: float
    first_token: float | None = None
    finished: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int | None = None
    error: str | None = None

    @property
    def ttft(self) -> float:
        """Time to first token."""
        return self.first_token - self.sent

    @property
    def e2e(self) -> float:
        """End-to-end latency."""
        return self.finished - self.sent

    @property
    def tpot(self) -> float | None:
        """Mean time per output token after the first."""
        if self.completion_tokens < 2:
            return None
        return (self.finished - self.first_token) / (self.completion_tokens - 1)


def render_prompts(trace: Trace, tokenizer: Any) -> dict[str, list[str]]:
    """Every request's prompt as text, rendered with the model's chat template.

    Rendering on the client and sending text to ``/v1/completions`` guarantees that all
    servers tokenize exactly the same string, whatever their own chat-template handling.
    """
    prompts = {}
    for session in trace.sessions:
        agent = trace.agents[session.agent]
        prompts[session.session_id] = [
            tokenizer.apply_chat_template(
                session.messages(turn, agent["system"]),
                tools=agent["tools"],
                tokenize=False,
                add_generation_prompt=True,
            )
            for turn in range(len(session.turns))
        ]
    return prompts


def tokenize_prompts(prompts: dict[str, list[str]], tokenizer: Any) -> dict[str, list[list[int]]]:
    """The rendered prompts as token ids, for replaying without server-side tokenization."""
    return {
        session_id: [tokenizer(text, add_special_tokens=False)["input_ids"] for text in texts]
        for session_id, texts in prompts.items()
    }


async def _one_request(
    http: Any, url: str, body: dict[str, Any], record: RequestRecord, origin: float
) -> None:
    """Send one streaming completion request and fill in ``record``."""
    record.sent = time.perf_counter() - origin
    try:
        async with http.post(url, json=body) as response:
            if response.status != 200:
                record.error = f"HTTP {response.status}: {(await response.text())[:200]}"
                return
            async for raw in response.content:
                if not raw.startswith(b"data: "):
                    continue
                now = time.perf_counter() - origin
                payload = raw[6:].strip()
                if payload == b"[DONE]":
                    break
                event = json.loads(payload)
                if "error" in event:
                    record.error = json.dumps(event["error"])[:200]
                    return
                if event.get("choices"):
                    if record.first_token is None:
                        record.first_token = now
                    record.finished = now
                usage = event.get("usage")
                if usage:
                    record.prompt_tokens = usage["prompt_tokens"]
                    record.completion_tokens = usage["completion_tokens"]
                    details = usage.get("prompt_tokens_details") or {}
                    record.cached_tokens = details.get("cached_tokens")
        if record.first_token is None:
            record.error = "stream ended without any token"
    except Exception as error:  # a failed request is a result, not a crash of the benchmark
        record.error = repr(error)[:200]


async def replay(
    trace: Trace, prompts: dict[str, list[Any]], base_url: str, model: str
) -> tuple[list[RequestRecord], float]:
    """Replay ``trace`` against an OpenAI-compatible server.

    ``prompts`` holds each request's prompt as text (:func:`render_prompts`) or as token
    ids (:func:`tokenize_prompts`); ``/v1/completions`` accepts both. Returns the
    per-request records and the wall-clock duration of the replay.
    """
    import aiohttp

    url = f"{base_url.rstrip('/')}/v1/completions"
    records: list[RequestRecord] = []
    origin = time.perf_counter()

    async def run_session(session: Session, http: aiohttp.ClientSession) -> None:
        await asyncio.sleep(max(0.0, session.start - (time.perf_counter() - origin)))
        for index, turn in enumerate(session.turns):
            body = {
                "model": model,
                "prompt": prompts[session.session_id][index],
                "max_tokens": turn.max_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            record = RequestRecord(session.session_id, session.agent, index, 0.0)
            records.append(record)
            await _one_request(http, url, body, record, origin)
            if record.error is not None:
                return  # the rest of the session depends on this turn
            await asyncio.sleep(turn.think)

    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http:
        await asyncio.gather(*(run_session(session, http) for session in trace.sessions))
    return records, time.perf_counter() - origin


def _percentiles(values: list[float]) -> dict[str, float]:
    import numpy as np

    if not values:
        return {}
    array = np.asarray(values) * 1e3
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def summarize(records: list[RequestRecord], duration: float) -> dict[str, Any]:
    """Aggregate a replay: throughput, latency percentiles (ms), and cache-hit share."""
    good = [r for r in records if r.error is None]
    failed = [r for r in records if r.error is not None]
    prompt_tokens = sum(r.prompt_tokens for r in good)
    completion_tokens = sum(r.completion_tokens for r in good)
    with_cache_info = [r for r in good if r.cached_tokens is not None]
    summary: dict[str, Any] = {
        "requests": len(records),
        "completed": len(good),
        "failed": len(failed),
        "errors": sorted({r.error for r in failed})[:5],
        "duration_s": duration,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "request_throughput": len(good) / duration,
        "output_tokens_per_second": completion_tokens / duration,
        "total_tokens_per_second": (prompt_tokens + completion_tokens) / duration,
        "ttft_ms": _percentiles([r.ttft for r in good]),
        "ttft_first_turn_ms": _percentiles([r.ttft for r in good if r.turn == 0]),
        "ttft_later_turns_ms": _percentiles([r.ttft for r in good if r.turn > 0]),
        "tpot_ms": _percentiles([r.tpot for r in good if r.tpot is not None]),
        "e2e_ms": _percentiles([r.e2e for r in good]),
        "client_reported_cached_token_fraction": (
            sum(r.cached_tokens for r in with_cache_info)
            / max(1, sum(r.prompt_tokens for r in with_cache_info))
            if with_cache_info
            else None
        ),
    }
    return summary


def iter_requests(trace: Trace) -> Iterator[tuple[Session, int, Turn]]:
    """Every ``(session, turn index, turn)`` of a trace."""
    for session in trace.sessions:
        for index, turn in enumerate(session.turns):
            yield session, index, turn


# ---------------------------------------------------------------------------------- CLI


def _tokenizer(model: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    gen = commands.add_parser("generate", help="write a JSONL trace")
    gen.add_argument("--sessions", type=int, default=100)
    gen.add_argument("--rate", type=float, default=None, help="requests/s; omit for max")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="tokenizer for sizing")
    gen.add_argument("--out", type=Path, required=True)

    rep = commands.add_parser("replay", help="replay a trace against an endpoint")
    rep.add_argument("trace", type=Path)
    rep.add_argument("--base-url", default="http://localhost:8000")
    rep.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    rep.add_argument("--out", type=Path, default=None, help="write the summary JSON here")
    args = parser.parse_args()

    tokenizer = _tokenizer(args.model)
    if args.command == "generate":

        def count(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])

        trace = generate(args.sessions, args.rate, args.seed, count)
        trace.save(args.out)
        print(f"{len(trace.sessions)} sessions, {trace.num_requests} requests -> {args.out}")
        print(f"sha256 {trace.sha256}")
        return
    trace = Trace.load(args.trace)
    prompts = render_prompts(trace, tokenizer)
    records, duration = asyncio.run(replay(trace, prompts, args.base_url, args.model))
    summary = {"trace_sha256": trace.sha256, **summarize(records, duration)}
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
