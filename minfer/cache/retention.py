"""Session-aware retention: which cached KV to give up first under memory pressure.

An agent session is a chain of model calls separated by tool calls. While a tool runs the
session holds no GPU work, but its history — tens of thousands of tokens of KV — sits in
the prefix cache waiting for the next call. Recency says nothing useful about that KV: the
session that has been paused longest is, if anything, the one about to come back, and the
most recently used entry may belong to a session that has just given its final answer and
will never be read again.

The policy here replaces "least recently used" with "expected to be needed last". Each
cached leaf is scored by the expected time until its next use, and eviction takes the
highest score first:

* a session that has declared itself finished: never needed again;
* a session paused in a tool call: the expected remaining pause, given the tool it called
  and how long it has already been away, from the pauses observed so far for that tool
  (:class:`ReturnPredictor`). Tool latencies are heavy-tailed, so the conditional
  expectation grows the longer a session stays away — a session ten seconds into a pause
  that usually takes 0.3 s is probably in the tail, and its score rises accordingly;
* KV that belongs to no session: its idle time, which orders such entries exactly as LRU
  does.

This is the idea of Continuum's KV time-to-live (Li et al., arXiv 2511.02230) and of
CacheWise's predictive eviction (Tiwari et al., arXiv 2606.16824), expressed as a ranking
rather than a pin with a deadline; nothing here is new.
"""

from __future__ import annotations

import bisect
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

ANY_TOOL = "*"
NEVER = float("inf")


@dataclass(frozen=True)
class SessionHint:
    """What a request says about the session it belongs to.

    Attributes:
        session_id: Identifies the agent session across its model calls.
        tools: Names of the tools the client runs after this response (empty: none).
            ``None`` means the client did not say; the server may infer it from the output.
        final: The session ends with this response. ``None``: unknown.
    """

    session_id: str
    tools: tuple[str, ...] | None = None
    final: bool | None = None


@dataclass(eq=False)
class SessionState:
    """What the engine knows about one session.

    Attributes:
        session_id: The client's identifier.
        active: Requests of the session currently waiting or running in the engine.
        finished_at: When its latest request completed (``None`` before the first).
        tool_key: Predictor key of the pause it is in (the tools it went off to run).
        final: The session said it is over; its KV will not be read again.
    """

    session_id: str
    active: int = 0
    finished_at: float | None = None
    tool_key: str = ANY_TOOL
    final: bool = False


def tool_key(tools: Sequence[str] | None) -> str:
    """Predictor key for a set of tool calls made together (order-insensitive)."""
    return "+".join(sorted(set(tools))) if tools else ANY_TOOL


class ReturnPredictor:
    """Online estimate of how much longer a paused session stays away.

    Keeps the most recent pauses per tool key (and overall) and answers with the empirical
    conditional mean ``E[pause - elapsed | pause > elapsed]``. A key's pauses are kept
    sorted with their suffix sums, rebuilt lazily after new observations, so a query is a
    binary search: eviction scores every cached leaf with one.

    Args:
        capacity: Pauses remembered per key.
        min_samples: Below this many samples for a key, the pooled history is used.
    """

    def __init__(self, capacity: int = 512, min_samples: int = 8) -> None:
        self.capacity = capacity
        self.min_samples = min_samples
        self._pauses: dict[str, deque[float]] = {ANY_TOOL: deque(maxlen=capacity)}
        # key -> (pauses in ascending order, sums of the pauses from each index on)
        self._sorted: dict[str, tuple[list[float], list[float]]] = {}

    def observe(self, key: str, pause: float) -> None:
        """Record that a session paused on ``key`` returned after ``pause`` seconds."""
        if key != ANY_TOOL:
            self._pauses.setdefault(key, deque(maxlen=self.capacity)).append(pause)
            self._sorted.pop(key, None)
        self._pauses[ANY_TOOL].append(pause)
        self._sorted.pop(ANY_TOOL, None)

    def _ordered(self, key: str) -> tuple[list[float], list[float]]:
        cached = self._sorted.get(key)
        if cached is None:
            ordered = sorted(self._pauses[key])
            suffix = [0.0] * (len(ordered) + 1)
            for index in range(len(ordered) - 1, -1, -1):
                suffix[index] = suffix[index + 1] + ordered[index]
            cached = self._sorted[key] = (ordered, suffix)
        return cached

    def expected_remaining(self, key: str, elapsed: float) -> float:
        """Expected seconds until a session ``elapsed`` seconds into a ``key`` pause returns.

        With no observed pause longer than ``elapsed`` there is nothing to condition on;
        the answer is then ``elapsed`` itself (it has been away this long, expect as long
        again), which also makes a predictor with no history rank sessions as LRU would.
        """
        for name in (key, ANY_TOOL):
            pauses = self._pauses.get(name)
            if pauses is None or len(pauses) < self.min_samples:
                continue
            ordered, suffix = self._ordered(name)
            first = bisect.bisect_right(ordered, elapsed)
            longer = len(ordered) - first
            if longer:
                return suffix[first] / longer - elapsed
        return elapsed


class SessionRetention:
    """Tracks sessions and scores cached KV by expected time until its next use.

    Args:
        clock: Monotonic time source (seconds).
        predictor: Pause-length model; a fresh :class:`ReturnPredictor` by default.
        max_sessions: Idle sessions remembered; the oldest are forgotten beyond this.
    """

    def __init__(
        self,
        clock: Callable[[], float],
        predictor: ReturnPredictor | None = None,
        max_sessions: int = 100_000,
    ) -> None:
        self.clock = clock
        self.predictor = predictor or ReturnPredictor()
        self.max_sessions = max_sessions
        self.sessions: dict[str, SessionState] = {}

    def begin(self, session_id: str) -> SessionState:
        """A request of ``session_id`` arrived: learn from the pause that just ended."""
        state = self.sessions.pop(session_id, None)
        if state is None:
            state = SessionState(session_id)
            if len(self.sessions) >= self.max_sessions:
                del self.sessions[next(iter(self.sessions))]
        elif state.active == 0 and state.finished_at is not None:
            self.predictor.observe(state.tool_key, self.clock() - state.finished_at)
        self.sessions[session_id] = state  # most recently active last
        state.active += 1
        state.final = False
        return state

    def end(self, state: SessionState, tools: Sequence[str] | None, final: bool | None) -> None:
        """A request of the session completed (or was dropped) and it went off to run ``tools``."""
        state.active = max(0, state.active - 1)
        state.finished_at = self.clock()
        self.describe(state, tools, final)

    def describe(
        self, state: SessionState, tools: Sequence[str] | None, final: bool | None
    ) -> None:
        """Set what the session is doing now (also used when the server infers it later)."""
        if tools is not None:
            state.tool_key = tool_key(tools)
        if final is not None:
            state.final = final

    def score(self, owner: SessionState | None, last_used: float, now: float) -> float:
        """Expected seconds until KV owned by ``owner`` is next needed (larger: evict sooner)."""
        if owner is None or owner.finished_at is None:
            return now - last_used
        if owner.active:
            return 0.0  # a request of the session is in the engine right now
        if owner.final:
            return NEVER
        return self.predictor.expected_remaining(owner.tool_key, now - owner.finished_at)
