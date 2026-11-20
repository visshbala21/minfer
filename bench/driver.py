"""Drive an in-process engine with a timed request trace and collect per-token timestamps.

Used by the engine-level benchmarks (no HTTP, no tokenizer): what is measured is the
engine's own scheduling and execution, with arrivals replayed in real time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from minfer.engine import Engine
from minfer.sampling import SamplingParams


@dataclass(frozen=True)
class TraceRequest:
    """One request of a trace: when it arrives, what it asks for, and a label for grouping."""

    arrival: float
    prompt: list[int]
    params: SamplingParams
    kind: str = "default"


@dataclass
class RequestTiming:
    """Timestamps (seconds since the start of the run) of one request."""

    kind: str
    prompt_tokens: int
    arrival: float
    token_times: list[float] = field(default_factory=list)

    @property
    def ttft(self) -> float:
        """Time from arrival to the first generated token."""
        return self.token_times[0] - self.arrival

    @property
    def tpot(self) -> float | None:
        """Mean time per output token after the first; ``None`` for one-token outputs."""
        if len(self.token_times) < 2:
            return None
        return (self.token_times[-1] - self.token_times[0]) / (len(self.token_times) - 1)

    @property
    def inter_token_gaps(self) -> list[float]:
        """Gaps between consecutive generated tokens."""
        return [b - a for a, b in zip(self.token_times, self.token_times[1:], strict=False)]


def run_trace(engine: Engine, trace: list[TraceRequest]) -> tuple[list[RequestTiming], float]:
    """Replay ``trace`` against ``engine`` in real time.

    Returns the per-request timings (in trace order) and the wall-clock duration.
    """
    pending = sorted(range(len(trace)), key=lambda i: trace[i].arrival)
    timings: dict[int, RequestTiming] = {}
    by_seq: dict[int, int] = {}
    cursor = 0
    start = time.perf_counter()
    while cursor < len(pending) or engine.has_work:
        now = time.perf_counter() - start
        while cursor < len(pending) and trace[pending[cursor]].arrival <= now:
            index = pending[cursor]
            request = trace[index]
            seq_id = engine.add_request(request.prompt, request.params)
            by_seq[seq_id] = index
            timings[index] = RequestTiming(request.kind, len(request.prompt), request.arrival)
            cursor += 1
        if not engine.has_work:
            time.sleep(max(0.0, trace[pending[cursor]].arrival - (time.perf_counter() - start)))
            continue
        outputs = engine.step()
        stamp = time.perf_counter() - start
        for out in outputs:
            timings[by_seq[out.seq_id]].token_times.append(stamp)
    duration = time.perf_counter() - start
    return [timings[i] for i in range(len(trace))], duration


def percentiles(values: list[float], scale: float = 1e3) -> dict[str, float]:
    """p50/p90/p99/max/mean of ``values`` (seconds), scaled to milliseconds by default."""
    if not values:
        return {}
    array = np.asarray(values) * scale
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "count": len(values),
    }
