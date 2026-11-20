"""Per-request generation state."""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field

from minfer.sampling import SamplingParams


class SequenceStatus(enum.Enum):
    """Lifecycle of a request inside the engine."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(eq=False)
class Sequence:
    """One generation request.

    ``tokens`` is the prompt followed by everything generated so far; ``num_cached`` is how
    many leading tokens have their keys/values in the KV cache.

    Invariant: an unfinished sequence always has at least one *pending* token
    (``len(tokens) > num_cached``). Running the model over all pending tokens produces the
    logits for the next token. First prefill, chunked prefill, decode, recompute after
    preemption and (later) a prefix-cache hit are therefore the same operation; they only
    differ in how many tokens are pending.

    Attributes:
        seq_id: Engine-unique id; also the key of the sequence's block table.
        tokens: Prompt + generated token ids.
        prompt_len: Number of prompt tokens.
        params: The request's sampling parameters.
        max_new_tokens: Generation budget after clamping to the context limit.
        arrival_time: ``time.perf_counter()`` when the request was submitted.
        num_cached: Leading tokens whose KV is in the cache.
        status: Current lifecycle state.
        finish_reason: ``"stop"`` or ``"length"`` once finished.
        num_preemptions: Times the sequence lost its KV and had to recompute.
        rng: The request's own random stream (seeded requests only).
    """

    seq_id: int
    tokens: list[int]
    prompt_len: int
    params: SamplingParams
    max_new_tokens: int
    arrival_time: float = 0.0
    num_cached: int = 0
    status: SequenceStatus = SequenceStatus.WAITING
    finish_reason: str | None = None
    num_preemptions: int = 0
    rng: random.Random | None = field(default=None, repr=False)

    @property
    def num_pending(self) -> int:
        """Tokens not yet in the KV cache."""
        return len(self.tokens) - self.num_cached

    @property
    def num_generated(self) -> int:
        """Tokens generated so far."""
        return len(self.tokens) - self.prompt_len

    @property
    def output_ids(self) -> list[int]:
        """The generated tokens."""
        return self.tokens[self.prompt_len :]
