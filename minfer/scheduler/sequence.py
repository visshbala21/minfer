"""Per-request generation state."""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field

from minfer.cache.radix_tree import RestoreTicket
from minfer.cache.retention import SessionHint, SessionState
from minfer.sampling import SamplingParams


@dataclass
class ReuseSpan:
    """Cached KV of identical tokens at another position, to be copied instead of computed.

    Attributes:
        dest_start: Position in this sequence where the span begins.
        source_start: Position the same tokens have in the sequence they were cached for.
        head: Leading tokens of the span that are recomputed in their new context; the
            rest is copied (with the keys rotated by ``dest_start - source_start``).
        source_blocks: The blocks holding the span's KV, pinned until the copy is done.
        block_size: Tokens per block.
    """

    dest_start: int
    source_start: int
    head: int
    source_blocks: list[int]
    block_size: int

    @property
    def copy_start(self) -> int:
        """First position whose KV is copied."""
        return self.dest_start + self.head

    @property
    def end(self) -> int:
        """Position just past the span."""
        return self.dest_start + len(self.source_blocks) * self.block_size


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
        synced_blocks: Leading full blocks already shared through the prefix cache.
        cache_hit_tokens: Tokens taken from the prefix cache at the first admission.
        draft_len: Current speculative draft length for this request (adapts to how many
            drafted tokens were accepted last time; 0 until the engine sets it).
        draft_acceptance: Running share of this request's drafted tokens that were accepted.
        draft_pause: Decode steps left during which nothing is drafted for this request
            (speculation switched itself off after too many rejections).
        rng: The request's own random stream (seeded requests only).
        session: The agent session the request belongs to, if it named one.
        hint: What the request said the session does after this response.
        restore: A copy of this sequence's cached history from the host tier that is in
            flight or has completed and not yet been adopted.
        restore_hit_tokens: Tokens adopted from the GPU cache before that copy started.
        reuse: Position-independent reuse still to be carried out, in position order.
        admitted_at: Scheduler-clock time of the first admission (``None`` before it).
        restore_began: When the current host-tier copy for it started.
        restore_seconds: Time spent waiting for host-tier copies, summed.
        first_token_at: Scheduler-clock time its first token was emitted.
        prefill_computed: Tokens run through prefill passes for it (recomputation after a
            preemption included).
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
    synced_blocks: int = 0
    cache_hit_tokens: int = 0
    draft_len: int = 0
    draft_acceptance: float = 1.0
    draft_pause: int = 0
    rng: random.Random | None = field(default=None, repr=False)
    session: SessionState | None = field(default=None, repr=False)
    hint: SessionHint | None = field(default=None, repr=False)
    restore: RestoreTicket | None = field(default=None, repr=False)
    restore_hit_tokens: int = 0
    reuse: list[ReuseSpan] = field(default_factory=list, repr=False)
    admitted_at: float | None = None
    restore_began: float = 0.0
    restore_seconds: float = 0.0
    first_token_at: float | None = None
    prefill_computed: int = 0

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
