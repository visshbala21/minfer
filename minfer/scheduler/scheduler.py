"""Continuous batching: admission, per-step work selection, and preemption.

The scheduler is pure Python over :class:`Sequence` objects and a :class:`BlockManager`; it
never touches a tensor, so every policy here is unit-tested on the CPU.

Each engine step asks :meth:`Scheduler.schedule` for a :class:`StepPlan`: which sequences
decode one token and which prefill how many tokens. All KV blocks the step will write are
allocated before the plan is returned, so the forward pass cannot run out of memory halfway.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from minfer.cache.block_manager import BlockManager, OutOfBlocks
from minfer.scheduler.sequence import Sequence, SequenceStatus


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduling limits.

    Attributes:
        max_num_seqs: Maximum sequences running (holding KV) at once.
        max_batched_tokens: Maximum tokens one engine step may process (decode tokens plus
            prefill tokens). A prompt longer than the remaining budget is prefilled over
            several steps.
        watermark: Fraction of the block pool kept free at admission time, so that already
            running sequences can keep growing without immediately preempting a newcomer.
    """

    max_num_seqs: int = 256
    max_batched_tokens: int = 8192
    watermark: float = 0.01

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1 or self.max_batched_tokens < 1:
            raise ValueError("max_num_seqs and max_batched_tokens must be positive")
        if not 0.0 <= self.watermark < 1.0:
            raise ValueError("watermark must be in [0, 1)")


@dataclass
class StepPlan:
    """Work selected for one engine step.

    Attributes:
        decode: Sequences with exactly one pending token.
        prefill: ``(sequence, num_tokens)`` pairs; ``num_tokens`` may be less than the
            sequence's pending count, in which case the step produces no token for it.
    """

    decode: list[Sequence] = field(default_factory=list)
    prefill: list[tuple[Sequence, int]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        """Whether the step has nothing to run."""
        return not self.decode and not self.prefill


class Scheduler:
    """First-come-first-served continuous batching with preemption by recompute.

    * **Admission.** Waiting sequences are admitted in order while a running slot is free
      and the pool can hold their whole pending context on top of what earlier admissions
      still have to allocate, plus the watermark. Admission stops at the first sequence that
      does not fit (no overtaking, so long prompts cannot starve).
    * **Step selection.** Running sequences are visited oldest first. A sequence with one
      pending token decodes; one with more prefills as much as the token budget allows.
    * **Preemption.** When a sequence cannot get a block, the most recently admitted running
      sequence is preempted: its blocks are freed and it returns to the *front* of the
      waiting queue keeping its generated tokens, to be recomputed later. Because sequences
      are visited oldest first, a victim is never one already selected for this step, and
      the oldest sequence can always make progress.
    """

    def __init__(self, config: SchedulerConfig, blocks: BlockManager) -> None:
        self.config = config
        self.blocks = blocks
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    # ---------------------------------------------------------------- queues

    @property
    def max_sequence_tokens(self) -> int:
        """Longest sequence (prompt + output) the block pool can hold."""
        return self.blocks.num_usable * self.blocks.block_size

    @property
    def has_work(self) -> bool:
        """Whether any sequence is waiting or running."""
        return bool(self.waiting or self.running)

    def add(self, seq: Sequence) -> None:
        """Queue a new sequence."""
        if len(seq.tokens) + 1 > self.max_sequence_tokens:
            raise ValueError(
                f"sequence of {len(seq.tokens)} tokens cannot fit in a KV cache of "
                f"{self.max_sequence_tokens} tokens"
            )
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def finish(self, seq: Sequence, reason: str) -> None:
        """Mark a running sequence finished and release its blocks."""
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        self.running.remove(seq)
        self.blocks.free(seq.seq_id)

    def abort(self, seq: Sequence) -> None:
        """Drop a sequence wherever it is (client disconnected)."""
        if seq.status is SequenceStatus.RUNNING:
            self.finish(seq, "abort")
        elif seq.status is SequenceStatus.WAITING:
            self.waiting.remove(seq)
            seq.status = SequenceStatus.FINISHED
            seq.finish_reason = "abort"

    # ------------------------------------------------------------- admission

    def _admit(self) -> None:
        """Move waiting sequences to running while slots and memory allow."""
        if not self.waiting:
            return
        blocks = self.blocks
        reserve = int(blocks.num_usable * self.config.watermark)
        # Blocks that running sequences still have to allocate to finish their prefill.
        promised = sum(
            blocks.blocks_missing(s.seq_id, len(s.tokens))
            for s in self.running
            if s.num_pending > 1
        )
        while self.waiting and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting[0]
            needed = blocks.blocks_for(len(seq.tokens) + 1)
            fits = blocks.num_free - promised - reserve >= needed
            # With nothing running the head must be admitted, or nothing ever would be.
            if not fits and self.running:
                break
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            promised += needed

    # ------------------------------------------------------------ preemption

    def _preempt(self, seq: Sequence) -> None:
        """Take a running sequence's KV away and put it at the front of the queue."""
        self.running.remove(seq)
        self.blocks.free(seq.seq_id)
        seq.num_cached = 0
        seq.num_preemptions += 1
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def _reserve(self, seq: Sequence, num_tokens: int) -> bool:
        """Allocate blocks so ``seq`` can hold ``num_tokens``, preempting younger sequences.

        Returns ``False`` if ``seq`` itself had to be preempted.
        """
        while True:
            try:
                self.blocks.ensure_capacity(seq.seq_id, num_tokens)
                return True
            except OutOfBlocks:
                victim = self.running[-1]
                self._preempt(victim)
                if victim is seq:
                    return False

    # -------------------------------------------------------------- planning

    def schedule(self) -> StepPlan:
        """Admit what fits, then select and allocate this step's work."""
        self._admit()
        plan = StepPlan()
        num_decode = sum(1 for s in self.running if s.num_pending == 1)
        prefill_budget = max(self.config.max_batched_tokens - num_decode, 0)
        index = 0
        while index < len(self.running):
            seq = self.running[index]
            index += 1
            pending = seq.num_pending
            if pending == 1:
                if self._reserve(seq, seq.num_cached + 1):
                    plan.decode.append(seq)
                continue
            count = min(pending, prefill_budget)
            if count == 0:
                continue
            if self._reserve(seq, seq.num_cached + count):
                plan.prefill.append((seq, count))
                prefill_budget -= count
        return plan
