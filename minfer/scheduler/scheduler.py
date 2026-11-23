"""Continuous batching: admission, per-step work selection, and preemption.

The scheduler is pure Python over :class:`Sequence` objects and a :class:`BlockManager`; it
never touches a tensor, so every policy here is unit-tested on the CPU.

Each engine step asks :meth:`Scheduler.schedule` for a :class:`StepPlan`: which sequences
decode one token and which prefill how many tokens. All KV blocks the step will write are
allocated before the plan is returned, so the forward pass cannot run out of memory halfway.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import islice

from minfer.cache.block_manager import BlockManager, OutOfBlocks
from minfer.cache.radix_tree import PrefixCache, RestoreTicket
from minfer.scheduler.sequence import ReuseSpan, Sequence, SequenceStatus


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduling limits.

    Attributes:
        max_num_seqs: Maximum sequences running (holding KV) at once.
        max_batched_tokens: Maximum tokens one engine step may process (decode tokens plus
            prefill tokens). A prompt longer than the remaining budget is prefilled over
            several steps.
        max_prefill_tokens: Maximum prefill tokens per step while any sequence is decoding
            (chunked prefill). ``None`` leaves prefill bounded only by
            ``max_batched_tokens``.
        watermark: Fraction of the block pool kept free at admission time, so that already
            running sequences can keep growing without immediately preempting a newcomer.
        cache_aware: With a prefix cache, admit the waiting sequence with the longest cached
            prefix first instead of the oldest.
        starvation_seconds: Starvation guard for cache-aware admission: a sequence that has
            waited this long is admitted in arrival order, ahead of any cache preference.
        admission_window: How many of the oldest waiting sequences cache-aware admission
            considers each time (bounds the per-step cost of looking prefixes up).
        min_restore_blocks: With a host tier, the smallest run of host-only blocks worth
            copying back for a request; shorter runs are recomputed (a copy is noticed one
            engine step after it completes, by which time a short prefill would be done).
        reuse_min_tokens: Position-independent reuse: shortest run of prompt tokens, cached
            at another position, whose KV is copied instead of computed. ``None`` disables
            the feature (it needs the prefix cache's content index).
        reuse_recompute_ratio: Share of each reused run, from its start, that is recomputed
            in its new context rather than copied.
    """

    max_num_seqs: int = 256
    max_batched_tokens: int = 8192
    max_prefill_tokens: int | None = None
    watermark: float = 0.01
    cache_aware: bool = True
    starvation_seconds: float = 1.0
    admission_window: int = 32
    min_restore_blocks: int = 8
    reuse_min_tokens: int | None = None
    reuse_recompute_ratio: float = 0.15

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1 or self.max_batched_tokens < 1:
            raise ValueError("max_num_seqs and max_batched_tokens must be positive")
        if self.max_prefill_tokens is not None and self.max_prefill_tokens < 1:
            raise ValueError("max_prefill_tokens must be positive")
        if not 0.0 <= self.watermark < 1.0:
            raise ValueError("watermark must be in [0, 1)")


@dataclass
class StepPlan:
    """Work selected for one engine step.

    Attributes:
        decode: Sequences with exactly one pending token.
        prefill: ``(sequence, num_tokens)`` pairs; ``num_tokens`` may be less than the
            sequence's pending count, in which case the step produces no token for it.
        copies: ``(sequence, span)`` pairs: KV to copy from another position before the
            step's forward passes (the sequence's ``num_cached`` already counts it).
    """

    decode: list[Sequence] = field(default_factory=list)
    prefill: list[tuple[Sequence, int]] = field(default_factory=list)
    copies: list[tuple[Sequence, ReuseSpan]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        """Whether the step has nothing to run."""
        return not self.decode and not self.prefill and not self.copies


class Scheduler:
    """Continuous batching with preemption by recompute and an optional prefix cache.

    * **Admission.** Waiting sequences are admitted one at a time while a running slot is
      free and the pool can hold the candidate's whole pending context on top of what
      earlier admissions still have to allocate, plus the watermark. Admission stops at the
      first candidate that does not fit. Without a prefix cache (or with ``cache_aware``
      off) the candidate is the oldest waiting sequence: plain FCFS, no overtaking.
    * **Cache-aware admission.** With a prefix cache the candidate is, among the oldest
      ``admission_window`` waiting sequences, the one with the longest cached prefix: it
      is the cheapest to start, and admitting it now uses cached blocks before memory
      pressure evicts them. Two things override the preference, in arrival order:
      preempted sequences (they already hold a place) and sequences that have waited
      longer than ``starvation_seconds``.
    * **Step selection.** Running sequences are visited oldest first. A sequence with one
      pending token decodes; one with more prefills as much as the token budget allows.
    * **Memory pressure.** When a sequence cannot get a block, blocks held only by the
      prefix cache are evicted first. If that is not enough, the most recently admitted
      running sequence is preempted: its blocks are released and it returns to the *front*
      of the waiting queue keeping its generated tokens, to be recomputed later. Because
      sequences are visited oldest first, a victim is never one already selected for this
      step, and the oldest sequence can always make progress.
    * **Host tier.** With one, a waiting sequence whose cached history was demoted to host
      memory has it copied back instead of recomputing it: the sequence leaves the queue
      while the copy runs (other sequences keep being admitted and stepped) and returns to
      its front when the copy completes.
    * **Position-independent reuse.** Optionally, runs of a prompt that are cached at a
      *different* position (a file read twice, a system prompt behind a different first
      paragraph) are not prefilled: the head of each run is computed in its new context and
      the rest of its KV is copied with the keys re-rotated. This is approximate — the
      copied KV was computed after different preceding tokens — and is off by default.
    * **Sharing.** With a prefix cache, every full block a sequence computes is inserted
      into the radix tree as soon as the step that filled it completes, and a sequence
      that is still prefilling re-checks the tree before each chunk. Requests with a
      common prefix that arrive together therefore compute it once: the first chunk of
      the first request is in the tree before the others are scheduled.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        blocks: BlockManager,
        cache: PrefixCache | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.blocks = blocks
        self.cache = cache
        self.clock = clock
        self.host = cache.host if cache is not None else None
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.restoring: list[Sequence] = []
        self._abandoned: list[RestoreTicket] = []
        self.num_preemptions = 0
        self.num_restores = 0

    # ---------------------------------------------------------------- queues

    @property
    def max_sequence_tokens(self) -> int:
        """Longest sequence (prompt + output) the block pool can hold."""
        return self.blocks.num_usable * self.blocks.block_size

    @property
    def has_work(self) -> bool:
        """Whether any sequence is waiting, running, or having its KV copied back."""
        return bool(self.waiting or self.running or self.restoring)

    def add(self, seq: Sequence) -> None:
        """Queue a new sequence."""
        if len(seq.tokens) + 1 > self.max_sequence_tokens:
            raise ValueError(
                f"sequence of {len(seq.tokens)} tokens cannot fit in a KV cache of "
                f"{self.max_sequence_tokens} tokens"
            )
        seq.status = SequenceStatus.WAITING
        seq.arrival_time = self.clock()
        self.waiting.append(seq)

    def finish(self, seq: Sequence, reason: str) -> None:
        """Mark a running sequence finished and release its blocks.

        Blocks the prefix cache also holds stay allocated (and become evictable).
        """
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        self.running.remove(seq)
        self._drop_reuse(seq, None)
        self.blocks.free(seq.seq_id)

    def abort(self, seq: Sequence) -> None:
        """Drop a sequence wherever it is (client disconnected)."""
        if seq.status is SequenceStatus.RUNNING:
            self.finish(seq, "abort")
        elif seq.status is SequenceStatus.WAITING:
            if seq in self.restoring:
                self.restoring.remove(seq)
            else:
                self.waiting.remove(seq)
            if seq.restore is not None:
                # The copy cannot be cancelled: let it finish into the cache, then let go.
                self._abandoned.append(seq.restore)
                seq.restore = None
            self.blocks.free(seq.seq_id)
            seq.status = SequenceStatus.FINISHED
            seq.finish_reason = "abort"

    # ------------------------------------------------------------- host tier

    def _poll_restores(self) -> None:
        """Finalize completed host-tier copies; their sequences rejoin the queue's front."""
        self.host.poll()
        ready = [seq for seq in self.restoring if seq.restore.transfer.done]
        if ready:
            self.restoring = [seq for seq in self.restoring if not seq.restore.transfer.done]
            now = self.clock()
            for seq in reversed(ready):
                seq.restore_seconds += now - seq.restore_began
                self.cache.finish_restore(seq.restore)
                self.waiting.appendleft(seq)
        for ticket in [t for t in self._abandoned if t.transfer.done]:
            self.cache.finish_restore(ticket)
            self.cache.release_restore(ticket)
            self._abandoned.remove(ticket)

    def _start_restore(self, seq: Sequence) -> bool:
        """Begin copying ``seq``'s demoted history back from the host, if it has any.

        Called right after the sequence adopted what the GPU cache has. The GPU blocks for
        the copy are taken from the free pool, evicting colder cached KV if necessary; if
        even that is not enough the history is recomputed instead.
        """
        blocks, cache, size = self.blocks, self.cache, self.blocks.block_size
        if seq.num_cached % size:
            return False
        limit = (len(seq.tokens) - 1) // size
        plan = cache.host_continuation(seq.tokens, seq.num_cached // size, limit)
        count = sum(matched for _, matched in plan)
        if count < self.config.min_restore_blocks:
            return False
        if blocks.num_free < count:
            cache.evict(count - blocks.num_free)
        if blocks.num_free < count:
            return False
        seq.restore = cache.begin_restore(plan, blocks.allocate(count))
        seq.restore_began = self.clock()
        self.num_restores += 1
        return True

    # ---------------------------------------------------------- prefix cache

    def _sync(self, seq: Sequence) -> int:
        """Exchange full blocks with the prefix cache; return the tokens newly adopted."""
        before = seq.num_cached
        seq.num_cached = self.cache.sync_sequence(seq.seq_id, seq.tokens, before)
        seq.synced_blocks = seq.num_cached // self.blocks.block_size
        return seq.num_cached - before

    def commit(self, seqs: Iterable[Sequence]) -> None:
        """Publish the blocks a step filled. Call after the step's tokens are appended."""
        if self.cache is None:
            return
        size = self.blocks.block_size
        for seq in seqs:
            if seq.num_cached // size > seq.synced_blocks:
                self._sync(seq)

    # ------------------------------------------------------------- admission

    def _next_candidate(self) -> Sequence:
        """The waiting sequence to try to admit next."""
        head = self.waiting[0]
        if self.cache is None or not self.config.cache_aware or len(self.waiting) == 1:
            return head
        if (
            head.num_preemptions
            or self.clock() - head.arrival_time >= self.config.starvation_seconds
        ):
            return head
        best, best_length = head, self.cache.match_length(head.tokens)
        for seq in islice(self.waiting, 1, self.config.admission_window):
            if seq.num_preemptions:
                continue
            length = self.cache.match_length(seq.tokens)
            if length > best_length:
                best, best_length = seq, length
        return best

    def _admit(self) -> None:
        """Move waiting sequences to running while slots and memory allow."""
        if not self.waiting:
            return
        blocks, cache = self.blocks, self.cache
        reserve = int(blocks.num_usable * self.config.watermark)
        # Blocks that running sequences still have to allocate to finish their prefill.
        promised = sum(
            blocks.blocks_missing(s.seq_id, len(s.tokens))
            for s in self.running
            if s.num_pending > 1
        )
        while self.waiting and len(self.running) < self.config.max_num_seqs:
            available = blocks.num_free + blocks.num_evictable - promised - reserve
            if available <= 0 and self.running:
                break
            seq = self._next_candidate()
            adopted = self._sync(seq) if cache is not None else 0
            if self.host is not None and seq.restore is None and self._start_restore(seq):
                # Its history is on its way back from host memory; meanwhile it keeps the
                # GPU prefix it adopted and others are admitted past it.
                self.waiting.remove(seq)
                self.restoring.append(seq)
                seq.restore_hit_tokens += adopted
                continue
            needed = blocks.blocks_missing(seq.seq_id, len(seq.tokens) + 1)
            # Adopting cached blocks pinned them, so they no longer count as evictable.
            fits = blocks.num_free + blocks.num_evictable - promised - reserve >= needed
            # With nothing running the candidate must be admitted, or nothing ever would be.
            if not fits and self.running:
                if seq.restore is not None:
                    # Restored KV stays pinned for it; it is admitted when memory frees up.
                    seq.restore_hit_tokens += adopted
                elif adopted:
                    blocks.free(seq.seq_id)
                    seq.num_cached = seq.synced_blocks = 0
                break
            self.waiting.remove(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            if seq.admitted_at is None:
                seq.admitted_at = self.clock()
            promised += needed
            if seq.restore is not None:
                cache.release_restore(seq.restore)  # the sequence's table holds them now
                seq.restore = None
            adopted += seq.restore_hit_tokens
            seq.restore_hit_tokens = 0
            if self.config.reuse_min_tokens is not None and not seq.num_preemptions:
                # Not after a preemption: the pinned sources may be what it lacked room for.
                self._plan_reuse(seq)
            if cache is not None:
                stats = cache.stats
                stats.lookups += 1
                stats.lookup_tokens += len(seq.tokens) - 1
                stats.hit_tokens += adopted
                stats.hits += adopted > 0
                if not seq.num_preemptions:
                    seq.cache_hit_tokens = adopted

    # --------------------------------------------------- position-independent

    def _plan_reuse(self, seq: Sequence) -> None:
        """Find KV for ``seq``'s uncached tokens that is cached at other positions; pin it."""
        size, ratio = self.blocks.block_size, self.config.reuse_recompute_ratio
        runs = self.cache.find_reuse(
            seq.tokens, seq.num_cached, len(seq.tokens) - 1, self.config.reuse_min_tokens
        )
        for dest_start, source_start, source_blocks in runs:
            head = math.ceil(ratio * len(source_blocks) * size)
            if head >= len(source_blocks) * size:
                continue
            self.blocks.retain(source_blocks)
            seq.reuse.append(ReuseSpan(dest_start, source_start, head, source_blocks, size))

    def _drop_reuse(self, seq: Sequence, before: int | None) -> None:
        """Unpin and forget planned reuse whose copy would start before ``before``.

        ``None`` drops all of it. Needed when the sequence got past the point where a copy
        begins some other way — the prefix cache caught up with those positions — or will
        not get there at all.
        """
        kept = []
        for span in seq.reuse:
            if before is None or span.copy_start < before:
                self.blocks.release(span.source_blocks)
            else:
                kept.append(span)
        seq.reuse = kept

    def finish_reuse(self, span: ReuseSpan) -> None:
        """The engine has copied ``span``'s KV: unpin its source."""
        self.blocks.release(span.source_blocks)

    def _schedule_reuse(self, seq: Sequence, plan: StepPlan) -> bool:
        """Emit the copies ``seq`` has reached. Returns ``False`` if it was preempted."""
        self._drop_reuse(seq, seq.num_cached)
        while seq.reuse and seq.reuse[0].copy_start == seq.num_cached:
            span = seq.reuse[0]
            if not self._reserve(seq, span.end):
                return False
            seq.reuse.pop(0)
            seq.num_cached = span.end
            plan.copies.append((seq, span))
        return True

    # ------------------------------------------------------------ preemption

    def _preempt(self, seq: Sequence) -> None:
        """Take a running sequence's KV away and put it at the front of the queue."""
        self.running.remove(seq)
        self._drop_reuse(seq, None)
        self.blocks.free(seq.seq_id)
        seq.num_cached = seq.synced_blocks = 0
        seq.num_preemptions += 1
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def _reserve(self, seq: Sequence, num_tokens: int) -> bool:
        """Allocate blocks so ``seq`` can hold ``num_tokens``.

        Evicts unreferenced cache blocks first and preempts younger sequences after that.
        Returns ``False`` if ``seq`` itself had to be preempted.
        """
        blocks = self.blocks
        while True:
            try:
                blocks.ensure_capacity(seq.seq_id, num_tokens)
                return True
            except OutOfBlocks:
                short = blocks.blocks_missing(seq.seq_id, num_tokens) - blocks.num_free
                if self.cache is not None and self.cache.evict(short):
                    continue
                victim = self.running[-1]
                self._preempt(victim)
                if victim is seq:
                    return False

    # -------------------------------------------------------------- planning

    def schedule(self) -> StepPlan:
        """Admit what fits, then select and allocate this step's work."""
        if self.host is not None:
            self._poll_restores()
        self._admit()
        plan = StepPlan()
        num_decode = sum(1 for s in self.running if s.num_pending == 1)
        prefill_budget = max(self.config.max_batched_tokens - num_decode, 0)
        # Chunking exists to bound the gap between decode steps; with nobody decoding
        # there is no gap to protect, and small chunks would only add per-pass overhead.
        if self.config.max_prefill_tokens is not None and num_decode:
            prefill_budget = min(prefill_budget, self.config.max_prefill_tokens)
        index = 0
        while index < len(self.running):
            seq = self.running[index]
            index += 1
            if seq.num_pending == 1:
                if self._reserve(seq, seq.num_cached + 1):
                    plan.decode.append(seq)
                continue
            if prefill_budget == 0:
                continue
            if self.cache is not None:
                # Catch up with whatever other sequences cached since this one last looked.
                self.cache.stats.catchup_tokens += self._sync(seq)
            count = min(seq.num_pending, prefill_budget)
            if seq.reuse:
                if not self._schedule_reuse(seq, plan):
                    continue
                if seq.num_pending == 1:
                    continue  # the copies brought it to its last token: it decodes next step
                count = min(seq.num_pending, prefill_budget)
                if seq.reuse:  # compute up to the end of the next run's recomputed head
                    count = min(count, seq.reuse[0].copy_start - seq.num_cached)
            if self._reserve(seq, seq.num_cached + count):
                plan.prefill.append((seq, count))
                prefill_budget -= count
        if plan.copies:
            # A sequence can be preempted after its copy was queued (by its own next
            # allocation in this very pass): its table is gone, the copy must not run.
            kept = []
            for seq, span in plan.copies:
                if seq.status is SequenceStatus.RUNNING:
                    kept.append((seq, span))
                else:
                    self.blocks.release(span.source_blocks)
            plan.copies = kept
        return plan
