"""Scheduler policy on the CPU: admission, token budget, preemption, fairness."""

from __future__ import annotations

import random

import pytest

from minfer.cache import BlockManager
from minfer.sampling import SamplingParams
from minfer.scheduler import Scheduler, SchedulerConfig, Sequence, SequenceStatus

BLOCK = 4


def make(num_blocks=33, **config) -> tuple[Scheduler, BlockManager]:
    config.setdefault("watermark", 0.0)
    blocks = BlockManager(num_blocks, BLOCK)
    return Scheduler(SchedulerConfig(**config), blocks), blocks


def seq(seq_id: int, prompt_len: int, max_new: int = 8) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        tokens=list(range(prompt_len)),
        prompt_len=prompt_len,
        params=SamplingParams(max_tokens=max_new),
        max_new_tokens=max_new,
    )


def execute(plan, token=7) -> None:
    """Pretend to run the step: cache the tokens and append one where logits are produced."""
    for s, count in plan.prefill:
        done = count == s.num_pending
        s.num_cached += count
        if done:
            s.tokens.append(token)
    for s in plan.decode:
        s.num_cached += 1
        s.tokens.append(token)


def test_prefill_then_decode():
    sched, blocks = make()
    a = seq(0, 10)
    sched.add(a)
    plan = sched.schedule()
    assert plan.prefill == [(a, 10)] and plan.decode == []
    assert a.status is SequenceStatus.RUNNING
    assert len(blocks.table(0)) == 3
    execute(plan)
    plan = sched.schedule()
    assert plan.decode == [a] and plan.prefill == []
    execute(plan)
    assert a.num_cached == 11 and a.num_pending == 1


def test_max_num_seqs_limits_admission():
    sched, _ = make(num_blocks=200, max_num_seqs=3)
    seqs = [seq(i, 5) for i in range(5)]
    for s in seqs:
        sched.add(s)
    plan = sched.schedule()
    assert [s.seq_id for s, _ in plan.prefill] == [0, 1, 2]
    assert [s.seq_id for s in sched.waiting] == [3, 4]
    execute(plan)
    sched.finish(seqs[1], "stop")
    plan = sched.schedule()
    assert [s.seq_id for s in plan.decode] == [0, 2]
    assert [s.seq_id for s, _ in plan.prefill] == [3]


def test_token_budget_chunks_long_prompts():
    sched, _ = make(num_blocks=200, max_batched_tokens=16)
    a, b = seq(0, 30), seq(1, 6)
    sched.add(a)
    sched.add(b)
    plan = sched.schedule()
    assert plan.prefill == [(a, 16)]
    execute(plan)
    assert a.num_cached == 16 and len(a.tokens) == 30  # no token produced yet
    plan = sched.schedule()
    assert plan.prefill == [(a, 14), (b, 2)]
    execute(plan)
    plan = sched.schedule()
    # a decodes (1 token) and b gets the rest of its prompt.
    assert plan.decode == [a] and plan.prefill == [(b, 4)]


def test_decode_tokens_count_against_the_budget():
    sched, _ = make(num_blocks=400, max_batched_tokens=8)
    decoders = [seq(i, 1) for i in range(6)]
    for s in decoders:
        sched.add(s)
    execute(sched.schedule())
    late = seq(10, 20)
    sched.add(late)
    plan = sched.schedule()
    assert len(plan.decode) == 6
    assert plan.prefill == [(late, 2)]


def test_admission_waits_for_memory_without_overtaking():
    sched, blocks = make(num_blocks=11)  # 10 usable blocks = 40 tokens
    a, big, small = seq(0, 23), seq(1, 20), seq(2, 3)
    for s in (a, big, small):
        sched.add(s)
    plan = sched.schedule()
    # a needs 6 blocks; big needs 6 more and does not fit; small must not overtake big.
    assert plan.prefill == [(a, 23)]
    assert [s.seq_id for s in sched.waiting] == [1, 2]
    execute(plan)
    sched.finish(a, "stop")
    assert blocks.num_free == 10
    plan = sched.schedule()
    assert [s.seq_id for s, _ in plan.prefill] == [1, 2]


def test_admission_counts_blocks_promised_to_unfinished_prefills():
    sched, _ = make(num_blocks=11, max_batched_tokens=8)
    a = seq(0, 30)  # 8 blocks in total, allocated chunk by chunk
    sched.add(a)
    execute(sched.schedule())
    assert a.num_cached == 8
    b = seq(1, 10)  # 3 blocks: free now (8 free), but a still needs 6 of them
    sched.add(b)
    plan = sched.schedule()
    assert [s.seq_id for s, _ in plan.prefill] == [0]
    assert list(sched.waiting) == [b]


def test_watermark_keeps_headroom():
    sched, _ = make(num_blocks=101, watermark=0.1)
    a, b = seq(0, 4 * 80), seq(1, 4 * 12)
    sched.add(a)
    sched.add(b)
    plan = sched.schedule()
    # 100 usable, 10 kept free: a (81 blocks incl. the next token) fits, b (13) does not.
    assert [s.seq_id for s, _ in plan.prefill] == [0]


def test_preempts_youngest_and_requeues_at_front():
    sched, blocks = make(num_blocks=7)  # 6 usable
    a, b = seq(0, 11), seq(1, 11)  # 3 blocks each, both exactly fit
    sched.add(a)
    sched.add(b)
    execute(sched.schedule())
    c = seq(2, 2)
    sched.add(c)
    assert blocks.num_free == 0
    execute(sched.schedule())  # token 12 of each still fits in block 3
    assert a.num_cached == 12 and b.num_cached == 12
    plan = sched.schedule()  # a needs a 4th block: b is the youngest and loses its KV
    assert plan.decode == [a] and plan.prefill == []
    assert b.status is SequenceStatus.WAITING and b.num_cached == 0
    assert b.num_preemptions == 1 and sched.num_preemptions == 1
    assert len(b.tokens) == 13  # generated tokens are kept
    assert [s.seq_id for s in sched.waiting] == [1, 2]
    assert blocks.table(1) == []
    execute(plan)
    sched.finish(a, "length")
    plan = sched.schedule()
    assert plan.prefill[0] == (b, 13)  # recompute prompt + output


def test_sole_sequence_is_always_admitted():
    sched, _ = make(num_blocks=5, watermark=0.5)
    a = seq(0, 14)
    sched.add(a)
    assert sched.schedule().prefill == [(a, 14)]


def test_rejects_sequence_larger_than_the_pool():
    sched, _ = make(num_blocks=5)
    with pytest.raises(ValueError):
        sched.add(seq(0, 16))


def test_abort_waiting_and_running():
    sched, blocks = make(num_blocks=50, max_num_seqs=1)
    a, b = seq(0, 6), seq(1, 6)
    sched.add(a)
    sched.add(b)
    execute(sched.schedule())
    sched.abort(b)
    sched.abort(a)
    assert not sched.has_work and blocks.num_free == blocks.num_usable
    assert a.finish_reason == b.finish_reason == "abort"


def test_randomized_workload_completes_under_memory_pressure():
    """Every sequence finishes, in bounded steps, with block accounting intact."""
    rng = random.Random(0)
    sched, blocks = make(num_blocks=41, max_num_seqs=8, max_batched_tokens=24, watermark=0.05)
    todo = [seq(i, rng.randint(1, 40), max_new=rng.randint(1, 60)) for i in range(60)]
    finished = 0
    steps = 0
    while todo or sched.has_work:
        for _ in range(rng.randint(0, 2)):
            if todo:
                sched.add(todo.pop())
        plan = sched.schedule()
        scheduled = [s for s, _ in plan.prefill] + plan.decode
        assert len(set(map(id, scheduled))) == len(scheduled)
        assert sum(c for _, c in plan.prefill) + len(plan.decode) <= 24 + 8
        for s, count in plan.prefill:
            assert blocks.capacity(s.seq_id) >= s.num_cached + count
        for s in plan.decode:
            assert blocks.capacity(s.seq_id) >= s.num_cached + 1
        execute(plan)
        for s in scheduled:
            if s.num_pending == 1 and s.num_generated >= s.max_new_tokens:
                sched.finish(s, "length")
                finished += 1
        blocks.check_invariants(external_refs={})
        steps += 1
        assert steps < 20_000
    assert finished == 60
    assert blocks.num_free == blocks.num_usable
    assert sched.num_preemptions > 0


def test_prefill_chunk_budget_interleaves_long_prompts_with_decodes():
    sched, _ = make(num_blocks=400, max_prefill_tokens=16)
    short = seq(0, 3, max_new=50)
    sched.add(short)
    execute(sched.schedule())
    long_a, long_b = seq(1, 40), seq(2, 20)
    sched.add(long_a)
    sched.add(long_b)
    chunks = []
    for _ in range(4):
        plan = sched.schedule()
        assert plan.decode[0] is short  # the short request decodes in every step
        assert sum(count for _, count in plan.prefill) <= 16
        chunks.append([(s.seq_id, count) for s, count in plan.prefill])
        execute(plan)
    assert chunks == [[(1, 16)], [(1, 16)], [(1, 8), (2, 8)], [(2, 12)]]


def test_prefill_chunk_budget_does_not_apply_without_decoders():
    sched, _ = make(num_blocks=400, max_prefill_tokens=16)
    a = seq(0, 100)
    sched.add(a)
    assert sched.schedule().prefill == [(a, 100)]


# ------------------------------------------------------------------- prefix cache


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_cached(num_blocks=101, **config):
    from minfer.cache import PrefixCache

    config.setdefault("watermark", 0.0)
    blocks = BlockManager(num_blocks, BLOCK)
    cache = PrefixCache(blocks)
    clock = FakeClock()
    return Scheduler(SchedulerConfig(**config), blocks, cache, clock), blocks, cache, clock


def prompt_seq(seq_id: int, tokens: list[int], max_new: int = 4) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        tokens=list(tokens),
        prompt_len=len(tokens),
        params=SamplingParams(max_tokens=max_new),
        max_new_tokens=max_new,
    )


def run_to_completion(sched, seq_obj) -> None:
    while seq_obj.status is not SequenceStatus.FINISHED:
        plan = sched.schedule()
        execute(plan, token=1000 + seq_obj.seq_id)
        sched.commit([s for s, _ in plan.prefill] + plan.decode)
        if seq_obj.num_generated >= seq_obj.max_new_tokens:
            sched.finish(seq_obj, "length")


def test_cache_hit_skips_the_cached_prefix():
    sched, blocks, cache, _ = make_cached()
    system = list(range(100, 140))  # 10 blocks
    first = prompt_seq(0, [*system, 1, 2, 3])
    sched.add(first)
    run_to_completion(sched, first)
    assert blocks.num_evictable == cache.num_blocks > 0

    second = prompt_seq(1, [*system, 7, 8, 9, 10, 11])
    sched.add(second)
    plan = sched.schedule()
    assert plan.prefill == [(second, 5)]  # only the tokens after the shared 40
    assert second.num_cached == 40 and second.cache_hit_tokens == 40
    assert blocks.table(1)[:10] == [b for node in cache.nodes() for b in node.blocks][:10]
    stats = cache.stats
    assert (stats.lookups, stats.hits, stats.hit_tokens) == (2, 1, 40)
    assert stats.lookup_tokens == (43 - 1) + (45 - 1)


def test_cache_aware_admission_prefers_the_longest_cached_prefix():
    sched, _, _, _ = make_cached(max_num_seqs=1)
    system = list(range(100, 140))
    warm = prompt_seq(0, [*system, 1])
    sched.add(warm)
    run_to_completion(sched, warm)

    cold = prompt_seq(1, list(range(500, 530)))
    partial = prompt_seq(2, system[:20] + list(range(600, 620)))
    hot = prompt_seq(3, [*system, 5, 6])
    for s in (cold, partial, hot):
        sched.add(s)
    order = []
    while sched.has_work:
        sched.schedule()
        (current,) = sched.running
        if not order or order[-1] != current.seq_id:
            order.append(current.seq_id)
        run_to_completion(sched, current)
    assert order == [3, 2, 1]


def test_fcfs_when_cache_aware_is_off():
    sched, _, _, _ = make_cached(max_num_seqs=1, cache_aware=False)
    system = list(range(100, 140))
    warm = prompt_seq(0, [*system, 1])
    sched.add(warm)
    run_to_completion(sched, warm)
    cold, hot = prompt_seq(1, list(range(500, 530))), prompt_seq(2, [*system, 5, 6])
    sched.add(cold)
    sched.add(hot)
    sched.schedule()
    assert sched.running == [cold]
    assert hot.num_cached == 0  # still waiting, holds nothing


def test_starvation_guard_restores_arrival_order():
    sched, _, _, clock = make_cached(max_num_seqs=1, starvation_seconds=2.0)
    system = list(range(100, 140))
    warm = prompt_seq(0, [*system, 1])
    sched.add(warm)
    run_to_completion(sched, warm)

    cold = prompt_seq(1, list(range(500, 530)))
    sched.add(cold)
    served = []
    for i in range(6):  # a steady stream of cache-hitting requests arriving behind it
        clock.now += 0.5
        hot = prompt_seq(10 + i, [*system, 50 + i, 60 + i])
        sched.add(hot)
        sched.schedule()
        (current,) = sched.running
        served.append(current.seq_id)
        run_to_completion(sched, current)
    # Hot requests overtake until the cold one has waited 2 s, then it goes first.
    assert served[:3] == [10, 11, 12]
    assert served[3] == 1
    assert cold.status is SequenceStatus.FINISHED


def test_preempted_sequences_are_readmitted_before_cache_hits():
    sched, _, _, _ = make_cached(num_blocks=13, max_num_seqs=2)  # 12 usable blocks
    system = list(range(100, 116))  # 4 blocks
    a, b = prompt_seq(0, [*system, 1], max_new=30), prompt_seq(1, list(range(300, 318)), max_new=30)
    sched.add(a)
    sched.add(b)
    hot = prompt_seq(2, [*system, 9, 9])
    preempted = None
    for _ in range(40):
        plan = sched.schedule()
        execute(plan)
        sched.commit([s for s, _ in plan.prefill] + plan.decode)
        if sched.num_preemptions and preempted is None:
            preempted = sched.waiting[0]
            sched.add(hot)
            break
    assert preempted is b
    sched.finish(a, "length")
    sched.schedule()
    assert sched.running[0] is b  # not the cache-hitting newcomer


def test_eviction_is_tried_before_preemption():
    sched, blocks, cache, _ = make_cached(num_blocks=21)  # 20 usable
    old = prompt_seq(0, list(range(100, 148)))  # 12 blocks, then cached
    sched.add(old)
    run_to_completion(sched, old)
    assert blocks.num_evictable == cache.num_blocks >= 12
    a, b = prompt_seq(1, list(range(200, 230))), prompt_seq(2, list(range(300, 330)))
    sched.add(a)
    sched.add(b)
    plan = sched.schedule()
    # 16 blocks needed, 8 free: the cache gives up blocks, nobody is preempted.
    assert [s.seq_id for s, _ in plan.prefill] == [1, 2]
    assert sched.num_preemptions == 0 and cache.stats.evicted_blocks >= 8
    cache.check_invariants()
    blocks.check_invariants(external_refs=cache.block_refs())


def test_failed_admission_releases_adopted_blocks():
    sched, blocks, cache, _ = make_cached(num_blocks=21, max_num_seqs=4)  # 20 usable
    system = list(range(100, 132))  # 8 blocks
    warm = prompt_seq(0, [*system, 1])
    sched.add(warm)
    run_to_completion(sched, warm)
    hog = prompt_seq(1, list(range(400, 444)), max_new=1)  # 12 blocks: evicts nothing yet
    sched.add(hog)
    execute(sched.schedule())
    # A request that hits the cache but needs more than what is left must not keep its pins.
    big = prompt_seq(2, system + list(range(600, 640)))
    sched.add(big)
    sched.schedule()
    assert big.status is SequenceStatus.WAITING
    assert big.num_cached == 0 and blocks.table(2) == []
    assert blocks.num_evictable == cache.num_blocks
    blocks.check_invariants(external_refs=cache.block_refs())
