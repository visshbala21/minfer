"""The host-memory KV tier: allocator, copies, demotion and restore (CPU only)."""

from __future__ import annotations

import json
import random

import pytest
import torch
from test_engine import assert_all_released
from test_retention import Clock
from tiny import dense_greedy, make_tiny_model

from minfer.cache.block_manager import BlockManager
from minfer.cache.host_tier import ExtentAllocator, HostPoolFull, HostTier
from minfer.cache.kv_store import PagedKVStore
from minfer.cache.radix_tree import PrefixCache
from minfer.cache.retention import SessionHint, SessionRetention
from minfer.engine import Engine, EngineConfig
from minfer.sampling import SamplingParams

SIZE = 4


def test_extent_allocator():
    allocator = ExtentAllocator(10)
    assert allocator.allocate(4) == [(0, 4)]
    assert allocator.allocate(3) == [(4, 3)]
    allocator.free([1, 2, 5])
    assert allocator.num_free == 6
    # First fit, splitting across the holes it finds.
    assert allocator.allocate(4) == [(1, 2), (5, 1), (7, 1)]
    with pytest.raises(HostPoolFull):
        allocator.allocate(3)
    assert allocator.num_free == 2
    allocator.free([0, 3, 4, 6, 1, 2, 5, 7])
    assert allocator.allocate(10) == [(0, 10)]  # everything coalesced back into one extent
    allocator.free([3])
    with pytest.raises(ValueError):
        allocator.free([3])


def make_store(num_blocks: int = 32) -> PagedKVStore:
    store = PagedKVStore(3, num_blocks, SIZE, 2, 8, "cpu", torch.float32)
    generator = torch.Generator().manual_seed(0)
    for tensor in store.tensors.values():
        tensor.copy_(torch.randn(tensor.shape, generator=generator))
    return store


def test_offload_and_restore_round_trip():
    store = make_store()
    tier = HostTier(store, 16, chunk_blocks=3)
    original = store.tensors[torch.float32].clone()
    source = [5, 9, 2, 30, 11, 7, 8]
    slots, transfer = tier.offload(source)
    assert len(slots) == 7 and tier.num_free == 9 and tier.busy
    tier.poll()
    assert transfer.done and not tier.busy
    store.tensors[torch.float32][:, source] = 0.0  # the GPU copy is gone
    target = [20, 21, 3, 4, 12, 13, 14]
    restore = tier.restore(slots, target)
    tier.poll()
    assert restore.done
    restored = store.tensors[torch.float32]
    assert torch.equal(restored[:, target], original[:, source])
    untouched = [b for b in range(32) if b not in source and b not in target]
    assert torch.equal(restored[:, untouched], original[:, untouched])
    tier.free(slots)
    assert tier.num_free == 16
    assert (tier.stats.offloaded_blocks, tier.stats.restored_blocks) == (7, 7)


def test_staging_memory_in_flight_stays_under_its_cap():
    """Past the cap the oldest copies are finalized first; the data is unaffected."""
    store = make_store()
    original = store.tensors[torch.float32].clone()
    tier = HostTier(store, 16, chunk_blocks=2)
    tier.max_staging_bytes = 3 * tier.bytes_per_block  # room for one chunk and a half
    source = [5, 9, 2, 30, 11, 7, 8]
    slots, transfer = tier.offload(source)
    assert tier.stats.staging_waits > 0
    assert tier.stats.peak_staging_bytes <= tier.max_staging_bytes
    store.tensors[torch.float32][:, source] = 0.0
    target = [20, 21, 3, 4, 12, 13, 14]
    restore = tier.restore(slots, target)
    assert tier.staging_bytes <= tier.max_staging_bytes
    tier.poll()
    assert transfer.done and restore.done and not tier.busy and tier.staging_bytes == 0
    assert torch.equal(store.tensors[torch.float32][:, target], original[:, source])


# -------------------------------------------------------------------------- the tree


class Harness:
    """A prefix cache with a host tier over a small CPU store."""

    def __init__(self, gpu_blocks: int = 33, host_blocks: int = 24) -> None:
        self.clock = Clock()
        self.blocks = BlockManager(gpu_blocks, SIZE)
        self.store = make_store(gpu_blocks)
        self.retention = SessionRetention(self.clock)
        self.tier = HostTier(self.store, host_blocks)
        self.cache = PrefixCache(self.blocks, self.retention, self.clock, self.tier)

    def publish(self, name: str, tokens: list[int], pause: bool = True) -> None:
        """A request of session ``name`` computes ``tokens`` and finishes."""
        state = self.retention.begin(name)
        self.blocks.ensure_capacity(name, len(tokens))
        self.cache.sync_sequence(name, [*tokens, 0], len(tokens))
        self.blocks.free(name)
        self.retention.end(state, ["Bash"], False)
        self.cache.tag(tokens, state)
        if pause:
            self.cache.offload(tokens, state)
        self.clock.now += 1.0

    def check(self) -> None:
        self.tier.poll()
        self.cache.check_invariants()
        self.blocks.check_invariants(external_refs=self.cache.block_refs())


def tokens_of(seed: int, count: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1, 1000) for _ in range(count)]


def test_eviction_demotes_what_has_a_host_copy():
    h = Harness()
    a, b = tokens_of(1, 32), tokens_of(2, 32)
    h.publish("a", a)
    h.publish("b", b, pause=False)  # no host copy
    assert h.tier.num_free == 24 - 8 and h.cache.num_blocks == 16
    # "a" is the older entry and its pause has lasted longer: it goes first, but only to
    # the host; its tokens stay in the tree.
    assert h.cache.evict(5) == 5
    assert h.cache.match_length([*a, 0]) == 3 * SIZE
    assert h.cache.num_host_blocks == 5 and h.cache.stats.demoted_blocks == 5
    plan = h.cache.host_continuation([*a, 0], 3, 8)
    assert [(len(node.tokens) // SIZE, matched) for node, matched in plan] == [(5, 5)]
    assert h.cache.host_continuation([*a, 0], 2, 8) == []  # the GPU prefix ends mid-edge
    h.check()
    # "b" has no host copy: evicting it forgets it.
    assert h.cache.evict(3 + 8) == 11
    assert h.cache.match_length([*b, 0]) == 0 and h.cache.num_host_blocks == 8
    assert h.cache.host_continuation([*b, 0], 0, 8) == []
    assert [m for _, m in h.cache.host_continuation([*a, 0], 0, 8)] == [3, 5]
    h.check()


def test_restore_brings_kv_back_bit_for_bit():
    h = Harness()
    a = tokens_of(3, 32)
    h.publish("a", a)
    table = h.cache.sync([*a, 0], [], 8)  # the blocks its KV is in
    before = h.store.tensors[torch.float32][:, table].clone()
    assert h.cache.evict(8) == 8 and h.blocks.num_free == h.blocks.num_usable
    h.store.tensors[torch.float32].zero_()
    plan = h.cache.host_continuation([*a, 0], 0, 6)  # a request that needs six blocks
    assert sum(m for _, m in plan) == 6
    gpu = h.blocks.allocate(6)
    ticket = h.cache.begin_restore(plan, gpu)
    assert h.cache.match_length([*a, 0]) == 0  # not usable until the copy has landed
    assert h.cache.host_continuation([*a, 0], 0, 8) == []  # ... and not restored twice
    h.tier.poll()
    assert ticket.transfer.done
    h.cache.finish_restore(ticket)
    assert h.cache.match_length([*a, 0]) == 6 * SIZE
    assert torch.equal(h.store.tensors[torch.float32][:, gpu], before[:, :6])
    # The sequence adopts the blocks, then the ticket lets go of its references.
    h.blocks.adopt("again", h.cache.sync([*a, 0], [], 6))
    h.cache.release_restore(ticket)
    assert h.blocks.table("again") == gpu and h.blocks.num_evictable == 0
    h.blocks.free("again")
    assert h.blocks.num_evictable == 6
    # The last two blocks are still host-only, below the restored ones.
    assert [m for _, m in h.cache.host_continuation([*a, 0], 6, 8)] == [2]
    h.check()
    # Restored blocks kept their host copies: evicting them again is a demotion.
    assert h.cache.evict(6) == 6 and h.cache.num_host_blocks == 8
    h.check()


def test_recomputed_blocks_revive_a_host_only_node():
    h = Harness()
    a = tokens_of(4, 32)
    h.publish("a", a)
    h.cache.evict(8)
    # A sequence that recomputed the first five blocks itself publishes them.
    h.blocks.ensure_capacity("fresh", 20)
    own = list(h.blocks.table("fresh"))
    assert h.cache.sync_sequence("fresh", [*a[:20], 7, 7], 20) == 20
    assert h.blocks.table("fresh") == own and h.cache.num_blocks == 5
    assert h.cache.match_length([*a, 0]) == 20
    assert [m for _, m in h.cache.host_continuation([*a, 0], 5, 8)] == [3]
    h.blocks.free("fresh")
    h.check()


def test_forgetting_a_parent_drops_the_host_only_kv_below_it():
    h = Harness(host_blocks=4)
    a = tokens_of(5, 32)
    h.publish("a", a[:16], pause=False)  # four blocks, never offloaded
    h.publish("a", a)  # the host pool only has room for four: the first four get copies
    assert h.tier.num_free == 0
    h.publish("a", a, pause=False)
    h.check()
    assert h.cache.evict(8) == 8  # blocks 4-7 have no copy: forgotten; 0-3: demoted
    assert h.cache.num_host_blocks == 4
    assert [m for _, m in h.cache.host_continuation([*a, 0], 0, 8)] == [4]
    h.check()


def test_a_full_host_pool_drops_what_is_needed_later_than_the_newcomer():
    h = Harness(host_blocks=10)
    a, b, c = tokens_of(6, 24), tokens_of(7, 24), tokens_of(8, 24)
    for _ in range(8):
        h.retention.predictor.observe("Bash", 5.0)  # pauses last five seconds
    h.publish("a", a)
    h.cache.evict(6)  # "a" lives only in the host tier now
    h.clock.now = 2.0  # "a" is expected back in three seconds, a new pause lasts five
    h.publish("b", b)
    # "a" is needed sooner than "b": its only copy is not given up for b's backup, and
    # b gets host copies for the four blocks that fit.
    assert h.tier.num_free == 0 and h.cache.num_host_blocks == 6
    assert [m for _, m in h.cache.host_continuation([*a, 0], 0, 6)] == [6]
    h.check()
    h.retention.end(h.retention.begin("a"), [], True)  # "a" declares itself finished
    h.publish("c", c)  # now room is made by dropping the finished session's host-only KV
    assert h.cache.host_continuation([*a, 0], 0, 6) == []
    assert h.tier.num_free == 0 and h.cache.num_host_blocks == 0
    h.check()


# ------------------------------------------------------------------------ the engine


@pytest.fixture(scope="module")
def model():
    return make_tiny_model()


def make_engine(model, clock, host_gb: float, gpu_blocks: int = 41) -> Engine:
    config = EngineConfig(
        attention_backend="reference",
        num_blocks=gpu_blocks,
        block_size=SIZE,
        host_kv_gb=host_gb,
        min_restore_blocks=2,
    )
    return Engine(model, config, eos_token_ids=(), clock=clock)


def run(engine: Engine, prompt: list[int], session: SessionHint | None, max_tokens: int = 8):
    engine.add_request(prompt, SamplingParams(max_tokens=max_tokens), session)
    tokens, cached = [], 0
    while engine.has_work:
        for out in engine.step():
            tokens.append(out.token_id)
            cached = out.cached_tokens
    return tokens, cached


def host_gb_for(model, blocks: int) -> float:
    """A host tier of ``blocks`` blocks for the tiny model, in GiB."""
    config = model.config
    per_block = config.num_layers * 2 * SIZE * config.num_kv_heads * config.head_dim * 4
    return (blocks + 0.5) * per_block / 2**30


@pytest.mark.parametrize("host_blocks", [0, 64])
def test_evicted_session_is_restored_from_host_memory(model, host_blocks):
    rng = random.Random(21)

    def fresh(count: int) -> list[int]:
        return [rng.randrange(4, 199) for _ in range(count)]

    clock = Clock()
    engine = make_engine(model, clock, host_gb_for(model, host_blocks) if host_blocks else 0.0)
    agent, others = fresh(60), [fresh(60), fresh(60)]
    first = dense_greedy(model, agent, 8)
    hint = SessionHint("agent", ("Bash",), False)
    assert run(engine, agent, hint) == (first, 0)
    # Two more sessions pause after it; 40 GPU blocks hold two contexts, so the agent's —
    # the longest paused — is evicted.
    for index, prompt in enumerate(others):
        clock.now += 1.0
        run(engine, prompt, SessionHint(f"other{index}", ("Bash",), False))
    clock.now += 1.0
    followup = [*agent, *first, *fresh(10)]
    output, cached = run(engine, followup, hint)
    assert output == dense_greedy(model, followup, 8)
    metrics = engine.metrics()
    if host_blocks:
        assert cached >= 64  # the whole history came back from the host tier
        # Eviction trims a leaf from its tail, so the head of the history was still on the GPU.
        assert metrics["host_restores_total"] == 1 and metrics["host_restored_blocks_total"] >= 4
    else:
        assert cached < 32  # without the tier, most of it has to be recomputed
    assert_all_released(engine)


def test_request_log_records_each_requests_timeline(model, tmp_path):
    """The request log splits a request's time into queue, restore, prefill and decode."""
    rng = random.Random(5)
    log = tmp_path / "requests.jsonl"
    config = EngineConfig(
        attention_backend="reference",
        num_blocks=41,
        block_size=SIZE,
        host_kv_gb=host_gb_for(model, 64),
        min_restore_blocks=2,
        request_log=str(log),
    )
    engine = Engine(model, config, eos_token_ids=())
    agent = [rng.randrange(4, 199) for _ in range(60)]
    hint = SessionHint("agent", ("Bash",), False)
    first, _ = run(engine, agent, hint)
    for index in range(2):  # evict the agent's history to host memory
        other = [rng.randrange(4, 199) for _ in range(60)]
        run(engine, other, SessionHint(f"o{index}", ("Bash",), False))
    run(engine, [*agent, *first, 7, 8, 9], hint)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["session"] for r in records] == ["agent", "o0", "o1", "agent"]
    for r in records:
        assert 0 <= r["admitted_s"] <= r["first_token_s"] <= r["finished_s"]
        assert r["restore_s"] <= r["admitted_s"] and r["generated"] == 8
        assert r["prefill_computed"] == r["prompt_tokens"] - r["cached_tokens"]
    assert records[0]["restore_s"] == 0 and records[-1]["restore_s"] > 0
    assert records[-1]["cached_tokens"] >= 64


def test_random_sessions_under_gpu_and_host_pressure_match_the_oracle(model):
    """Many paused sessions, a GPU pool and a host pool both too small for them."""
    rng = random.Random(99)
    clock = Clock()
    engine = make_engine(model, clock, host_gb_for(model, 30), gpu_blocks=61)
    cache, tier = engine.prefix_cache, engine.host_tier
    system = [rng.randrange(4, 199) for _ in range(24)]
    histories = {
        f"s{index}": [*system, *(rng.randrange(4, 199) for _ in range(20))] for index in range(8)
    }
    params = SamplingParams(max_tokens=6)
    live: dict[int, tuple[str, list[int]]] = {}
    outputs: dict[int, list[int]] = {}
    remaining = {name: 5 for name in histories}
    for _ in range(400):
        clock.now += rng.random()
        idle = [
            n for n in histories if remaining[n] and all(n != name for name, _ in live.values())
        ]
        if idle and (len(live) < 3) and rng.random() < 0.6:
            name = rng.choice(idle)
            prompt = list(histories[name])
            final = remaining[name] == 1
            hint = SessionHint(name, () if final else ("Bash",), final)
            live[engine.add_request(prompt, params, hint)] = (name, prompt)
        for out in engine.step():
            outputs.setdefault(out.seq_id, []).append(out.token_id)
            if out.finish_reason:
                name, prompt = live.pop(out.seq_id)
                remaining[name] -= 1
                output = outputs.pop(out.seq_id)
                assert output == dense_greedy(model, prompt, 6), name
                histories[name] = [*prompt, *output, *(rng.randrange(4, 199) for _ in range(9))]
        cache.check_invariants()
        engine.blocks.check_invariants()
        if not any(remaining.values()) and not live:
            break
    assert not any(remaining.values())
    metrics = engine.metrics()
    assert metrics["host_demoted_blocks_total"] > 0 and metrics["host_restores_total"] > 0
    while engine.has_work:
        engine.step()
    tier.poll()
    assert_all_released(engine)


# ----------------------------------------------------------------------------- CUDA


@pytest.mark.gpu
@pytest.mark.parametrize("max_staging", [1 << 30, 1 << 20], ids=["uncapped", "capped"])
def test_round_trip_through_pinned_memory_on_the_gpu(max_staging):
    """The asynchronous path: side stream, pinned pool, chunked copies, polling."""
    store = PagedKVStore(4, 80, 16, 2, 128, "cuda", torch.bfloat16)
    tensor = store.tensors[torch.bfloat16]
    tensor.copy_(torch.randn(tensor.shape, device="cuda"))
    original = tensor.clone()
    tier = HostTier(store, 48, chunk_blocks=7, max_staging_bytes=max_staging)
    source = [3, 70, 12, 13, 14, 15, 41, 8, 9, 60, 61, 62, 63, 64, 65, 66, 67, 2, 30, 31]
    slots, transfer = tier.offload(source)
    tensor[:, source] = 0  # overwritten right away: the offload must already hold the bytes
    target = list(range(20, 40))
    restore = tier.restore(slots, target)
    tier.wait()
    assert transfer.done and restore.done and not tier.busy
    kept = [b for b in range(80) if b not in source and b not in target]
    assert torch.equal(tensor[:, target], original[:, source])
    assert torch.equal(tensor[:, kept], original[:, kept])


@pytest.mark.gpu
def test_host_tier_copies_do_not_wait_for_the_engine_stream():
    """B-014: starting an offload or scattering a restore must not block the host."""
    import time

    store = PagedKVStore(4, 64, 16, 2, 128, "cuda", torch.bfloat16)
    tier = HostTier(store, 32, chunk_blocks=8)
    slots, _ = tier.offload(list(range(8)))
    tier.restore(slots, list(range(48, 56)))
    tier.wait()  # every kernel used below has run once (first use can synchronize)
    restore = tier.restore(slots, list(range(32, 40)))
    torch.cuda.synchronize()
    torch.cuda._sleep(1_000_000_000)  # the engine's stream is busy for ~0.5 s
    began = time.perf_counter()
    tier.offload(list(range(8, 16)))
    tier.poll()  # scatters the completed restore
    blocked = time.perf_counter() - began
    torch.cuda.synchronize()
    assert restore.done
    assert blocked < 0.1, f"the host waited {blocked:.3f} s for the GPU"


@pytest.mark.gpu
def test_restore_staging_is_not_reused_before_its_scatter_has_run():
    """B-015: a restore's staging memory must outlive the scatter queued on the engine stream.

    The scatter is held back on the engine's stream; a second restore of the same size is
    started meanwhile. If the first one's staging memory went back to the allocator at once,
    the second copy lands in it before it is read, and the first target gets the wrong KV.
    """
    store = PagedKVStore(4, 64, 16, 2, 128, "cuda", torch.bfloat16)
    tensor = store.tensors[torch.bfloat16]
    tensor.copy_(torch.randn(tensor.shape, device="cuda"))
    original = tensor.clone()
    tier = HostTier(store, 32, chunk_blocks=8)
    first, second = list(range(0, 8)), list(range(8, 16))
    slots_first, _ = tier.offload(first)
    slots_second, _ = tier.offload(second)
    tier.restore(slots_first, list(range(48, 56)))
    tier.wait()  # every kernel used below has run once (first use can synchronize)
    targets_first, targets_second = list(range(32, 40)), list(range(40, 48))
    restore = tier.restore(slots_first, targets_first)
    torch.cuda.synchronize()  # the copy into staging is complete; the scatter is not queued
    torch.cuda._sleep(200_000_000)  # hold the engine stream (~0.1 s)
    tier.poll()  # queues the scatter behind the sleep
    assert restore.done
    tier.restore(slots_second, targets_second)  # same size: its staging could reuse the first's
    tier.wait()
    torch.cuda.synchronize()
    assert torch.equal(tensor[:, targets_first], original[:, first])
    assert torch.equal(tensor[:, targets_second], original[:, second])


@pytest.mark.gpu
def test_engine_on_the_gpu_restores_an_evicted_session_bit_for_bit(tokenizer):
    """Restored KV is the KV that was computed: the continuation cannot differ."""
    from conftest import PRIMARY_MODEL

    def prompt(text: str) -> list[int]:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        return tokenizer(rendered, add_special_tokens=False)["input_ids"]

    filler = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(150))
    agent = prompt("Summarize this log, then wait for more. " + filler)
    others = [
        prompt(f"Session {i}. Count the items. " + filler[::-1][: 9000 + i]) for i in range(3)
    ]
    outputs = {}
    for name, blocks, host in (("roomy", 4096, 0.0), ("tight", 400, 2.0)):
        config = EngineConfig(num_blocks=blocks, host_kv_gb=host)
        engine = Engine.from_pretrained(PRIMARY_MODEL, config)
        hint = SessionHint("agent", ("Bash",), False)
        first, _ = run(engine, agent, hint, max_tokens=24)
        for index, other in enumerate(others):
            run(engine, other, SessionHint(f"other{index}", ("Bash",), False), max_tokens=8)
        followup = [*agent, *first, *prompt("Now list three of the items.")]
        outputs[name] = (first, *run(engine, followup, hint, max_tokens=48))
        metrics = engine.metrics()
        if host:
            assert metrics["host_demoted_blocks_total"] > 0
            assert metrics["host_restores_total"] >= 1
            engine.host_tier.wait()
        assert_all_released(engine)
        del engine
        torch.cuda.empty_cache()
    roomy, tight = outputs["roomy"], outputs["tight"]
    assert tight[0] == roomy[0] and tight[1] == roomy[1]
    assert tight[2] >= len(agent) - 16  # all of the history but its last partial block
