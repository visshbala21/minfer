"""Phase 2 acceptance on the GPU: the real model through FlashInfer paged attention.

* 64 concurrent requests complete (with ample memory, and under memory pressure).
* Greedy outputs are compared token-for-token with vLLM on a 20-prompt fixture
  (``tests/fixtures/vllm_greedy.json``, produced by ``bench/vllm_greedy_fixture.py``).
  bf16 inference is not bit-reproducible across engines — or across batch compositions of
  one engine: vLLM's own serial and batched outputs in the fixture differ for about half
  of the prompts. The assertion is therefore that every divergence is a float32 near-tie
  between the float32 model's two best tokens. See DESIGN.md M-002.
"""

from __future__ import annotations

import json
import random

import pytest
import torch
from conftest import PRIMARY_MODEL, TESTS_DIR

from bench.vllm_equivalence import compare, teacher_forced
from minfer.engine import Engine, EngineConfig
from minfer.model import load_model
from minfer.sampling import SamplingParams

pytestmark = pytest.mark.gpu

# Largest float32 logit gap between two tokens that bf16 arithmetic is allowed to flip.
# Measured worst case on the fixture is 0.33 (bench/results/vllm_greedy_equivalence.json);
# bf16 logits are individually up to ~1.2 away from float32 on the same data.
NEAR_TIE_MARGIN = 0.5


@pytest.fixture(scope="module")
def engine():
    return Engine.from_pretrained(PRIMARY_MODEL, EngineConfig(num_blocks=8192))


@pytest.fixture(scope="module")
def reference():
    return load_model(PRIMARY_MODEL, dtype=torch.float32)


@pytest.fixture(scope="module")
def fixture():
    return json.loads((TESTS_DIR / "fixtures" / "vllm_greedy.json").read_text())


def chat_prompts(tokenizer, count: int) -> list[list[int]]:
    topics = ["rivers", "prime numbers", "volcanoes", "TCP", "jazz", "enzymes", "chess", "tides"]
    prompts = []
    for i in range(count):
        text = f"Request {i}: write {1 + i % 4} sentences about {topics[i % len(topics)]}."
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        prompts.append(tokenizer(rendered, add_special_tokens=False)["input_ids"])
    return prompts


def mixed_params(count: int) -> list[SamplingParams]:
    rng = random.Random(0)
    params = []
    for i in range(count):
        max_tokens = rng.randint(8, 96)
        if i % 3 == 0:
            params.append(SamplingParams(max_tokens=max_tokens))
        else:
            params.append(
                SamplingParams(max_tokens=max_tokens, temperature=0.8, top_p=0.95, top_k=40, seed=i)
            )
    return params


def check_outputs(engine: Engine, outputs: list[list[int]], params: list[SamplingParams]) -> None:
    vocab = engine.model.config.vocab_size
    for out, p in zip(outputs, params, strict=True):
        assert 1 <= len(out) <= p.max_tokens
        assert all(0 <= t < vocab for t in out)
        assert len(out) == p.max_tokens or out[-1] in engine.eos_token_ids
        assert not any(t in engine.eos_token_ids for t in out[:-1])
    assert not engine.has_work
    blocks, cache = engine.blocks, engine.prefix_cache
    assert blocks.num_free + blocks.num_evictable == blocks.num_usable
    blocks.check_invariants(external_refs=cache.block_refs() if cache else {})


def test_64_concurrent_requests_complete(engine, tokenizer):
    prompts = chat_prompts(tokenizer, 64)
    params = mixed_params(64)
    finished_before = engine.stats.requests_finished
    outputs = engine.generate(prompts, params)
    check_outputs(engine, outputs, params)
    assert engine.stats.requests_finished - finished_before == 64
    assert engine.scheduler.num_preemptions == 0
    # The model produces language, not noise: outputs decode to mostly-ASCII prose.
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert len(text.split()) >= 3 and text.isascii()


def test_64_concurrent_requests_complete_under_memory_pressure(tokenizer):
    # 100 blocks of 16 tokens hold 1,600 tokens; the 64 requests need about 5,500 at peak
    # (less what the prefix cache shares of the common chat-template header).
    small = Engine.from_pretrained(PRIMARY_MODEL, EngineConfig(num_blocks=101))
    prompts = chat_prompts(tokenizer, 64)
    params = mixed_params(64)
    outputs = small.generate(prompts, params)
    check_outputs(small, outputs, params)
    assert small.scheduler.num_preemptions > 0


def test_chunked_prefill_and_serial_execution_complete(tokenizer):
    tight = Engine.from_pretrained(
        PRIMARY_MODEL, EngineConfig(num_blocks=2048, max_batched_tokens=48, max_num_seqs=3)
    )
    prompts = chat_prompts(tokenizer, 12)
    params = mixed_params(12)
    check_outputs(tight, tight.generate(prompts, params), params)


def test_engine_logits_track_the_float32_reference(engine, reference, fixture):
    """FlashInfer prefill + decode, fed vLLM's tokens, against cache-free float32."""
    report = teacher_forced(engine, reference, fixture)
    positions = report["positions"]
    assert positions >= 1500
    assert report["minfer_bf16_vs_fp32_mean_abs_logit_error"] < 0.15
    assert report["minfer_bf16_argmax_equals_fp32_argmax"] / positions > 0.97
    assert report["minfer_bf16_argmax_equals_vllm_token"] / positions > 0.97
    for d in report["disagreements"]:
        assert d["both_in_fp32_top2"], d
        assert d["fp32_margin_between_them"] < NEAR_TIE_MARGIN, d


@pytest.mark.parametrize("batched", [False, True], ids=["one-at-a-time", "batched"])
def test_greedy_matches_vllm_up_to_float32_near_ties(engine, reference, fixture, batched):
    assert len(fixture["cases"]) == 20
    report = compare(engine, reference, fixture, batched)
    for case in report:
        if case["identical"]:
            continue
        # Never a length-only difference: there is always a first differing token...
        assert "minfer_token" in case, case
        # ...and at that position both engines chose among float32's two best tokens,
        # which float32 itself separates by less than bf16 can resolve.
        assert case["both_in_fp32_top2"], case
        assert case["fp32_top2_margin"] < NEAR_TIE_MARGIN, case
    assert any(case["identical"] for case in report)


# ------------------------------------------------------------------ Phase 3: CUDA graphs


def _decode_both_ways(engine: Engine, count: int, steps: int, seed: int):
    """Yield ``(eager_logits, graph_logits)`` for ``steps`` decode passes over ``count`` rows.

    Both passes run on identical cache state: a decode pass writes the new token's K/V to
    the same slot either way, so running one after the other is idempotent.
    """
    from minfer.attention import SequenceSpan, build_paged_batch

    rng = random.Random(seed)
    model, blocks, size = engine.model, engine.blocks, engine.config.block_size
    seqs = [
        [rng.randrange(1000, 100_000) for _ in range(rng.randint(1, 200))] for _ in range(count)
    ]
    ids = [("graph-test", i) for i in range(count)]
    try:
        with torch.inference_mode():
            for seq_id, seq in zip(ids, seqs, strict=True):
                blocks.ensure_capacity(seq_id, len(seq) + steps)
            spans = [SequenceSpan(s, 0, blocks.table(i)) for i, s in zip(ids, seqs, strict=True)]
            batch = build_paged_batch(spans, size, engine.device)
            model(batch.token_ids, batch.positions, engine.backend.prefill(batch))
            for _ in range(steps):
                tokens = [rng.randrange(1000, 100_000) for _ in range(count)]
                spans = [
                    SequenceSpan([t], len(s), blocks.table(i))
                    for i, s, t in zip(ids, seqs, tokens, strict=True)
                ]
                batch = build_paged_batch(spans, size, engine.device)
                hidden = model(batch.token_ids, batch.positions, engine.backend.decode(batch))
                eager = model.logits(hidden).float()
                yield eager, engine.decode_graphs.run(spans).float()
                for seq, token in zip(seqs, tokens, strict=True):
                    seq.append(token)
    finally:
        for seq_id in ids:
            blocks.free(seq_id)


@pytest.mark.parametrize("count", [1, 4, 16, 256])
def test_graph_decode_is_bit_identical_to_eager_at_bucket_sizes(engine, count):
    """Replaying the captured graph is the same computation as the eager pass.

    36 steps take every sequence across at least two block boundaries, so the page tables
    the graph reads change between replays.
    """
    assert engine.decode_graphs is not None
    assert engine.decode_graphs.bucket_size(count) == count
    for eager, graph in _decode_both_ways(engine, count, steps=36, seed=count):
        assert torch.equal(eager, graph)


@pytest.mark.parametrize("count", [3, 13, 100])
def test_graph_decode_with_padding_rows_matches_eager(engine, count):
    """A batch padded up to its bucket differs from the eager pass only by bf16 rounding.

    Padding changes the batch dimension of every matmul, and cuBLAS does not promise the
    same rounding for different shapes (the effect behind M-002); it must not change more
    than that.
    """
    assert engine.decode_graphs.bucket_size(count) > count
    for eager, graph in _decode_both_ways(engine, count, steps=20, seed=count):
        difference = (eager - graph).abs()
        assert difference.max().item() < 1.5
        assert difference.mean().item() < 0.05


def test_graphs_off_runs_eagerly(tokenizer):
    eager = Engine.from_pretrained(PRIMARY_MODEL, EngineConfig(num_blocks=2048, cuda_graphs=False))
    assert eager.decode_graphs is None
    prompts = chat_prompts(tokenizer, 8)
    params = mixed_params(8)
    check_outputs(eager, eager.generate(prompts, params), params)


def test_batches_beyond_the_largest_graph_fall_back_to_eager(tokenizer):
    small = Engine.from_pretrained(
        PRIMARY_MODEL, EngineConfig(num_blocks=2048, cuda_graph_batch_sizes=(1, 2, 4))
    )
    assert small.decode_graphs.max_batch_size == 4
    prompts = chat_prompts(tokenizer, 12)
    params = mixed_params(12)
    check_outputs(small, small.generate(prompts, params), params)


# -------------------------------------------------------------- Phase 4: prefix cache


def test_prefix_cache_on_matches_off_up_to_float32_near_ties(reference, tokenizer):
    """Real model, multi-turn sessions: the cache changes how much is computed, not what.

    Exact identity is asserted on the CPU float32 model (tests/test_prefix_cache_engine.py).
    In bf16 a cached block may have been filled by a different kernel or batch than a
    recomputed one, so near-ties can flip; anything else would be a bug.
    """
    from bench.prefix_cache_equivalence import compare, run

    off_engine = Engine.from_pretrained(
        PRIMARY_MODEL, EngineConfig(num_blocks=4096, prefix_cache=False)
    )
    on_engine = Engine.from_pretrained(PRIMARY_MODEL, EngineConfig(num_blocks=4096))
    prompts, off, _ = run(off_engine, tokenizer, 6, 32, None)
    _, on, _ = run(on_engine, tokenizer, 6, 32, prompts)
    requests = sum(map(len, prompts))
    divergences = compare(reference, prompts, off, on)
    for d in divergences:
        assert d["both_in_fp32_top2"], d
        assert d["fp32_top2_margin"] < NEAR_TIE_MARGIN, d
    assert len(divergences) < requests / 2

    stats = on_engine.prefix_cache.stats
    assert stats.lookups == requests
    # Shared system prompts and resent history. The six first turns arrive together and
    # miss (measured: 0.75), so the bound is well below what later turns achieve.
    assert stats.hit_rate > 0.6
    processed_off = off_engine.stats.prefill_tokens + off_engine.stats.decode_tokens
    processed_on = on_engine.stats.prefill_tokens + on_engine.stats.decode_tokens
    assert processed_off - processed_on == stats.prefill_tokens_saved
    on_engine.prefix_cache.check_invariants()


# ------------------------------------------------------- Phase 7: speculative decoding


@pytest.mark.parametrize("speculation", ["ngram", "draft"])
def test_speculation_on_the_real_model(reference, tokenizer, speculation):
    """Greedy output with speculation equals plain greedy up to float32 near-ties.

    Exact identity is asserted on the CPU float32 model (tests/test_speculation.py). Here
    verification runs through the bf16 prefill kernel instead of the decode kernel, so the
    usual near-tie flips apply (DESIGN.md M-002, M-010).
    """
    from bench.vllm_equivalence import reference_margin

    text = "Repeat this list three times, exactly: alpha, beta, gamma, delta, epsilon."
    prompts = [
        *chat_prompts(tokenizer, 6),
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
        ),
    ]
    prompts = [list(p["input_ids"]) if not isinstance(p, list) else p for p in prompts]
    params = SamplingParams(max_tokens=64)
    plain = Engine.from_pretrained(PRIMARY_MODEL, EngineConfig(num_blocks=4096))
    expected = plain.generate(prompts, params)
    del plain
    engine = Engine.from_pretrained(
        PRIMARY_MODEL, EngineConfig(num_blocks=4096, speculation=speculation)
    )
    outputs = engine.generate(prompts, params)
    stats = engine.stats
    assert stats.spec_rounds > 0 and 0 < stats.spec_accepted <= stats.spec_proposed
    assert stats.steps < sum(map(len, outputs))  # some steps emitted several tokens
    for prompt, ours, theirs in zip(prompts, outputs, expected, strict=True):
        if ours == theirs:
            continue
        at = next(i for i, (x, y) in enumerate(zip(ours, theirs, strict=False)) if x != y)
        margin = reference_margin(reference, prompt + ours[:at], ours[at], theirs[at])
        gap = abs(margin["fp32_logit_minfer_token"] - margin["fp32_logit_vllm_token"])
        assert gap < 1.5, (speculation, at, margin)
    check_outputs(engine, outputs, [params] * len(prompts))


# ------------------------------------------------- Phase 8: Triton paged-attention kernel


@pytest.mark.parametrize(
    ("batch", "context", "splits"),
    [(1, 1, 1), (3, 17, 1), (5, 200, 4), (2, 3000, 16), (64, 130, None), (1, 5000, None)],
)
def test_triton_decode_kernel_matches_flashinfer_and_float32(batch, context, splits):
    """Same attention as FlashInfer's decode kernel, to bf16 rounding; float32 is the judge."""
    from bench.triton_kernel import BLOCK, HEAD_DIM, NUM_HEADS, NUM_KV_HEADS, make_case
    from bench.triton_kernel import reference as exact_attention
    from minfer.attention.flashinfer_backend import flashinfer
    from minfer.attention.triton_decode import paged_decode_attention

    case = make_case(batch, context, torch.Generator().manual_seed(batch * 1000 + context))
    ours = paged_decode_attention(
        case["q"], case["cache"], case["table"], case["lengths"], splits=splits
    ).float()
    workspace = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        case["indptr"],
        case["indices"],
        case["last_page_len"],
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    theirs = wrapper.run(case["q"], case["cache"]).float()
    exact = exact_attention(case)
    scale = exact.abs().mean().item()
    # Tolerance: bf16 has 8 significant bits; outputs are averages of O(1) values.
    assert (ours - theirs).abs().max().item() < 0.02
    assert (ours - exact).abs().max().item() < 0.02
    assert (ours - exact).abs().mean().item() < 0.01 * max(scale, 0.05)
    assert (ours - exact).abs().mean().item() < 2.0 * (theirs - exact).abs().mean().item() + 1e-4


def test_triton_decode_kernel_handles_ragged_lengths():
    """Sequences of different lengths in one batch, including one shorter than a chunk."""
    from bench.triton_kernel import BLOCK, make_case
    from bench.triton_kernel import reference as exact_attention
    from minfer.attention.triton_decode import paged_decode_attention

    lengths = [1, 15, 16, 17, 333, 2000]
    case = make_case(len(lengths), max(lengths), torch.Generator().manual_seed(7))
    case["lengths"] = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    out = paged_decode_attention(case["q"], case["cache"], case["table"], case["lengths"], splits=8)
    for row, length in enumerate(lengths):
        single = {
            "q": case["q"][row : row + 1],
            "cache": case["cache"],
            "table": case["table"][row : row + 1, : -(-length // BLOCK)],
            "lengths": case["lengths"][row : row + 1],
        }
        expected = exact_attention(single)
        assert (out[row : row + 1].float() - expected).abs().max().item() < 0.02, length


def test_engine_with_triton_decode_kernel(reference, fixture, tokenizer):
    engine = Engine.from_pretrained(
        PRIMARY_MODEL, EngineConfig(num_blocks=8192, decode_kernel="triton")
    )
    report = teacher_forced(engine, reference, fixture)
    positions = report["positions"]
    assert report["minfer_bf16_vs_fp32_mean_abs_logit_error"] < 0.15
    assert report["minfer_bf16_argmax_equals_fp32_argmax"] / positions > 0.97
    for d in report["disagreements"]:
        assert d["fp32_margin_between_them"] < NEAR_TIE_MARGIN, d
    prompts = chat_prompts(tokenizer, 40)
    params = mixed_params(40)
    check_outputs(engine, engine.generate(prompts, params), params)
    # The captured graphs replay the same kernel the eager path runs.
    for eager, graph in _decode_both_ways(engine, 16, steps=20, seed=3):
        assert torch.equal(eager, graph)


# ------------------------------------------------------------------------ Phase 9: FP8


@pytest.mark.parametrize(
    "overrides",
    [
        {"kv_cache_dtype": "fp8"},
        {"weight_dtype": "fp8"},
        {"kv_cache_dtype": "fp8", "weight_dtype": "fp8"},
    ],
    ids=["fp8-kv", "fp8-weights", "fp8-both"],
)
def test_fp8_engine_tracks_the_float32_model(reference, fixture, tokenizer, overrides):
    """FP8 moves logits by quantization noise only (it once did far more: DESIGN.md B-005)."""
    engine = Engine.from_pretrained(PRIMARY_MODEL, EngineConfig(num_blocks=8192, **overrides))
    report = teacher_forced(engine, reference, fixture)
    positions = report["positions"]
    # Measured: 0.19 (KV), 0.26 (weights), 0.33 (both), against 0.05 for bf16.
    assert report["minfer_bf16_vs_fp32_mean_abs_logit_error"] < 0.5
    assert report["minfer_bf16_argmax_equals_fp32_argmax"] / positions > 0.93
    if "kv_cache_dtype" in overrides:
        calibration = engine.kv_calibration
        # Layer 0's keys cannot take FP8 (logit error ~15); every other layer can.
        assert calibration.high_precision_layers == (0,)
        assert calibration.logit_error[0] > 5 and calibration.logit_error[1:].max() < 1
        assert engine.store.layer(0).dtype == torch.bfloat16
        assert engine.store.layer(1).dtype == torch.float8_e4m3fn
        plain_bytes = 28 * 2
        assert engine.store.nbytes * plain_bytes == (2 + 27) * (
            engine.store.layer(0).numel() * 28 * 2
        )
    prompts = chat_prompts(tokenizer, 24)
    params = mixed_params(24)
    check_outputs(engine, engine.generate(prompts, params), params)


# ----------------------------------------------------------- Phase 11: fused kernels


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_fused_kernels_match_the_reference_modules(dtype):
    """Each fused kernel against the PyTorch modules it replaces, on real weights."""
    from minfer.model import fused
    from minfer.model.qwen2 import apply_rope

    model = load_model(PRIMARY_MODEL, dtype=dtype)
    config = model.config
    generator = torch.Generator().manual_seed(0)
    rows = 37
    x = (torch.randn(rows, config.hidden_size, generator=generator) * 2).to("cuda", dtype)
    delta = torch.randn(rows, config.hidden_size, generator=generator).to("cuda", dtype)
    exact = dtype == torch.bfloat16  # in bf16 the reference's own roundings are replicated
    tolerance = 0.0 if exact else 2e-6

    norm = model.layers[3].input_layernorm
    total, normed = fused.add_rms_norm(x, delta, norm.weight, config.rms_norm_eps)
    assert torch.equal(total, x + delta)
    assert (normed.float() - norm(x + delta).float()).abs().max().item() <= tolerance
    same, normed = fused.add_rms_norm(x, None, norm.weight, config.rms_norm_eps)
    assert same is x and (normed.float() - norm(x).float()).abs().max().item() <= tolerance

    mlp = model.layers[3].mlp
    gate, up = mlp.gate_proj(x), mlp.up_proj(x)
    activated = fused.silu_mul(torch.cat([gate, up], dim=-1))
    reference = torch.nn.functional.silu(gate) * up
    assert (activated.float() - reference.float()).abs().max().item() <= (0.0 if exact else 2e-5)

    attention = model.layers[3].self_attn
    positions = torch.randint(0, 5000, (rows,), generator=generator).to("cuda")
    q0, k0, v0 = attention.q_proj(x), attention.k_proj(x), attention.v_proj(x)
    q, k, v = fused.rope_qkv(
        torch.cat([q0, k0, v0], dim=-1),
        positions,
        model.rope.cos,
        model.rope.sin,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
    )
    cos, sin = model.rope(positions)
    heads = (rows, config.num_heads, config.head_dim)
    kv_heads = (rows, config.num_kv_heads, config.head_dim)
    assert torch.equal(v, v0.view(kv_heads))
    # The fused rotation rounds once where the reference rounds three times: equal up to
    # that rounding (relative to the operands, since the two products can cancel).
    for ours, raw, shape in ((q, q0, heads), (k, k0, kv_heads)):
        expected = apply_rope(raw.view(shape), cos, sin).float()
        bound = raw.view(shape).float().abs().amax(dim=-1, keepdim=True) * (
            2**-6 if exact else 1e-6
        )
        assert ((ours.float() - expected).abs() <= bound + 1e-6).all()


def test_fused_model_matches_the_unfused_model(tokenizer):
    from minfer.attention import DenseCausalAttention

    ids = tokenizer("The quick brown fox jumps over the lazy dog. " * 12)["input_ids"]
    tokens = torch.tensor(ids, device="cuda")
    positions = torch.arange(len(ids), device="cuda")
    for dtype, max_diff, mean_diff in ((torch.float32, 1e-3, 1e-4), (torch.bfloat16, 3.0, 0.15)):
        model = load_model(PRIMARY_MODEL, dtype=dtype)
        with torch.inference_mode():
            dense = DenseCausalAttention(1, len(ids))
            before = model.logits(model(tokens, positions, dense)).float()
            model.fuse()
            model.fuse()  # idempotent
            after = model.logits(model(tokens, positions, dense)).float()
        difference = (after - before).abs()
        assert difference.max().item() < max_diff and difference.mean().item() < mean_diff
        assert (after.argmax(-1) == before.argmax(-1)).float().mean().item() > 0.95
        assert not hasattr(model.layers[0].self_attn, "q_proj")  # merged weights replaced them
        del model


# ------------------------------------------------------------ Phase 11: prefill graphs


def test_prefill_graph_buckets_cover_what_they_claim(engine):
    graphs = engine.prefill_graphs
    assert graphs is not None
    assert graphs.bucket(1, 5) == (2, 32)
    assert graphs.bucket(3, 137) == (4, 192)
    assert graphs.bucket(2, 1024) == (2, 1024)  # an exact fit needs no padding
    # One spare token cannot be shared out among the padding spans of (4, 768).
    assert graphs.bucket(2, 767) == (4, 896)
    assert graphs.bucket(2, 766) == (4, 768)
    assert graphs.bucket(1, 1025) is None  # large passes run eagerly
    assert graphs.bucket(40, 100) is None and graphs.bucket(1, 5000) is None
    for seqs, tokens in graphs._buckets:
        assert tokens >= seqs


@pytest.mark.parametrize(
    "shape", [[5], [100, 30, 7], [150] * 6, [1000], [33] * 20, [1, 1, 1], [600, 400]]
)
def test_prefill_graph_matches_the_eager_pass(engine, shape):
    """Same logits as the eager prefill pass (up to bf16 noise from the padded batch shape)."""
    from minfer.attention import SequenceSpan, build_paged_batch

    rng = random.Random(sum(shape) + len(shape))
    model, blocks, graphs = engine.model, engine.blocks, engine.prefill_graphs
    ids = [("prefill-graph", i) for i in range(len(shape))]

    def tokens(count):
        return [rng.randrange(1000, 100_000) for _ in range(count)]

    try:
        with torch.inference_mode():
            spans, rows, offset = [], [], 0
            for seq_id, new in zip(ids, shape, strict=True):
                cached = rng.randint(0, 600)
                blocks.ensure_capacity(seq_id, cached + new)
                table = blocks.table(seq_id)
                if cached:  # give the span a cached context to attend to
                    batch = build_paged_batch([SequenceSpan(tokens(cached), 0, table)], 16, "cuda")
                    model(batch.token_ids, batch.positions, engine.backend.prefill(batch))
                spans.append(SequenceSpan(tokens(new), cached, table))
                offset += new
                rows.append(offset - 1)
            batch = build_paged_batch(spans, 16, "cuda")
            hidden = model(batch.token_ids, batch.positions, engine.backend.prefill(batch))
            index = torch.tensor(rows, device="cuda")
            eager = model.logits(hidden.index_select(0, index)).float()
            key = graphs.bucket(len(spans), offset)
            assert key is not None
            graph = graphs.run(key, spans, rows).float()
        difference = (graph - eager).abs()
        assert difference.max().item() < 1.5 and difference.mean().item() < 0.06
    finally:
        for seq_id in ids:
            blocks.free(seq_id)


def test_engine_without_prefill_graphs(tokenizer):
    eager = Engine.from_pretrained(
        PRIMARY_MODEL, EngineConfig(num_blocks=2048, prefill_graphs=False)
    )
    assert eager.prefill_graphs is None and eager.decode_graphs is not None
    prompts = chat_prompts(tokenizer, 8)
    params = mixed_params(8)
    check_outputs(eager, eager.generate(prompts, params), params)


def test_engine_with_unified_batches(tokenizer):
    """Decoding sequences ride in the ragged prefill pass; graphs cover up to 256 spans."""
    unified = Engine.from_pretrained(
        PRIMARY_MODEL,
        EngineConfig(num_blocks=4096, unified_batches=True, max_batched_tokens=96),
    )
    graphs = unified.prefill_graphs
    assert graphs.bucket(40, 100) == (64, 128) and graphs.bucket(300, 400) is None
    prompts = chat_prompts(tokenizer, 48)
    params = mixed_params(48)
    check_outputs(unified, unified.generate(prompts, params), params)
