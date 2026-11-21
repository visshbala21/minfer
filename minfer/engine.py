"""The inference engine: model + paged KV cache + scheduler, advanced one step at a time.

:class:`Engine` is synchronous and single-threaded by design. ``add_request`` queues work,
``step`` runs one scheduling iteration (at most one prefill pass and one decode pass) and
returns the tokens it produced. The server drives ``step`` from a dedicated thread; tests
and benchmarks call :meth:`Engine.generate`.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from collections.abc import Sequence as Seq
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from minfer import sampling
from minfer.attention.paged import (
    PagedAttentionBackend,
    ReferencePagedBackend,
    SequenceSpan,
    build_paged_batch,
)
from minfer.cache.block_manager import (
    BlockManager,
    OutOfBlocks,
    kv_bytes_per_block,
    plan_num_blocks,
)
from minfer.cache.host_tier import HostTier
from minfer.cache.kv_store import FP8, PagedKVStore
from minfer.cache.radix_tree import PrefixCache
from minfer.cache.retention import SessionHint, SessionRetention
from minfer.model.config import ModelConfig
from minfer.model.loader import load_model, resolve_model_dir
from minfer.model.qwen2 import Qwen2Model
from minfer.quant.calibration import (
    CALIBRATION_TEXTS,
    CalibrationContext,
    KVCalibration,
    summarize,
)
from minfer.quant.fp8 import quantize_linears
from minfer.sampling import SamplingParams
from minfer.scheduler import Scheduler, SchedulerConfig, Sequence
from minfer.spec.base import Draft, Drafter
from minfer.spec.ngram import PromptLookupDrafter


@dataclass(frozen=True)
class EngineConfig:
    """Engine construction options.

    Attributes:
        block_size: Token positions per KV block.
        num_blocks: Size of the block pool, including the reserved block. ``None`` sizes it
            from the device memory that is free after the weights are loaded.
        kv_memory_fraction: Share of free device memory given to the KV cache when
            ``num_blocks`` is ``None``.
        kv_gb: KV cache size in GiB, instead of ``kv_memory_fraction`` (for experiments
            that give every engine the same KV budget).
        max_num_seqs: Maximum concurrently running sequences.
        max_batched_tokens: Maximum tokens processed per engine step.
        watermark: Fraction of the pool kept free at admission (see ``SchedulerConfig``).
        max_model_len: Context limit; ``None`` uses the model's ``max_position_embeddings``.
        attention_backend: ``"flashinfer"`` (CUDA) or ``"reference"`` (plain PyTorch, any
            device and dtype; slow).
        kv_cache_dtype: ``"fp8"`` stores keys and values in 8-bit E4M3 with calibrated
            per-layer scales, except in the layers calibration finds too sensitive, which
            keep the model dtype (for Qwen2.5-1.5B: one layer of 28, so a token costs 52%
            of its bf16 memory); ``"auto"`` stores everything in the model dtype.
        weight_dtype: ``"fp8"`` converts the transformer's linear layers to FP8 weights
            with dynamically quantized activations (:mod:`minfer.quant.fp8`).
        fused_kernels: Run the transformer blocks through merged projections and fused
            Triton kernels (:meth:`Qwen2Model.fuse`): about 4× fewer kernel launches per
            step. CUDA only; ignored for tensor-parallel shards and FP8 weights.
        decode_kernel: Decode attention kernel of the FlashInfer backend: FlashInfer's own
            or the Triton kernel in :mod:`minfer.attention.triton_decode`. ``"auto"`` picks
            the Triton kernel (faster in the engine, M-011) unless the KV cache is FP8,
            which only FlashInfer's kernels read.
        cuda_graphs: Replay captured CUDA graphs for decode passes (FlashInfer backend
            only). Batches larger than the largest captured size run eagerly.
        prefill_graphs: Also replay captured graphs for prefill passes whose shape fits a
            (sequences, tokens) bucket (needs ``cuda_graphs``; not with an FP8 KV cache).
        cuda_graph_batch_sizes: Decode batch sizes to capture; sizes above ``max_num_seqs``
            are rounded down to it.
        prefill_chunk_tokens: Chunked prefill: while any sequence is decoding, at most this
            many prefill tokens per step, so a long prompt is interleaved with decode steps
            instead of stalling them. ``None`` disables chunking (prefill is bounded by
            ``max_batched_tokens`` only).
        prefix_cache: Share KV blocks between requests with a common token prefix through
            the radix-tree prefix cache.
        cache_aware_scheduling: Admit the waiting request with the longest cached prefix
            first (needs ``prefix_cache``); off means first-come-first-served.
        starvation_seconds: A request that has waited this long is admitted in arrival
            order regardless of cache-aware preference.
        eviction_policy: Which cached KV goes first under memory pressure: ``"session"``
            evicts what is expected to be needed last (finished sessions first, then
            sessions by expected remaining tool-call pause; see
            :mod:`minfer.cache.retention`), ``"lru"`` the least recently used. Requests
            that name no session are ordered by recency under both.
        host_kv_gb: Size of the host-memory KV tier in GiB (0 disables it; needs the
            ``"session"`` eviction policy). A session's cached KV is copied to pinned host
            memory when it pauses for a tool call; if the GPU copy is evicted meanwhile,
            the next request has it copied back instead of recomputing it.
        min_restore_blocks: Shortest run of host-only blocks copied back rather than
            recomputed.
        speculation: Speculative decoding: ``"ngram"`` drafts by prompt lookup, ``"draft"``
            with a small draft model (``draft_model``); ``None`` disables it.
        num_draft_tokens: Maximum tokens drafted per sequence per step. Each request's
            draft length adapts between 1 and this value to its recent acceptance.
        ngram_size: Context length matched by prompt-lookup drafting.
        draft_model: Local directory or Hub id of the draft model.
        draft_num_blocks: KV pool size of the draft model.
        seed: Seed of the engine-level random stream used by unseeded sampling requests.
    """

    block_size: int = 16
    num_blocks: int | None = None
    kv_memory_fraction: float = 0.9
    kv_gb: float | None = None
    max_num_seqs: int = 256
    max_batched_tokens: int = 8192
    watermark: float = 0.01
    max_model_len: int | None = None
    attention_backend: Literal["flashinfer", "reference"] = "flashinfer"
    kv_cache_dtype: Literal["auto", "fp8"] = "auto"
    weight_dtype: Literal["auto", "fp8"] = "auto"
    fused_kernels: bool = True
    decode_kernel: Literal["auto", "flashinfer", "triton"] = "auto"
    cuda_graphs: bool = True
    prefill_graphs: bool = True
    cuda_graph_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    prefill_chunk_tokens: int | None = 2048
    prefix_cache: bool = True
    cache_aware_scheduling: bool = True
    starvation_seconds: float = 1.0
    eviction_policy: Literal["lru", "session"] = "session"
    host_kv_gb: float = 0.0
    min_restore_blocks: int = 8
    speculation: Literal["ngram", "draft"] | None = None
    num_draft_tokens: int = 4
    ngram_size: int = 3
    draft_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    draft_num_blocks: int | None = None
    seed: int = 0


@dataclass(frozen=True)
class StepOutput:
    """A token produced for one request by one engine step.

    Attributes:
        seq_id: The request the token belongs to.
        token_id: The sampled token (a stop token is reported, with ``finish_reason`` set).
        finish_reason: ``None`` while the request continues; ``"stop"`` or ``"length"``.
        cached_tokens: On the final token only: prompt tokens the request took from the
            prefix cache when it was first admitted.
    """

    seq_id: int
    token_id: int
    finish_reason: str | None
    cached_tokens: int = 0


@dataclass
class EngineStats:
    """Monotonic counters, exported by the server's ``/metrics`` endpoint."""

    steps: int = 0
    requests_added: int = 0
    requests_finished: int = 0
    prompt_tokens: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    verify_tokens: int = 0
    generated_tokens: int = 0
    spec_rounds: int = 0
    spec_proposed: int = 0
    spec_accepted: int = 0
    step_seconds: float = 0.0


def read_eos_token_ids(model_dir: Path, fallback: int) -> tuple[int, ...]:
    """End-of-sequence token ids from ``generation_config.json`` (or the model config)."""
    path = model_dir / "generation_config.json"
    if path.exists():
        eos = json.loads(path.read_text()).get("eos_token_id")
        if isinstance(eos, int):
            return (eos,)
        if eos:
            return tuple(eos)
    return (fallback,)


class Engine:
    """Continuous-batching inference engine over a paged KV cache."""

    def __init__(
        self,
        model: Qwen2Model,
        config: EngineConfig | None = None,
        eos_token_ids: Seq[int] | None = None,
        calibration_prompts: Seq[Seq[int]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = model
        self.clock = clock
        self.config = config = config or EngineConfig()
        self.device = model.device
        mc = model.config
        self.eos_token_ids = frozenset(
            eos_token_ids if eos_token_ids is not None else (mc.eos_token_id,)
        )
        self.max_model_len = min(
            config.max_model_len or mc.max_position_embeddings, mc.max_position_embeddings
        )

        fp8_kv = config.kv_cache_dtype == "fp8"
        if fp8_kv and config.decode_kernel == "triton":
            raise ValueError("the Triton decode kernel does not read an FP8 KV cache")
        self.decode_kernel = config.decode_kernel
        if self.decode_kernel == "auto":
            self.decode_kernel = "flashinfer" if fp8_kv else "triton"
        if config.weight_dtype == "fp8":
            quantize_linears(model)
        elif config.fused_kernels and self.device.type == "cuda" and model.shard is None:
            model.fuse()
        calibration = self._calibrate_kv(calibration_prompts or ()) if fp8_kv else None
        self.kv_calibration = calibration
        sensitive = calibration.high_precision_layers if calibration else ()
        num_fp8_layers = mc.num_layers - len(sensitive) if fp8_kv else 0
        num_blocks = config.num_blocks
        if num_blocks is None:
            if self.device.type != "cuda":
                raise ValueError("num_blocks must be given explicitly off-GPU")
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            shape = (model.num_kv_heads, mc.head_dim, config.block_size)
            per_block = kv_bytes_per_block(num_fp8_layers, *shape, 1) + kv_bytes_per_block(
                mc.num_layers - num_fp8_layers, *shape, model.dtype.itemsize
            )
            if config.speculation == "draft" and config.draft_num_blocks is None:
                # The draft model's pool gets as many blocks as the target's: budget both.
                dc = ModelConfig.from_pretrained(resolve_model_dir(config.draft_model))
                per_block += kv_bytes_per_block(
                    dc.num_layers,
                    dc.num_kv_heads,
                    dc.head_dim,
                    config.block_size,
                    model.dtype.itemsize,
                )
            if config.kv_gb is not None:
                num_blocks = int(config.kv_gb * 2**30) // per_block
            else:
                num_blocks = plan_num_blocks(free_bytes, per_block, config.kv_memory_fraction)
        self.blocks = BlockManager(num_blocks, config.block_size)
        self.store = PagedKVStore(
            mc.num_layers,
            num_blocks,
            config.block_size,
            model.num_kv_heads,
            mc.head_dim,
            self.device,
            model.dtype,
            FP8 if fp8_kv else None,
            sensitive,
        )
        self.backend: PagedAttentionBackend = self._make_backend(self.store)
        self.decode_graphs = None
        if calibration is not None:
            # Before any CUDA graph is captured: the scales are baked into the graphs.
            self.store.set_scales(calibration.k_scale, calibration.v_scale)
        if config.attention_backend == "flashinfer" and config.cuda_graphs:
            from minfer.attention.decode_graphs import DecodeGraphs

            sizes = {min(size, config.max_num_seqs) for size in config.cuda_graph_batch_sizes}
            self.decode_graphs = DecodeGraphs(
                model,
                self.store,
                self.backend.workspace,
                batch_sizes=sorted(sizes),
                max_pages_per_sequence=self.blocks.blocks_for(self.max_model_len),
                decode_kernel=self.decode_kernel,
            )
        self.prefill_graphs = None
        if (
            config.attention_backend == "flashinfer"
            and config.cuda_graphs
            and config.prefill_graphs
            and not fp8_kv
        ):
            from minfer.attention.prefill_graphs import PrefillGraphs

            self.prefill_graphs = PrefillGraphs(
                model,
                self.store,
                self.backend.workspace,
                max_pages_per_sequence=self.blocks.blocks_for(self.max_model_len),
            )
        self.retention = None
        if config.prefix_cache and config.eviction_policy == "session":
            self.retention = SessionRetention(clock)
        self.host_tier = None
        if config.host_kv_gb > 0:
            if self.retention is None:
                raise ValueError(
                    "the host tier needs the prefix cache with eviction_policy='session'"
                )
            host_blocks = int(config.host_kv_gb * 2**30) // (self.store.nbytes // num_blocks)
            self.host_tier = HostTier(self.store, host_blocks)
        self.prefix_cache = (
            PrefixCache(self.blocks, self.retention, clock, self.host_tier)
            if config.prefix_cache
            else None
        )
        self.scheduler = Scheduler(
            SchedulerConfig(
                max_num_seqs=config.max_num_seqs,
                max_batched_tokens=config.max_batched_tokens,
                max_prefill_tokens=config.prefill_chunk_tokens,
                watermark=config.watermark,
                cache_aware=config.cache_aware_scheduling,
                starvation_seconds=config.starvation_seconds,
                min_restore_blocks=config.min_restore_blocks,
            ),
            self.blocks,
            self.prefix_cache,
        )
        self.stats = EngineStats()
        self.idle = False  # the last step had nothing to run (e.g. only copies in flight)
        self.drafter: Drafter | None = None
        if config.speculation == "ngram":
            self.drafter = PromptLookupDrafter(config.ngram_size)
        elif config.speculation == "draft":
            from minfer.spec.draft_model import DraftModelDrafter

            self.drafter = DraftModelDrafter.from_pretrained(
                config.draft_model, self, config.draft_num_blocks
            )
        self._sequences: dict[int, Sequence] = {}
        self._next_id = 0
        self._rng = random.Random(config.seed)

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | Path,
        config: EngineConfig | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
    ) -> Engine:
        """Load a model from a local directory or the Hugging Face Hub and wrap it."""
        model_dir = resolve_model_dir(name_or_path)
        model = load_model(model_dir, device=device, dtype=dtype)
        eos = read_eos_token_ids(model_dir, model.config.eos_token_id)
        calibration = None
        if config is not None and config.kv_cache_dtype == "fp8":
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
            calibration = [tokenizer.encode(text).ids for text in CALIBRATION_TEXTS]
        return cls(model, config, eos_token_ids=eos, calibration_prompts=calibration)

    @torch.inference_mode()
    def _calibrate_kv(self, prompts: Seq[Seq[int]]) -> KVCalibration:
        """Measure the model's keys and values on ``prompts`` (before the store exists).

        Each prompt is prefilled once through a small scratch store in the model dtype,
        with a context that records per-layer magnitudes and the attention-logit error
        FP8 keys would cause.
        """
        if not prompts:
            raise ValueError("an FP8 KV cache needs calibration prompts")
        model, size = self.model, self.config.block_size
        mc = model.config
        pages = max(-(-len(prompt) // size) for prompt in prompts)
        scratch = PagedKVStore(
            mc.num_layers,
            pages + 1,
            size,
            model.num_kv_heads,
            mc.head_dim,
            self.device,
            model.dtype,
        )
        backend = self._make_backend(scratch)
        peaks = torch.zeros(mc.num_layers, 2, device=self.device)
        errors = torch.zeros(mc.num_layers, device=self.device)
        for prompt in prompts:
            table = list(range(1, 1 + -(-len(prompt) // size)))
            batch = build_paged_batch([SequenceSpan(list(prompt), 0, table)], size, self.device)
            context = CalibrationContext(backend.prefill(batch), peaks, errors)
            model(batch.token_ids, batch.positions, context)
        return summarize(peaks, errors)

    def _make_backend(self, store: PagedKVStore) -> PagedAttentionBackend:
        """The configured attention backend over ``store``."""
        if self.config.attention_backend == "flashinfer":
            from minfer.attention.flashinfer_backend import FlashInferBackend

            return FlashInferBackend(store, self.model.num_heads, self.decode_kernel)
        return ReferencePagedBackend(store)

    # -------------------------------------------------------------- requests

    def add_request(
        self,
        prompt_ids: Seq[int],
        params: SamplingParams | None = None,
        session: SessionHint | None = None,
    ) -> int:
        """Queue a generation request and return its id.

        ``session`` ties the request to an agent session, which the retention policy uses
        to decide whose cached KV to keep under memory pressure.

        Raises:
            ValueError: If the prompt is empty or leaves no room to generate within the
                context limit or the KV cache.
        """
        params = params or SamplingParams()
        prompt = list(prompt_ids)
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        limit = min(self.max_model_len, self.scheduler.max_sequence_tokens)
        if len(prompt) >= limit:
            raise ValueError(
                f"prompt has {len(prompt)} tokens; at most {limit - 1} fit the context limit"
            )
        seq = Sequence(
            seq_id=self._next_id,
            tokens=prompt,
            prompt_len=len(prompt),
            params=params,
            max_new_tokens=min(params.max_tokens, limit - len(prompt)),
            rng=random.Random(params.seed) if params.seed is not None else None,
        )
        self._next_id += 1
        self.scheduler.add(seq)
        if session is not None and self.retention is not None:
            seq.session = self.retention.begin(session.session_id)
            seq.hint = session
        self._sequences[seq.seq_id] = seq
        self.stats.requests_added += 1
        self.stats.prompt_tokens += len(prompt)
        return seq.seq_id

    def abort(self, seq_id: int) -> None:
        """Cancel a request; a no-op if it already finished."""
        seq = self._sequences.pop(seq_id, None)
        if seq is not None:
            self.scheduler.abort(seq)
            self._end_session(seq)
            if self.drafter is not None:
                self.drafter.release(seq_id)

    def note_session(
        self, session_id: str, tools: Seq[str] | None = None, final: bool | None = None
    ) -> None:
        """Tell the engine what a session is doing after its latest response.

        For servers that learn it only from the generated text (a tool call, or none): the
        request-time :class:`SessionHint` carries the same information up front.
        """
        if self.retention is not None and session_id in self.retention.sessions:
            self.retention.describe(self.retention.sessions[session_id], tools, final)

    def _end_session(self, seq: Sequence) -> None:
        """A request finished: record the session's pause and mark its cached KV."""
        state, hint = seq.session, seq.hint
        if state is None:
            return
        self.retention.end(state, hint.tools, hint.final)
        cache = self.prefix_cache
        cache.tag(seq.tokens, state)
        if self.host_tier is not None:
            # A pause begins: put what this turn added in host memory, so that losing the
            # GPU copy to memory pressure costs a transfer instead of a prefill. A session
            # that is over gives its host copies back.
            if state.final:
                cache.forget_host(seq.tokens)
            else:
                cache.offload(seq.tokens, state)

    @property
    def has_work(self) -> bool:
        """Whether any request is waiting or running, or a host-tier copy needs polling."""
        return self.scheduler.has_work or (self.host_tier is not None and self.host_tier.busy)

    def metrics(self) -> dict[str, float]:
        """A snapshot of counters and gauges (what the server's ``/metrics`` exports)."""
        stats, blocks, scheduler = self.stats, self.blocks, self.scheduler
        snapshot: dict[str, float] = {
            "requests_running": len(scheduler.running),
            "requests_waiting": len(scheduler.waiting),
            "requests_total": stats.requests_added,
            "requests_finished_total": stats.requests_finished,
            "prompt_tokens_total": stats.prompt_tokens,
            "generation_tokens_total": stats.generated_tokens,
            "prefill_tokens_total": stats.prefill_tokens,
            "decode_tokens_total": stats.decode_tokens,
            "verify_tokens_total": stats.verify_tokens,
            "spec_rounds_total": stats.spec_rounds,
            "spec_proposed_tokens_total": stats.spec_proposed,
            "spec_accepted_tokens_total": stats.spec_accepted,
            "engine_steps_total": stats.steps,
            "engine_step_seconds_total": stats.step_seconds,
            "preemptions_total": scheduler.num_preemptions,
            "kv_blocks": blocks.num_usable,
            "kv_blocks_used": blocks.num_used,
            "kv_blocks_evictable": blocks.num_evictable,
            "kv_usage_ratio": blocks.num_used / blocks.num_usable,
        }
        cache = self.prefix_cache
        if cache is not None:
            cs = cache.stats
            snapshot.update(
                prefix_cache_hit_rate=cs.hit_rate,
                prefix_cache_lookup_tokens_total=cs.lookup_tokens,
                prefix_cache_hit_tokens_total=cs.hit_tokens,
                prefix_cache_catchup_tokens_total=cs.catchup_tokens,
                prefill_tokens_saved_total=cs.prefill_tokens_saved,
                prefix_cache_requests_hit_total=cs.hits,
                prefix_cache_blocks=cache.num_blocks,
                prefix_cache_nodes=cache.num_nodes,
                prefix_cache_evicted_blocks_total=cs.evicted_blocks,
            )
        host = self.host_tier
        if host is not None:
            cs = cache.stats
            snapshot.update(
                host_kv_blocks=host.num_blocks,
                host_kv_blocks_used=host.num_blocks - host.num_free,
                host_offloaded_blocks_total=host.stats.offloaded_blocks,
                host_demoted_blocks_total=cs.demoted_blocks,
                host_restored_blocks_total=cs.restored_blocks,
                host_restores_total=scheduler.num_restores,
            )
        return snapshot

    # ------------------------------------------------------------------ step

    @torch.inference_mode()
    def step(self) -> list[StepOutput]:
        """Run one scheduling iteration and return the tokens it produced."""
        started = time.perf_counter()
        plan = self.scheduler.schedule()
        self.idle = plan.empty
        if plan.empty:
            return []
        decode = plan.decode
        verify: list[tuple[Sequence, Draft]] = []
        if self.drafter is not None and decode:
            decode, verify = self._draft(decode)
        sampled: list[Sequence] = []
        rows: list[Tensor] = []
        verify_logits = None
        if plan.prefill or verify:
            verify_logits = self._run_prefill(plan.prefill, verify, sampled, rows)
        if decode:
            self._run_decode(decode, sampled, rows)
        emitted: list[tuple[Sequence, list[int]]] = []
        if sampled:
            logits = rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)
            tokens = self._sample(logits, sampled).tolist()
            for seq, token in zip(sampled, tokens, strict=True):
                seq.tokens.append(token)
                emitted.append((seq, [token]))
        if verify:
            emitted += self._accept(verify, verify_logits)
        # Publish newly filled blocks to the prefix cache before any sequence finishes and
        # gives up its table.
        self.scheduler.commit(plan.decode)
        self.scheduler.commit(seq for seq, _ in plan.prefill)
        outputs: list[StepOutput] = []
        for seq, tokens in emitted:
            outputs += [StepOutput(seq.seq_id, token, None) for token in tokens[:-1]]
            outputs.append(self._emit(seq, tokens[-1]))
        self.stats.generated_tokens += len(outputs)
        self.stats.steps += 1
        self.stats.step_seconds += time.perf_counter() - started
        return outputs

    def _run_prefill(
        self,
        prefill: list[tuple[Sequence, int]],
        verify: list[tuple[Sequence, Draft]],
        sampled: list[Sequence],
        rows: list[Tensor],
    ) -> Tensor | None:
        """One ragged pass over prefill chunks and speculative verification spans.

        Collects next-token logits for the prefill spans that reach their sequence's end,
        and returns the logits of every row of the verification spans (``None`` without
        any): a span of the pending token plus ``k`` drafted tokens yields ``k + 1`` rows,
        row ``i`` being the model's prediction after ``i`` drafted tokens.
        """
        spans = []
        last_rows = []
        offset = 0
        for seq, count in prefill:
            start = seq.num_cached
            table = self.blocks.table(seq.seq_id)
            spans.append(SequenceSpan(seq.tokens[start : start + count], start, table))
            offset += count
            if count == seq.num_pending:
                sampled.append(seq)
                last_rows.append(offset - 1)
            seq.num_cached = start + count
        self.stats.prefill_tokens += offset
        verify_start = offset
        for seq, draft in verify:
            table = self.blocks.table(seq.seq_id)
            spans.append(SequenceSpan([seq.tokens[-1], *draft.tokens], seq.num_cached, table))
            offset += 1 + len(draft.tokens)
        self.stats.verify_tokens += offset - verify_start
        graphs = self.prefill_graphs
        if graphs is not None and not verify:
            key = graphs.bucket(len(spans), offset)
            if key is not None:
                logits = graphs.run(key, spans, last_rows)
                if last_rows:
                    rows.append(logits)
                return None
        batch = build_paged_batch(spans, self.config.block_size, self.device)
        hidden = self.model(batch.token_ids, batch.positions, self.backend.prefill(batch))
        if last_rows:
            index = torch.tensor(last_rows, device=self.device)
            rows.append(self.model.logits(hidden.index_select(0, index)))
        return self.model.logits(hidden[verify_start:]) if verify else None

    def _run_decode(
        self, decode: list[Sequence], sampled: list[Sequence], rows: list[Tensor]
    ) -> None:
        """Decode pass: one token per sequence, logits for every row."""
        spans = []
        for seq in decode:
            spans.append(
                SequenceSpan(seq.tokens[-1:], seq.num_cached, self.blocks.table(seq.seq_id))
            )
            seq.num_cached += 1
        graphs = self.decode_graphs
        if graphs is not None and len(spans) <= graphs.max_batch_size:
            logits = graphs.run(spans)
        else:
            batch = build_paged_batch(spans, self.config.block_size, self.device)
            hidden = self.model(batch.token_ids, batch.positions, self.backend.decode(batch))
            logits = self.model.logits(hidden)
        self.stats.decode_tokens += len(spans)
        sampled.extend(decode)
        rows.append(logits)

    # ------------------------------------------------------------ speculation

    def _draft(self, decode: list[Sequence]) -> tuple[list[Sequence], list[tuple[Sequence, Draft]]]:
        """Ask the drafter for continuations; split ``decode`` into plain and speculative.

        A sequence is verified speculatively only if it got a non-empty draft and its block
        table can be grown to hold it; otherwise it takes the ordinary decode pass.
        """
        maximum = self.config.num_draft_tokens
        limits = []
        for seq in decode:
            if seq.draft_len == 0:
                seq.draft_len = maximum
            # Leave room for the token the target model contributes after the draft.
            budget = min(
                seq.max_new_tokens - seq.num_generated - 1,
                self.max_model_len - len(seq.tokens) - 1,
            )
            limits.append(max(0, min(seq.draft_len, budget)))
        drafts = self.drafter.propose(decode, limits)
        plain, verify = [], []
        for seq, draft in zip(decode, drafts, strict=True):
            if draft.tokens and self._reserve_draft(seq, len(draft.tokens)):
                verify.append((seq, draft))
            else:
                plain.append(seq)
        return plain, verify

    def _reserve_draft(self, seq: Sequence, num_draft: int) -> bool:
        """Grow ``seq``'s table for a draft, without evicting or preempting for it."""
        try:
            self.blocks.ensure_capacity(seq.seq_id, seq.num_cached + 1 + num_draft)
        except OutOfBlocks:
            return False
        return True

    def _accept(
        self, verify: list[tuple[Sequence, Draft]], logits: Tensor
    ) -> list[tuple[Sequence, list[int]]]:
        """Decide how much of each draft stands and append the resulting tokens.

        Greedy requests accept a drafted token iff it is the target's argmax, so their
        output is exactly what non-speculative decoding produces. Sampling requests use
        speculative sampling (accept with probability ``min(1, p/q)``, otherwise resample
        from the normalized residual ``max(p - q, 0)``), which leaves the output
        distribution equal to the target model's.

        "Rolling back" rejected tokens is logical: their K/V was written past the end of
        what ``num_cached`` now covers and is overwritten by the next pass.
        """
        starts, offset = [], 0
        for _, draft in verify:
            starts.append(offset)
            offset += 1 + len(draft.tokens)
        greedy_pick = sampling.greedy(logits).tolist()
        stochastic = [i for i, (seq, _) in enumerate(verify) if not seq.params.greedy]
        resolved = self._accept_sampled(verify, starts, logits, stochastic) if stochastic else {}

        stats = self.stats
        emitted = []
        for index, (seq, draft) in enumerate(verify):
            drafted = draft.tokens
            if index in resolved:
                accepted, bonus = resolved[index]
            else:
                start = starts[index]
                accepted = 0
                while (
                    accepted < len(drafted) and greedy_pick[start + accepted] == drafted[accepted]
                ):
                    accepted += 1
                bonus = greedy_pick[start + accepted]
            stats.spec_rounds += 1
            stats.spec_proposed += len(drafted)
            stats.spec_accepted += accepted
            maximum = self.config.num_draft_tokens
            seq.draft_len = (
                min(maximum, 2 * seq.draft_len)
                if accepted == len(drafted)
                else max(1, accepted + 1)
            )

            tokens = [*drafted[:accepted], bonus]
            # Cut at the first stop token: nothing after it may be emitted.
            stops = self.eos_token_ids if not seq.params.ignore_eos else frozenset()
            for position, token in enumerate(tokens):
                if token in stops or token in seq.params.stop_token_ids:
                    tokens = tokens[: position + 1]
                    break
            # K/V is valid for the pending token and for each accepted draft token; the
            # last appended token always stays pending.
            seq.num_cached += 1 + min(accepted, len(tokens) - 1)
            seq.tokens.extend(tokens)
            emitted.append((seq, tokens))
        return emitted

    def _accept_sampled(
        self,
        verify: list[tuple[Sequence, Draft]],
        starts: list[int],
        logits: Tensor,
        stochastic: list[int],
    ) -> dict[int, tuple[int, int]]:
        """Speculative sampling for the non-greedy sequences: ``index -> (accepted, bonus)``."""
        device = logits.device
        row_index, temperature, top_k, top_p = [], [], [], []
        for index in stochastic:
            seq, draft = verify[index]
            count = 1 + len(draft.tokens)
            row_index += range(starts[index], starts[index] + count)
            temperature += [seq.params.temperature] * count
            top_k += [seq.params.top_k] * count
            top_p += [seq.params.top_p] * count
        target = sampling.warped_probs(
            logits.index_select(0, torch.tensor(row_index, device=device)),
            torch.tensor(temperature, device=device, dtype=torch.float32),
            torch.tensor(top_k, device=device),
            torch.tensor(top_p, device=device, dtype=torch.float32),
        )
        # Probability the target assigns to each drafted token, and the draft's own.
        local, rows, columns, draft_prob = 0, [], [], []
        local_start = {}
        for index in stochastic:
            _, draft = verify[index]
            local_start[index] = local
            for i, token in enumerate(draft.tokens):
                rows.append(local + i)
                columns.append(token)
            if draft.probs is None:
                draft_prob += [1.0] * len(draft.tokens)
            else:
                picked = torch.tensor(draft.tokens, device=device)
                draft_prob += draft.probs.gather(-1, picked[:, None]).squeeze(-1).tolist()
            local += 1 + len(draft.tokens)
        target_prob = target[rows, columns].tolist() if rows else []

        final_rows, final_uniforms, plan = [], [], []
        cursor = 0
        for index in stochastic:
            seq, draft = verify[index]
            rng = seq.rng or self._rng
            accepted = 0
            for _ in draft.tokens:
                ratio = target_prob[cursor + accepted] / draft_prob[cursor + accepted]
                if rng.random() >= ratio:
                    break
                accepted += 1
            cursor += len(draft.tokens)
            row = local_start[index] + accepted
            if accepted < len(draft.tokens):
                # Rejected: resample from the part of the target the draft under-covers.
                rejected = draft.tokens[accepted]
                if draft.probs is None:
                    target[row, rejected] = 0.0
                else:
                    target[row] = (target[row] - draft.probs[accepted]).clamp_(min=0.0)
            final_rows.append(row)
            final_uniforms.append(rng.random())
            plan.append((index, accepted))
        bonus = sampling.sample_from_probs(
            target.index_select(0, torch.tensor(final_rows, device=device)),
            torch.tensor(final_uniforms, device=device, dtype=torch.float64),
        ).tolist()
        return {
            index: (accepted, token) for (index, accepted), token in zip(plan, bonus, strict=True)
        }

    def _sample(self, logits: Tensor, seqs: list[Sequence]) -> Tensor:
        """Pick the next token for each row according to its request's parameters."""
        stochastic = [i for i, seq in enumerate(seqs) if not seq.params.greedy]
        tokens = sampling.greedy(logits)
        if not stochastic:
            return tokens
        params = [seqs[i].params for i in stochastic]
        uniforms = [(seqs[i].rng or self._rng).random() for i in stochastic]
        device = logits.device
        index = torch.tensor(stochastic, device=device)
        drawn = sampling.sample(
            logits.index_select(0, index),
            torch.tensor([p.temperature for p in params], device=device, dtype=torch.float32),
            torch.tensor([p.top_k for p in params], device=device),
            torch.tensor([p.top_p for p in params], device=device, dtype=torch.float32),
            torch.tensor(uniforms, device=device, dtype=torch.float64),
        )
        tokens[index] = drawn
        return tokens

    def _emit(self, seq: Sequence, token: int) -> StepOutput:
        """Report the last token appended to ``seq``; finish it on a stop condition."""
        params = seq.params
        reason = None
        if (token in self.eos_token_ids and not params.ignore_eos) or (
            token in params.stop_token_ids
        ):
            reason = "stop"
        elif seq.num_generated >= seq.max_new_tokens:
            reason = "length"
        if reason is not None:
            self.scheduler.finish(seq, reason)
            self._end_session(seq)
            del self._sequences[seq.seq_id]
            self.stats.requests_finished += 1
            if self.drafter is not None:
                self.drafter.release(seq.seq_id)
            return StepOutput(seq.seq_id, token, reason, seq.cache_hit_tokens)
        return StepOutput(seq.seq_id, token, None)

    # ----------------------------------------------------------- convenience

    def generate(
        self, prompts: Seq[Seq[int]], params: SamplingParams | Seq[SamplingParams] | None = None
    ) -> list[list[int]]:
        """Run a set of requests to completion and return each one's generated tokens.

        All requests are submitted up front; the scheduler batches them as memory allows.
        Stop tokens are included in the returned ids.
        """
        if params is None or isinstance(params, SamplingParams):
            per_request = [params] * len(prompts)
        else:
            per_request = list(params)
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, per_request, strict=True)]
        outputs: dict[int, list[int]] = {seq_id: [] for seq_id in ids}
        while self.has_work:
            for out in self.step():
                if out.seq_id in outputs:
                    outputs[out.seq_id].append(out.token_id)
        return [outputs[seq_id] for seq_id in ids]
