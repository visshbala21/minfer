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
from collections.abc import Sequence as Seq
from dataclasses import dataclass, field
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
from minfer.cache.block_manager import BlockManager, kv_bytes_per_block, plan_num_blocks
from minfer.cache.kv_store import PagedKVStore
from minfer.cache.radix_tree import PrefixCache
from minfer.model.loader import load_model, resolve_model_dir
from minfer.model.qwen2 import Qwen2Model
from minfer.sampling import SamplingParams
from minfer.scheduler import Scheduler, SchedulerConfig, Sequence, StepPlan


@dataclass(frozen=True)
class EngineConfig:
    """Engine construction options.

    Attributes:
        block_size: Token positions per KV block.
        num_blocks: Size of the block pool, including the reserved block. ``None`` sizes it
            from the device memory that is free after the weights are loaded.
        kv_memory_fraction: Share of free device memory given to the KV cache when
            ``num_blocks`` is ``None``.
        max_num_seqs: Maximum concurrently running sequences.
        max_batched_tokens: Maximum tokens processed per engine step.
        watermark: Fraction of the pool kept free at admission (see ``SchedulerConfig``).
        max_model_len: Context limit; ``None`` uses the model's ``max_position_embeddings``.
        attention_backend: ``"flashinfer"`` (CUDA) or ``"reference"`` (plain PyTorch, any
            device and dtype; slow).
        cuda_graphs: Replay captured CUDA graphs for decode passes (FlashInfer backend
            only). Batches larger than the largest captured size run eagerly.
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
        seed: Seed of the engine-level random stream used by unseeded sampling requests.
    """

    block_size: int = 16
    num_blocks: int | None = None
    kv_memory_fraction: float = 0.9
    max_num_seqs: int = 256
    max_batched_tokens: int = 8192
    watermark: float = 0.01
    max_model_len: int | None = None
    attention_backend: Literal["flashinfer", "reference"] = "flashinfer"
    cuda_graphs: bool = True
    cuda_graph_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    prefill_chunk_tokens: int | None = 2048
    prefix_cache: bool = True
    cache_aware_scheduling: bool = True
    starvation_seconds: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class StepOutput:
    """A token produced for one request by one engine step.

    Attributes:
        seq_id: The request the token belongs to.
        token_id: The sampled token (a stop token is reported, with ``finish_reason`` set).
        finish_reason: ``None`` while the request continues; ``"stop"`` or ``"length"``.
    """

    seq_id: int
    token_id: int
    finish_reason: str | None


@dataclass
class EngineStats:
    """Monotonic counters, exported by the server's ``/metrics`` endpoint."""

    steps: int = 0
    requests_added: int = 0
    requests_finished: int = 0
    prompt_tokens: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    generated_tokens: int = 0
    step_seconds: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)


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
    ) -> None:
        self.model = model
        self.config = config = config or EngineConfig()
        self.device = model.device
        mc = model.config
        self.eos_token_ids = frozenset(
            eos_token_ids if eos_token_ids is not None else (mc.eos_token_id,)
        )
        self.max_model_len = min(
            config.max_model_len or mc.max_position_embeddings, mc.max_position_embeddings
        )

        num_blocks = config.num_blocks
        if num_blocks is None:
            if self.device.type != "cuda":
                raise ValueError("num_blocks must be given explicitly off-GPU")
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            per_block = kv_bytes_per_block(
                mc.num_layers, mc.num_kv_heads, mc.head_dim, config.block_size, model.dtype.itemsize
            )
            num_blocks = plan_num_blocks(free_bytes, per_block, config.kv_memory_fraction)
        self.blocks = BlockManager(num_blocks, config.block_size)
        self.store = PagedKVStore(
            mc.num_layers,
            num_blocks,
            config.block_size,
            mc.num_kv_heads,
            mc.head_dim,
            self.device,
            model.dtype,
        )
        self.backend: PagedAttentionBackend
        self.decode_graphs = None
        if config.attention_backend == "flashinfer":
            from minfer.attention.flashinfer_backend import FlashInferBackend

            self.backend = FlashInferBackend(self.store, mc.num_heads)
            if config.cuda_graphs:
                from minfer.attention.decode_graphs import DecodeGraphs

                sizes = {min(size, config.max_num_seqs) for size in config.cuda_graph_batch_sizes}
                self.decode_graphs = DecodeGraphs(
                    model,
                    self.store,
                    self.backend.workspace,
                    batch_sizes=sorted(sizes),
                    max_pages_per_sequence=self.blocks.blocks_for(self.max_model_len),
                )
        else:
            self.backend = ReferencePagedBackend(self.store)
        self.prefix_cache = PrefixCache(self.blocks) if config.prefix_cache else None
        self.scheduler = Scheduler(
            SchedulerConfig(
                max_num_seqs=config.max_num_seqs,
                max_batched_tokens=config.max_batched_tokens,
                max_prefill_tokens=config.prefill_chunk_tokens,
                watermark=config.watermark,
                cache_aware=config.cache_aware_scheduling,
                starvation_seconds=config.starvation_seconds,
            ),
            self.blocks,
            self.prefix_cache,
        )
        self.stats = EngineStats()
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
        return cls(model, config, eos_token_ids=eos)

    # -------------------------------------------------------------- requests

    def add_request(self, prompt_ids: Seq[int], params: SamplingParams | None = None) -> int:
        """Queue a generation request and return its id.

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
        self._sequences[seq.seq_id] = seq
        self.stats.requests_added += 1
        self.stats.prompt_tokens += len(prompt)
        return seq.seq_id

    def abort(self, seq_id: int) -> None:
        """Cancel a request; a no-op if it already finished."""
        seq = self._sequences.pop(seq_id, None)
        if seq is not None:
            self.scheduler.abort(seq)

    @property
    def has_work(self) -> bool:
        """Whether any request is waiting or running."""
        return self.scheduler.has_work

    # ------------------------------------------------------------------ step

    @torch.inference_mode()
    def step(self) -> list[StepOutput]:
        """Run one scheduling iteration and return the tokens it produced."""
        started = time.perf_counter()
        plan = self.scheduler.schedule()
        if plan.empty:
            return []
        sampled: list[Sequence] = []
        rows: list[Tensor] = []
        if plan.prefill:
            self._run_prefill(plan, sampled, rows)
        if plan.decode:
            self._run_decode(plan, sampled, rows)
        if sampled:
            logits = rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)
            tokens = self._sample(logits, sampled).tolist()
            for seq, token in zip(sampled, tokens, strict=True):
                seq.tokens.append(token)
        else:
            tokens = []
        # Publish newly filled blocks to the prefix cache before any sequence finishes and
        # gives up its table.
        self.scheduler.commit(plan.decode)
        self.scheduler.commit(seq for seq, _ in plan.prefill)
        outputs = [self._emit(seq, token) for seq, token in zip(sampled, tokens, strict=True)]
        self.stats.generated_tokens += len(outputs)
        self.stats.steps += 1
        self.stats.step_seconds += time.perf_counter() - started
        return outputs

    def _run_prefill(self, plan: StepPlan, sampled: list[Sequence], rows: list[Tensor]) -> None:
        """Prefill pass; collects logits for the spans that reach their sequence's end."""
        spans = []
        last_rows = []
        offset = 0
        for seq, count in plan.prefill:
            start = seq.num_cached
            spans.append(
                SequenceSpan(
                    seq.tokens[start : start + count], start, self.blocks.table(seq.seq_id)
                )
            )
            offset += count
            if count == seq.num_pending:
                sampled.append(seq)
                last_rows.append(offset - 1)
            seq.num_cached = start + count
        batch = build_paged_batch(spans, self.config.block_size, self.device)
        hidden = self.model(batch.token_ids, batch.positions, self.backend.prefill(batch))
        self.stats.prefill_tokens += batch.num_tokens
        if last_rows:
            index = torch.tensor(last_rows, device=self.device)
            rows.append(self.model.logits(hidden.index_select(0, index)))

    def _run_decode(self, plan: StepPlan, sampled: list[Sequence], rows: list[Tensor]) -> None:
        """Decode pass: one token per sequence, logits for every row."""
        spans = []
        for seq in plan.decode:
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
        sampled.extend(plan.decode)
        rows.append(logits)

    def _sample(self, logits: Tensor, seqs: list[Sequence]) -> Tensor:
        """Pick the next token for each row according to its request's parameters."""
        stochastic = [i for i, seq in enumerate(seqs) if not seq.params.greedy]
        if not stochastic:
            return sampling.greedy(logits)
        tokens = sampling.greedy(logits)
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
        """Report a token already appended to ``seq``; finish it on a stop condition."""
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
            del self._sequences[seq.seq_id]
            self.stats.requests_finished += 1
        return StepOutput(seq.seq_id, token, reason)

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
