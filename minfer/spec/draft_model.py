"""Draft-model speculation: a small model proposes tokens for the target to verify.

The draft model keeps its own paged KV cache (its own block pool, page tables and radix
tree — the target's machinery, instantiated a second time), so that drafting a sequence
costs one small decode pass per drafted token rather than a re-read of the whole context.

Per step, for the sequences that want a draft:

1. **Catch up.** The draft cache is brought in line with the tokens the target has
   confirmed: K/V written for drafted tokens that were then rejected is discarded
   (logically, by lowering the cached length), and the tokens the draft has not seen —
   the whole prompt the first time, usually just the target's last token afterwards — go
   through one ragged prefill pass. Its last row gives the first draft token.
2. **Extend.** Each further draft token is one decode pass over the sequences that still
   want more, fed with the token drafted in the previous pass.

For sampling requests the draft tokens are *sampled* from the draft model's (warped)
distribution, and that distribution is returned with them: speculative sampling needs it
to accept or reject each token without biasing the output.
"""

from __future__ import annotations

import random
from collections.abc import Sequence as Seq
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from minfer import sampling
from minfer.attention.paged import (
    PagedAttentionBackend,
    ReferencePagedBackend,
    SequenceSpan,
    build_paged_batch,
)
from minfer.cache.block_manager import BlockManager, OutOfBlocks
from minfer.cache.kv_store import PagedKVStore
from minfer.cache.radix_tree import PrefixCache
from minfer.model.loader import load_model
from minfer.model.qwen2 import Qwen2Model
from minfer.scheduler.sequence import Sequence
from minfer.spec.base import Draft

if TYPE_CHECKING:
    from minfer.engine import Engine


class DraftModelDrafter:
    """Drafter backed by a small language model with its own paged KV cache.

    Args:
        model: The draft model.
        num_blocks: Size of the draft model's KV block pool.
        block_size: Token positions per block.
        attention_backend: ``"flashinfer"`` or ``"reference"``.
        cuda_graphs: Capture CUDA graphs for the draft model's decode passes.
        cuda_graph_batch_sizes: Batch sizes to capture.
        max_model_len: Longest sequence the target accepts (sizes the graph buffers).
        seed: Seed of the stream used to sample drafts for unseeded sampling requests.
    """

    def __init__(
        self,
        model: Qwen2Model,
        num_blocks: int,
        block_size: int = 16,
        attention_backend: str = "flashinfer",
        cuda_graphs: bool = True,
        cuda_graph_batch_sizes: Seq[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256),
        max_model_len: int = 32768,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.device = model.device
        config = model.config
        self.blocks = BlockManager(num_blocks, block_size)
        self.store = PagedKVStore(
            config.num_layers,
            num_blocks,
            block_size,
            model.num_kv_heads,
            config.head_dim,
            self.device,
            model.dtype,
        )
        self.cache = PrefixCache(self.blocks)
        self.backend: PagedAttentionBackend
        self.decode_graphs = None
        if attention_backend == "flashinfer":
            from minfer.attention.flashinfer_backend import FlashInferBackend

            self.backend = FlashInferBackend(self.store, model.num_heads)
            if cuda_graphs:
                from minfer.attention.decode_graphs import DecodeGraphs

                self.decode_graphs = DecodeGraphs(
                    model,
                    self.store,
                    self.backend.workspace,
                    batch_sizes=cuda_graph_batch_sizes,
                    max_pages_per_sequence=self.blocks.blocks_for(max_model_len),
                )
        else:
            self.backend = ReferencePagedBackend(self.store)
        # Per sequence: how many confirmed tokens the draft cache held when it last drafted,
        # and the drafted tokens whose K/V it wrote after them.
        self._confirmed: dict[int, int] = {}
        self._tail: dict[int, list[int]] = {}
        self._rng = random.Random(seed)

    @classmethod
    def from_pretrained(
        cls, name_or_path: str | Path, engine: Engine, num_blocks: int | None = None
    ) -> DraftModelDrafter:
        """Load a draft model and configure it like ``engine`` (device, dtype, backend)."""
        target = engine.model
        model = load_model(name_or_path, device=target.device, dtype=target.dtype)
        if model.config.vocab_size != target.config.vocab_size:
            raise ValueError("draft and target models must share a vocabulary")
        config = engine.config
        return cls(
            model,
            num_blocks or engine.blocks.num_blocks,
            block_size=config.block_size,
            attention_backend=config.attention_backend,
            cuda_graphs=config.cuda_graphs,
            cuda_graph_batch_sizes=sorted(
                {min(size, config.max_num_seqs) for size in config.cuda_graph_batch_sizes}
            ),
            max_model_len=engine.max_model_len,
            seed=config.seed,
        )

    # ------------------------------------------------------------------ state

    def release(self, seq_id: int) -> None:
        """See :meth:`minfer.spec.base.Drafter.release`."""
        self._confirmed.pop(seq_id, None)
        self._tail.pop(seq_id, None)
        self.blocks.free(seq_id)

    def _valid_length(self, seq: Sequence) -> int:
        """Leading tokens of ``seq`` whose K/V in the draft cache is still correct."""
        seq_id = seq.seq_id
        if seq_id not in self._confirmed:
            return self.cache.sync_sequence(seq_id, seq.tokens, 0)
        confirmed = self._confirmed[seq_id]
        tokens = seq.tokens
        valid = confirmed
        for drafted in self._tail[seq_id]:
            if valid >= len(tokens) - 1 or tokens[valid] != drafted:
                break
            valid += 1
        return valid

    def _reserve(self, seq_id: int, num_tokens: int) -> bool:
        """Grow a draft block table, evicting cached prefixes if the pool is full."""
        try:
            self.blocks.ensure_capacity(seq_id, num_tokens)
            return True
        except OutOfBlocks:
            short = self.blocks.blocks_missing(seq_id, num_tokens) - self.blocks.num_free
            if not self.cache.evict(short):
                return False
        try:
            self.blocks.ensure_capacity(seq_id, num_tokens)
        except OutOfBlocks:
            return False
        return True

    # ---------------------------------------------------------------- drafting

    def _pick(
        self, logits: Tensor, seqs: list[Sequence], probs: dict[int, list[Tensor]]
    ) -> list[int]:
        """One draft token per row; records the sampling distribution of stochastic rows."""
        tokens = sampling.greedy(logits)
        stochastic = [i for i, seq in enumerate(seqs) if not seq.params.greedy]
        if stochastic:
            params = [seqs[i].params for i in stochastic]
            device = logits.device
            index = torch.tensor(stochastic, device=device)
            warped = sampling.warped_probs(
                logits.index_select(0, index),
                torch.tensor([p.temperature for p in params], device=device, dtype=torch.float32),
                torch.tensor([p.top_k for p in params], device=device),
                torch.tensor([p.top_p for p in params], device=device, dtype=torch.float32),
            )
            uniforms = [(seqs[i].rng or self._rng).random() for i in stochastic]
            drawn = sampling.sample_from_probs(
                warped, torch.tensor(uniforms, device=device, dtype=torch.float64)
            )
            tokens[index] = drawn
            for row, i in enumerate(stochastic):
                probs.setdefault(seqs[i].seq_id, []).append(warped[row])
        return tokens.tolist()

    def _decode(self, spans: list[SequenceSpan]) -> Tensor:
        """Logits of a one-token-per-sequence pass (CUDA graph when one fits)."""
        graphs = self.decode_graphs
        if graphs is not None and len(spans) <= graphs.max_batch_size:
            return graphs.run(spans)
        batch = build_paged_batch(spans, self.blocks.block_size, self.device)
        hidden = self.model(batch.token_ids, batch.positions, self.backend.decode(batch))
        return self.model.logits(hidden)

    @torch.inference_mode()
    def propose(self, seqs: Seq[Sequence], limits: Seq[int]) -> list[Draft]:
        """See :meth:`minfer.spec.base.Drafter.propose`."""
        size = self.blocks.block_size
        drafts = {seq.seq_id: Draft([]) for seq in seqs}
        active: list[Sequence] = []
        want: dict[int, int] = {}
        spans = []
        for seq, limit in zip(seqs, limits, strict=True):
            if limit < 1:
                continue
            seq_id, length = seq.seq_id, len(seq.tokens)
            valid = self._valid_length(seq)
            if not self._reserve(seq_id, length + limit - 1):
                self.release(seq_id)
                continue
            spans.append(SequenceSpan(seq.tokens[valid:], valid, self.blocks.table(seq_id)))
            active.append(seq)
            want[seq_id] = limit
        if not active:
            return [drafts[seq.seq_id] for seq in seqs]

        probs: dict[int, list[Tensor]] = {}
        # Sequences one token behind (the usual case after the first round) catch up
        # through the CUDA-graph decode pass; longer gaps need the ragged prefill pass.
        single = [i for i, span in enumerate(spans) if len(span.token_ids) == 1]
        ragged = [i for i, span in enumerate(spans) if len(span.token_ids) > 1]
        picked: list[int] = [0] * len(active)
        if single:
            logits = self._decode([spans[i] for i in single])
            tokens = self._pick(logits, [active[i] for i in single], probs)
            for i, token in zip(single, tokens, strict=True):
                picked[i] = token
        if ragged:
            batch = build_paged_batch([spans[i] for i in ragged], size, self.device)
            hidden = self.model(batch.token_ids, batch.positions, self.backend.prefill(batch))
            last_rows = batch.qo_indptr[1:].to(self.device, torch.int64) - 1
            logits = self.model.logits(hidden.index_select(0, last_rows))
            tokens = self._pick(logits, [active[i] for i in ragged], probs)
            for i, token in zip(ragged, tokens, strict=True):
                picked[i] = token
        for seq, token in zip(active, picked, strict=True):
            drafts[seq.seq_id].tokens.append(token)
            length = len(seq.tokens)
            self._confirmed[seq.seq_id] = length
            self._tail[seq.seq_id] = []
            # Share the confirmed tokens' full blocks with later requests.
            self.cache.sync_sequence(seq.seq_id, seq.tokens, length - 1)

        step = 1
        while True:
            active = [seq for seq in active if want[seq.seq_id] > step]
            if not active:
                break
            spans = []
            for seq in active:
                seq_id = seq.seq_id
                fed = drafts[seq_id].tokens[-1]
                position = len(seq.tokens) + step - 1
                spans.append(SequenceSpan((fed,), position, self.blocks.table(seq_id)))
                self._tail[seq_id].append(fed)
            logits = self._decode(spans)
            for seq, token in zip(active, self._pick(logits, active, probs), strict=True):
                drafts[seq.seq_id].tokens.append(token)
            step += 1

        for seq_id, rows in probs.items():
            drafts[seq_id].probs = torch.stack(rows)
        return [drafts[seq.seq_id] for seq in seqs]
