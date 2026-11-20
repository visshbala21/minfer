"""Tensor-parallel execution of the model across processes (one rank per device).

Rank 0 is the *driver*: it runs the engine — scheduler, block manager, prefix cache,
sampling — exactly as on a single device, with a model that holds rank 0's slice of the
weights. Ranks 1..N-1 are *workers*: each holds its own slice and its own shard of the KV
cache, and executes the same forward passes in lockstep.

Per pass, the driver broadcasts the batch layout (tokens, positions, KV slots, page tables)
to the workers over pipes; every rank then runs the transformer stack on its shard, and the
two row-parallel projections of each layer meet in an all-reduce (NCCL between GPUs, gloo
between CPU processes). Block ids are allocated by the driver alone, so the same block id
addresses the same token positions in every rank's KV shard — the KV cache is sharded by
attention head with no coordination beyond the batch layout itself.
"""

from __future__ import annotations

import multiprocessing
import socket
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from minfer.attention.base import AttentionContext
from minfer.attention.paged import PagedAttentionBackend, PagedBatch, ReferencePagedBackend
from minfer.cache.kv_store import PagedKVStore
from minfer.engine import Engine, EngineConfig
from minfer.model.qwen2 import Qwen2Model, Shard

ShardFactory = Callable[[Shard, torch.device], Qwen2Model]


def all_reduce(tensor: Tensor) -> Tensor:
    """Sum ``tensor`` across all ranks of the default process group, in place."""
    dist.all_reduce(tensor)
    return tensor


def _join(rank: int, world_size: int, address: str, backend: str) -> None:
    dist.init_process_group(
        backend, init_method=f"tcp://{address}", rank=rank, world_size=world_size
    )


def _make_backend(store: PagedKVStore, model: Qwen2Model, kind: str) -> PagedAttentionBackend:
    if kind == "flashinfer":
        from minfer.attention.flashinfer_backend import FlashInferBackend

        return FlashInferBackend(store, model.num_heads)
    return ReferencePagedBackend(store)


def worker_main(
    rank: int,
    world_size: int,
    address: str,
    dist_backend: str,
    device: str,
    factory: ShardFactory,
    attention_backend: str,
    pipe: Connection,
) -> None:
    """Entry point of a worker rank: mirror the driver's forward passes on this shard."""
    _join(rank, world_size, address, dist_backend)
    target = torch.device(device)
    model = factory(Shard(rank, world_size, all_reduce), target)
    pipe.send("ready")
    backend: PagedAttentionBackend | None = None
    with torch.inference_mode():
        while True:
            message = pipe.recv()
            kind = message[0]
            if kind == "stop":
                break
            if kind == "store":
                _, num_blocks, block_size = message
                config = model.config
                store = PagedKVStore(
                    config.num_layers,
                    num_blocks,
                    block_size,
                    model.num_kv_heads,
                    config.head_dim,
                    target,
                    model.dtype,
                )
                backend = _make_backend(store, model, attention_backend)
                pipe.send("ready")
                continue
            fields = message[1]
            batch = PagedBatch(
                token_ids=fields["token_ids"].to(target),
                positions=fields["positions"].to(target),
                slots=fields["slots"].to(target),
                qo_indptr=fields["qo_indptr"],
                kv_indptr=fields["kv_indptr"],
                kv_indices=fields["kv_indices"],
                kv_last_page_len=fields["kv_last_page_len"],
                block_size=fields["block_size"],
            )
            context = backend.prefill(batch) if kind == "prefill" else backend.decode(batch)
            model(batch.token_ids, batch.positions, context)
    dist.destroy_process_group()


class _BroadcastingBackend:
    """Driver-side attention backend: announce each pass to the workers, then run it."""

    def __init__(self, inner: PagedAttentionBackend, group: TensorParallelGroup) -> None:
        self.inner = inner
        self.group = group

    def prefill(self, batch: PagedBatch) -> AttentionContext:
        """See :meth:`PagedAttentionBackend.prefill`."""
        self.group.broadcast("prefill", batch)
        return self.inner.prefill(batch)

    def decode(self, batch: PagedBatch) -> AttentionContext:
        """See :meth:`PagedAttentionBackend.decode`."""
        self.group.broadcast("decode", batch)
        return self.inner.decode(batch)


class TensorParallelGroup:
    """A driver process plus ``world_size - 1`` worker processes sharing one model.

    Args:
        factory: Picklable callable building one rank's model from ``(shard, device)``.
        devices: One device string per rank, e.g. ``["cuda:0", "cuda:1"]``.
        dist_backend: ``"nccl"`` (GPUs) or ``"gloo"`` (CPU).
        attention_backend: ``"flashinfer"`` or ``"reference"``.
    """

    def __init__(
        self,
        factory: ShardFactory,
        devices: list[str],
        dist_backend: str = "nccl",
        attention_backend: str = "flashinfer",
    ) -> None:
        self.world_size = len(devices)
        self.attention_backend = attention_backend
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            address = f"127.0.0.1:{probe.getsockname()[1]}"
        context = multiprocessing.get_context("spawn")
        self._pipes: list[Connection] = []
        self._workers = []
        for rank in range(1, self.world_size):
            ours, theirs = context.Pipe()
            worker = context.Process(
                target=worker_main,
                args=(
                    rank,
                    self.world_size,
                    address,
                    dist_backend,
                    devices[rank],
                    factory,
                    attention_backend,
                    theirs,
                ),
                name=f"minfer-tp-rank-{rank}",
                daemon=True,
            )
            worker.start()
            theirs.close()
            self._pipes.append(ours)
            self._workers.append(worker)
        _join(0, self.world_size, address, dist_backend)
        self.model = factory(Shard(0, self.world_size, all_reduce), torch.device(devices[0]))
        self._wait_ready()

    def _wait_ready(self) -> None:
        for pipe in self._pipes:
            if pipe.recv() != "ready":
                raise RuntimeError("tensor-parallel worker failed to start")

    def engine(self, config: EngineConfig, eos_token_ids: Any = None) -> Engine:
        """An :class:`Engine` on the driver whose passes run on every rank.

        CUDA-graph capture and draft-model speculation are not available under tensor
        parallelism (an all-reduce cannot be captured in a graph).
        """
        if config.cuda_graphs or config.speculation == "draft":
            raise ValueError("tensor parallelism needs cuda_graphs=False and no draft model")
        engine = Engine(self.model, config, eos_token_ids)
        for pipe in self._pipes:
            pipe.send(("store", engine.blocks.num_blocks, config.block_size))
        self._wait_ready()
        engine.backend = _BroadcastingBackend(engine.backend, self)
        return engine

    def broadcast(self, kind: str, batch: PagedBatch) -> None:
        """Send one pass's batch layout to every worker."""
        fields = {
            "token_ids": batch.token_ids.cpu(),
            "positions": batch.positions.cpu(),
            "slots": batch.slots.cpu(),
            "qo_indptr": batch.qo_indptr,
            "kv_indptr": batch.kv_indptr,
            "kv_indices": batch.kv_indices,
            "kv_last_page_len": batch.kv_last_page_len,
            "block_size": batch.block_size,
        }
        for pipe in self._pipes:
            pipe.send((kind, fields))

    def close(self) -> None:
        """Stop the workers and leave the process group."""
        for pipe in self._pipes:
            pipe.send(("stop",))
        for worker in self._workers:
            worker.join(timeout=30)
        dist.destroy_process_group()
