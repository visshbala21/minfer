"""Tensor parallelism without a second GPU: what can be measured on this machine.

    .venv/bin/python -m bench.tensor_parallel_check

Runs a small randomly initialized Qwen2 (so that no weights need downloading and the run
is CPU-only) on one process and on two processes joined by gloo, and records:

* that the two-process engine generates exactly the single-process tokens;
* the KV memory each rank holds (the cache is sharded by attention head);
* the communication the design implies per token — two all-reduces of a hidden-state
  vector per layer — for the tiny model here and, from their configs, for the models the
  design targets.

It deliberately records no speed-up: CPU processes on one host say nothing about two H100s
on NVLink. The multi-GPU benchmark is a TODO in PROGRESS.md.
Writes ``bench/results/tensor_parallel_check.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from minfer.attention import DenseCausalAttention
from minfer.cache.block_manager import kv_bytes_per_block
from minfer.engine import Engine, EngineConfig
from minfer.model import ModelConfig, Qwen2Model
from minfer.model.loader import shard_state_dict
from minfer.model.qwen2 import Shard
from minfer.parallel import TensorParallelGroup
from minfer.sampling import SamplingParams

TINY = ModelConfig(
    vocab_size=1024,
    hidden_size=256,
    intermediate_size=512,
    num_layers=4,
    num_heads=8,
    num_kv_heads=4,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    max_position_embeddings=1024,
    tie_word_embeddings=False,
    eos_token_id=0,
    torch_dtype=torch.float32,
)
BLOCK = 16


def full_model() -> Qwen2Model:
    """The unsharded model with deterministic random weights."""
    model = Qwen2Model(TINY)
    generator = torch.Generator().manual_seed(0)
    for name, param in model.named_parameters():
        noise = torch.randn(param.shape, generator=generator)
        param.copy_(
            1.0 + 0.1 * noise if "norm" in name else noise * (1.0 if "embed" in name else 0.08)
        )
    return model.eval()


def shard_model(shard: Shard, device: torch.device) -> Qwen2Model:
    """One rank of :func:`full_model` (top-level so worker processes can build it)."""
    full = full_model()
    model = Qwen2Model(TINY, shard)
    model.load_state_dict(shard_state_dict(full.state_dict(), TINY, shard.rank, shard.world_size))
    return model.to(device).eval()


def communication(config: ModelConfig, itemsize: int) -> dict[str, int]:
    """All-reduce traffic per token implied by two row-parallel projections per layer."""
    return {
        "all_reduces_per_token": 2 * config.num_layers,
        "bytes_all_reduced_per_token": 2 * config.num_layers * config.hidden_size * itemsize,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("bench/results/tensor_parallel_check.json")
    )
    args = parser.parse_args()

    rng = random.Random(0)
    prompts = [[rng.randrange(1, 1024) for _ in range(rng.randint(8, 200))] for _ in range(16)]
    params = SamplingParams(max_tokens=48, ignore_eos=True)
    config = EngineConfig(
        num_blocks=512, block_size=BLOCK, attention_backend="reference", cuda_graphs=False
    )

    single = Engine(full_model(), config)
    start = time.perf_counter()
    expected = single.generate(prompts, params)
    single_seconds = time.perf_counter() - start

    group = TensorParallelGroup(
        shard_model, ["cpu", "cpu"], dist_backend="gloo", attention_backend="reference"
    )
    try:
        engine = group.engine(config)
        start = time.perf_counter()
        outputs = engine.generate(prompts, params)
        parallel_seconds = time.perf_counter() - start
        per_rank_kv = engine.store.nbytes
    finally:
        group.close()

    # Cache-free check of the reference itself, so "identical" is anchored to the model.
    model = full_model()
    with torch.inference_mode():
        tokens = prompts[0] + expected[0][:-1]
        ids = torch.tensor(tokens)
        hidden = model(ids, torch.arange(len(tokens)), DenseCausalAttention(1, len(tokens)))
        dense = model.logits(hidden[len(prompts[0]) - 1 :]).argmax(-1).tolist()

    targets = {}
    for name in ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-14B-Instruct"):
        # Only the architecture is needed: fetch config.json, not the weights.
        cfg = ModelConfig.from_pretrained(Path(hf_hub_download(name, "config.json")).parent)
        per_token = kv_bytes_per_block(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, 1, 2)
        targets[name] = {
            "kv_bytes_per_token_single_device": per_token,
            "kv_bytes_per_token_per_rank_world_size_2": per_token // 2,
            "valid_world_sizes": [n for n in (2, 4, 8) if cfg.num_kv_heads % n == 0],
            **communication(cfg, 2),
        }

    results = {
        "what": "CPU-only correctness and accounting check; NOT a performance measurement",
        "model": "random-weight Qwen2, 4 layers, hidden 256, 8 heads / 4 kv heads, float32",
        "world_size": 2,
        "dist_backend": "gloo",
        "requests": len(prompts),
        "generated_tokens": sum(map(len, outputs)),
        "outputs_identical_to_single_process": outputs == expected,
        "single_process_matches_cache_free_forward": dense == expected[0],
        "kv_store_bytes_single_process": single.store.nbytes,
        "kv_store_bytes_per_rank": per_rank_kv,
        "tiny_model_communication": communication(TINY, 4),
        "cpu_wall_seconds_not_a_benchmark": {
            "single_process": single_seconds,
            "two_processes_gloo": parallel_seconds,
        },
        "target_models": targets,
    }
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
