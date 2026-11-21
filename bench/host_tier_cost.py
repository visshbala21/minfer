"""What bringing a session's KV back costs: a copy from host memory vs recomputing it.

    .venv/bin/python -m bench.host_tier_cost

For histories of 1k to 30k tokens, on the real model:

* **restore** — KV blocks copied from pinned host memory into the GPU store through the
  host tier (side-stream copy, then the scatter into the store), timed from the request to
  the point the blocks are usable. Also the offload in the other direction.
* **recompute** — the engine prefilling the same number of tokens from scratch (one
  request, nothing cached, chunked prefill off), timed to its first token.

The ratio is what the host tier buys for a session whose GPU copy was evicted, and the
restore's bandwidth says how it scales. Writes ``bench/results/host_tier_cost_7b.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from bench import models
from minfer.cache.host_tier import HostTier
from minfer.engine import Engine, EngineConfig
from minfer.sampling import SamplingParams

TOKENS = (1_024, 4_096, 8_192, 16_384, 30_000)
REPEATS = 5


def median(values: list[float]) -> float:
    return sorted(values)[len(values) // 2]


def time_copies(engine: Engine, tier: HostTier, num_blocks: int) -> tuple[float, float]:
    """Median seconds to offload and to restore ``num_blocks`` blocks."""
    blocks = list(range(1, 1 + num_blocks))
    targets = list(range(1 + num_blocks, 1 + 2 * num_blocks))
    offloads, restores = [], []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        slots, transfer = tier.offload(blocks)
        while not transfer.done:
            tier.poll()
        offloads.append(time.perf_counter() - start)
        start = time.perf_counter()
        transfer = tier.restore(slots, targets)
        while not transfer.done:
            tier.poll()
        torch.cuda.synchronize()  # the scatter into the store has landed
        restores.append(time.perf_counter() - start)
        tier.free(slots)
    return median(offloads), median(restores)


def time_prefill(engine: Engine, num_tokens: int, seed: int) -> float:
    """Median seconds from submitting an uncached ``num_tokens`` prompt to its first token."""
    generator = torch.Generator().manual_seed(seed)
    times = []
    for _ in range(REPEATS):
        prompt = torch.randint(1000, 100_000, (num_tokens,), generator=generator).tolist()
        torch.cuda.synchronize()
        start = time.perf_counter()
        engine.add_request(prompt, SamplingParams(max_tokens=1))
        while engine.has_work:
            engine.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        engine.prefix_cache.clear()
    return median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=models.PRIMARY, help="model id, or 7b / 1.5b")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    model = models.resolve(args.model)
    out = args.out or models.results_path("bench/results/host_tier_cost.json", model)

    size = 16
    most = -(-max(TOKENS) // size)
    config = EngineConfig(
        num_blocks=2 * most + 64, prefill_chunk_tokens=None, max_batched_tokens=32_768
    )
    engine = Engine.from_pretrained(model, config)
    tier = HostTier(engine.store, most + 8)
    time_prefill(engine, 2048, seed=1)  # warm-up
    time_copies(engine, tier, 64)
    rows = []
    for tokens in TOKENS:
        blocks = -(-tokens // size)
        megabytes = blocks * tier.bytes_per_block / 2**20
        offload, restore = time_copies(engine, tier, blocks)
        prefill = time_prefill(engine, tokens, seed=tokens)
        row = {
            "tokens": tokens,
            "blocks": blocks,
            "megabytes": megabytes,
            "offload_ms": offload * 1e3,
            "restore_ms": restore * 1e3,
            "recompute_ms": prefill * 1e3,
            "restore_gb_per_s": megabytes / 1024 / restore,
            "offload_gb_per_s": megabytes / 1024 / offload,
            "recompute_tokens_per_s": tokens / prefill,
            "recompute_over_restore": prefill / restore,
        }
        rows.append(row)
        print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}))
    result = {
        "model": model,
        "gpu": torch.cuda.get_device_name(0),
        "kv_megabytes_per_1k_tokens": 1000 / size * tier.bytes_per_block / 2**20,
        "what": f"median of {REPEATS}; restore includes the scatter into the GPU store",
        "rows": rows,
    }
    out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
