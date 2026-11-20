"""Prefill-pass latency: eager vs fused vs CUDA graph (Phase 11).

    .venv/bin/python -m bench.prefill_latency

Times one prefill pass of a single sequence (cold: nothing cached before it) at several
prompt lengths, end to end — batch layout, attention plan, forward pass, unembedding of the
last row and the host sync that reads the token — for three engine configurations:
the Phase 3 engine (eager pass, one kernel per operator), eager with fused kernels, and
fused kernels replayed from a prefill CUDA graph. Writes
``bench/results/prefill_latency.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from minfer.attention import SequenceSpan, build_paged_batch
from minfer.engine import Engine, EngineConfig

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LENGTHS = (16, 64, 256, 512, 1024, 2048)
RUNS = {
    "eager_unfused": {"fused_kernels": False, "prefill_graphs": False},
    "eager_fused": {"fused_kernels": True, "prefill_graphs": False},
    "graph_fused": {"fused_kernels": True, "prefill_graphs": True},
}


@torch.inference_mode()
def one_pass(engine: Engine, tokens: list[int]) -> float:
    """Seconds for one cold prefill pass of ``tokens``, including the result sync."""
    blocks = engine.blocks
    blocks.ensure_capacity("prefill", len(tokens))
    span = SequenceSpan(tokens, 0, blocks.table("prefill"))
    torch.cuda.synchronize()
    start = time.perf_counter()
    graphs = engine.prefill_graphs
    key = graphs.bucket(1, len(tokens)) if graphs is not None else None
    if key is not None:
        logits = graphs.run(key, [span], [len(tokens) - 1])
    else:
        batch = build_paged_batch([span], engine.config.block_size, engine.device)
        hidden = engine.model(batch.token_ids, batch.positions, engine.backend.prefill(batch))
        logits = engine.model.logits(hidden[-1:])
    logits.argmax(-1).tolist()
    elapsed = time.perf_counter() - start
    blocks.free("prefill")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("bench/results/prefill_latency.json"))
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    results: dict = {
        "model": MODEL,
        "gpu": torch.cuda.get_device_name(0),
        "what": "one cold single-sequence prefill pass, p50 ms over repeats",
        "repeats": args.repeats,
        "runs": {},
    }
    for label, overrides in RUNS.items():
        engine = Engine.from_pretrained(MODEL, EngineConfig(num_blocks=8192, **overrides))
        rng = random.Random(0)
        rows = []
        for length in LENGTHS:
            times = []
            for _ in range(args.repeats + 3):
                times.append(
                    one_pass(engine, [rng.randrange(1000, 100_000) for _ in range(length)])
                )
            rows.append({"tokens": length, "pass_ms_p50": float(np.median(times[3:]) * 1e3)})
        results["runs"][label] = rows
        print(label, rows)
        del engine
        torch.cuda.empty_cache()
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
