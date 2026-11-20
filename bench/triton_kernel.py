"""Triton paged-attention decode kernel vs FlashInfer: numerics and speed (Phase 8).

    .venv/bin/python -m bench.triton_kernel

One attention layer in isolation, with the engine's real shapes (Qwen2.5-1.5B: 12 query
heads over 2 KV heads, head_dim 128, 16-token blocks, bf16). For each batch size and
context length: random queries and a randomly filled paged KV cache whose blocks are
deliberately scattered (a shuffled block table), then

* the two kernels' outputs are compared with each other and with a float32 reference
  computed by plain PyTorch on the gathered keys and values;
* **kernel only**: each kernel is replayed 28 times from a CUDA graph, which is how the
  engine runs decode attention and measures what the GPU executes, with no Python launcher
  and no host-side planning. FlashInfer is timed in both decode configurations (with and
  without ``use_tensor_cores``) and the faster one is the baseline;
* **end to end**: one engine step's worth of decode attention in wall-clock time — the
  host-side preparation (FlashInfer's ``plan()``; for the Triton kernel, densifying the page
  table and copying it to the device) followed by the 28 replayed kernels;
* for reference, each kernel called from Python (for sub-0.1 ms kernels this mostly measures
  the launcher).

Writes ``bench/results/triton_kernel.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from minfer.attention.flashinfer_backend import flashinfer
from minfer.attention.paged import dense_block_table
from minfer.attention.triton_decode import paged_decode_attention

NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, BLOCK = 12, 2, 128, 16
BATCH_SIZES = (1, 8, 32, 64, 256)
CONTEXTS = (128, 1024, 4096, 16384)
LAYERS = 28  # kernel calls per decode step of the model these shapes come from


def make_case(batch: int, context: int, generator: torch.Generator) -> dict:
    """Random queries and a scattered paged cache with ``context`` positions per sequence."""
    pages = -(-context // BLOCK)
    num_blocks = batch * pages + 1
    device = "cuda"
    cache = torch.randn(
        (num_blocks, 2, BLOCK, NUM_KV_HEADS, HEAD_DIM), generator=generator, dtype=torch.float32
    ).to(device, torch.bfloat16)
    order = torch.randperm(num_blocks - 1, generator=generator) + 1
    table = order.view(batch, pages).to(torch.int32)
    q = torch.randn((batch, NUM_HEADS, HEAD_DIM), generator=generator).to(device, torch.bfloat16)
    lengths = torch.full((batch,), context, dtype=torch.int32)
    return {
        "q": q,
        "cache": cache,
        "table": table.to(device),
        "lengths": lengths.to(device),
        "indptr": torch.arange(batch + 1, dtype=torch.int32) * pages,
        "indices": table.reshape(-1).contiguous(),
        "last_page_len": torch.full((batch,), (context - 1) % BLOCK + 1, dtype=torch.int32),
    }


def reference(case: dict) -> torch.Tensor:
    """Float32 attention over the gathered keys/values (the ground truth)."""
    table = case["table"].long()
    batch, pages = table.shape
    context = int(case["lengths"][0])
    blocks = case["cache"][table.reshape(-1)].float()
    keys = blocks[:, 0].reshape(batch, pages * BLOCK, NUM_KV_HEADS, HEAD_DIM)[:, :context]
    values = blocks[:, 1].reshape(batch, pages * BLOCK, NUM_KV_HEADS, HEAD_DIM)[:, :context]
    group = NUM_HEADS // NUM_KV_HEADS
    q = case["q"].float().view(batch, NUM_KV_HEADS, group, HEAD_DIM)
    scores = torch.einsum("bhgd,bthd->bhgt", q, keys) * HEAD_DIM**-0.5
    out = torch.einsum("bhgt,bthd->bhgd", scores.softmax(-1), values)
    return out.reshape(batch, NUM_HEADS, HEAD_DIM)


def timed(fn: object, repeats: int) -> float:
    """Milliseconds per call of ``fn``: best of five rounds of ``repeats`` calls."""
    for _ in range(5):
        fn()
    best = float("inf")
    for _ in range(5):  # best of five rounds: robust to a stray hiccup
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - start) / repeats * 1e3)
    return best


def graph_ms_per_call(
    fn: object, calls: int = LAYERS, repeats: int = 50
) -> tuple[torch.cuda.CUDAGraph, float]:
    """Capture ``calls`` calls in one CUDA graph; return it and the GPU time per call.

    This is how the engine runs decode attention (28 layers per captured step): Python
    launch overhead is gone and only what the GPU executes remains.
    """
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    best = float("inf")
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            graph.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - start) / (repeats * calls) * 1e3)
    return graph, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("bench/results/triton_kernel.json"))
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    generator = torch.Generator().manual_seed(0)
    rows = []
    for context in CONTEXTS:
        for batch in BATCH_SIZES:
            if batch * context > 2_500_000:
                continue  # keeps the test cache within a few GB
            case = make_case(batch, context, generator)

            def plan(case: dict = case) -> None:
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

            def run_flashinfer(case: dict = case) -> torch.Tensor:
                return wrapper.run(case["q"], case["cache"])

            def run_triton(case: dict = case) -> torch.Tensor:
                return paged_decode_attention(
                    case["q"], case["cache"], case["table"], case["lengths"]
                )

            def plan_and_run(plan: object = plan, run: object = run_flashinfer) -> None:
                plan()
                run()

            plan()
            exact = reference(case)
            ours, theirs = run_triton().float(), run_flashinfer().float()
            scale = exact.abs().mean().item()
            row = {
                "batch_size": batch,
                "context_tokens": context,
                "output_mean_abs": scale,
                "triton_vs_flashinfer_max_abs": (ours - theirs).abs().max().item(),
                "triton_vs_fp32_max_abs": (ours - exact).abs().max().item(),
                "flashinfer_vs_fp32_max_abs": (theirs - exact).abs().max().item(),
                "triton_vs_fp32_mean_abs": (ours - exact).abs().mean().item(),
                "flashinfer_vs_fp32_mean_abs": (theirs - exact).abs().mean().item(),
                "triton_ms": timed(run_triton, args.repeats),
                "flashinfer_kernel_ms": timed(run_flashinfer, args.repeats),
                "flashinfer_plan_and_kernel_ms": timed(plan_and_run, args.repeats),
            }
            # Kernel only: the kernels as the engine runs them, replayed from a CUDA graph
            # (no Python launcher, no plan()). FlashInfer is measured in both of its decode
            # configurations and the faster one is the baseline: SGLang and vLLM select the
            # tensor-core path for this head layout (group size 6).
            pages = case["table"].shape[1]
            csr = SimpleNamespace(
                kv_indptr=case["indptr"],
                kv_indices=case["indices"],
                kv_last_page_len=case["last_page_len"],
                block_size=BLOCK,
            )
            flashinfer_graphs = {}
            for name, tensor_cores in (("flashinfer", False), ("flashinfer_tc", True)):
                graph_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    workspace,
                    "NHD",
                    use_cuda_graph=True,
                    use_tensor_cores=tensor_cores,
                    paged_kv_indptr_buffer=torch.zeros(batch + 1, dtype=torch.int32, device="cuda"),
                    paged_kv_indices_buffer=torch.zeros(
                        batch * pages, dtype=torch.int32, device="cuda"
                    ),
                    paged_kv_last_page_len_buffer=torch.zeros(
                        batch, dtype=torch.int32, device="cuda"
                    ),
                )

                def graph_plan(case: dict = case, w: object = graph_wrapper) -> None:
                    w.plan(
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

                def graph_run(case: dict = case, w: object = graph_wrapper) -> torch.Tensor:
                    return w.run(case["q"], case["cache"])

                graph_plan()
                error = (graph_run().float() - exact).abs().max().item()
                row[f"{name}_graph_vs_fp32_max_abs"] = error
                graph, kernel_ms = graph_ms_per_call(graph_run)
                row[f"{name}_graph_ms"] = kernel_ms
                row[f"{name}_plan_ms"] = timed(graph_plan, args.repeats)
                flashinfer_graphs[name] = (kernel_ms, graph, graph_plan)

            out = torch.empty_like(case["q"])
            table_buffer = torch.zeros_like(case["table"])
            lengths_buffer = torch.ones_like(case["lengths"])

            def triton_prepare(
                csr: object = csr,
                table_buffer: torch.Tensor = table_buffer,
                lengths_buffer: torch.Tensor = lengths_buffer,
            ) -> None:
                # What the engine does per step instead of plan(): densify the page table
                # on the host and copy it into the graph's static buffers.
                table, lengths = dense_block_table(csr)
                table_buffer.copy_(table)
                lengths_buffer.copy_(lengths)

            def graph_triton(
                case: dict = case,
                out: torch.Tensor = out,
                table_buffer: torch.Tensor = table_buffer,
                lengths_buffer: torch.Tensor = lengths_buffer,
            ) -> None:
                paged_decode_attention(
                    case["q"], case["cache"], table_buffer, lengths_buffer, out=out
                )

            triton_prepare()
            triton_graph, row["triton_graph_ms"] = graph_ms_per_call(graph_triton)
            row["triton_prepare_ms"] = timed(triton_prepare, args.repeats)

            best = min(flashinfer_graphs, key=lambda name, g=flashinfer_graphs: g[name][0])
            best_ms, best_graph, best_plan = flashinfer_graphs[best]
            row["flashinfer_best_config"] = best
            row["flashinfer_best_graph_ms"] = best_ms
            row["speedup_in_graph"] = row["flashinfer_graph_ms"] / row["triton_graph_ms"]
            row["speedup_in_graph_vs_best"] = best_ms / row["triton_graph_ms"]

            # End to end: one engine step's worth of decode attention — the host-side
            # preparation once, then the 28 layers' kernels — in wall-clock time.
            def step_flashinfer(plan: object = best_plan, graph: object = best_graph) -> None:
                plan()
                graph.replay()

            def step_triton(graph: object = triton_graph) -> None:
                triton_prepare()
                graph.replay()

            row["flashinfer_step_ms"] = timed(step_flashinfer, 50)
            row["triton_step_ms"] = timed(step_triton, 50)
            row["speedup_end_to_end"] = row["flashinfer_step_ms"] / row["triton_step_ms"]
            row["triton_vs_flashinfer_kernel"] = row["flashinfer_kernel_ms"] / row["triton_ms"]
            rows.append(row)
            print(
                json.dumps(
                    {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}
                )
            )
            del case
            torch.cuda.empty_cache()
    results = {
        "gpu": torch.cuda.get_device_name(0),
        "heads": [NUM_HEADS, NUM_KV_HEADS],
        "head_dim": HEAD_DIM,
        "block_size": BLOCK,
        "dtype": "bfloat16",
        "what": (
            f"one attention layer; ms per call, best of 5 rounds of {args.repeats} calls; "
            f"*_step_ms is host preparation plus {LAYERS} replayed calls"
        ),
        "rows": rows,
    }
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
