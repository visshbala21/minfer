# Progress

One line per phase. Status is one of: todo / in progress / done / cut.
The acceptance column records the *measured* result of the phase's acceptance test.

| Phase | Status | Acceptance result |
|---|---|---|
| 0 — Environment | done | 1× H100 80GB; three isolated venvs built from pinned lock files; versions recorded in `bench/ENV.md`; FlashInfer JIT smoke test passes (paged decode vs fp32 reference, max abs err 1.8e-3 in bf16). |
| 1 — Model | done | `pytest tests/test_model.py`: 19 passed. fp32 logits vs transformers on 5 diverse prompts: max abs diff 8.0e-5 (budget 1e-3). bf16 tolerance documented in DESIGN.md M-001 (mean abs ≤ 0.085, top-1 ≥ 0.939 vs the fp32 reference — 0.95×–1.02× transformers' own bf16 error). Open item B-001 (PyTorch bf16 SDPA nondeterminism in the test oracle) documented with a reproduction. |
| 2 — Paged KV cache + continuous batching | done | `pytest`: 81 passed (GPU + CPU). 64 concurrent requests complete with ample memory, under memory pressure (preemption + recompute) and with multi-step prefill. vLLM greedy comparison on 20 prompts (`bench/results/vllm_greedy_equivalence.json`, DESIGN.md M-002): 8/20 outputs token-identical one-at-a-time, 11/20 batched — vLLM agrees with *itself* (serial vs batched) on 9/20. Every divergence is documented as a float32 near-tie between float32's top-2 tokens (margin ≤ 0.33 logits); teacher-forced per-position agreement with vLLM's tokens is 1,977/2,001 (98.8%). On the CPU model, engine outputs are identical to a cache-free oracle under all batching/preemption configurations. |
| 3 — CUDA graphs + chunked prefill | done | CUDA graphs (`bench/results/decode_latency.json`, M-004): decode step p50 11.54 → 3.75 ms at batch 1 (3.08×), 11.89 → 5.11 ms at batch 64 (2.33×), 1.12× at batch 256 with 2k contexts; replay is bit-identical to eager at bucket sizes. Chunked prefill (`bench/results/chunked_prefill.json`, M-005), mixed long/short workload, chunk 2048 vs off: per-token latency p99 50.0 → 27.8 ms, worst stall 148 → 32 ms. **Caveat:** p99 of the per-request *mean* TPOT does not improve (10.38 → 11.40 ms) and long-prompt TTFT p50 rises 100 → 147 ms; the acceptance criterion is met at the token level only. Flags: `--[no-]cuda-graphs`, `--prefill-chunk-tokens N`, `--no-chunked-prefill`. |
| 4 — Radix-tree prefix cache + cache-aware scheduling | todo | |
| 5 — OpenAI-compatible server | todo | |
| 6 — Agent workload benchmark | todo | |
| 7 — Speculative decoding (cap ~6h) | todo | |
| 8 — Custom Triton paged-attention decode kernel (cap ~6h) | todo | |
| 9 — FP8 | todo | |
| 10 — Tensor parallelism | todo | |
| 11 — Profiling + optimization | todo | |
| 12 — Ablations, README, CI, Modal | todo | |

## Known constraints of the build machine

- **One GPU.** Phase 10 will follow the plan's single-GPU fallback (2-process CPU/gloo tests; the
  2×H100 benchmark stays a marked TODO).
- **No Modal credentials.** The Modal apps (Phase 12) can be written but not deployed from here.
