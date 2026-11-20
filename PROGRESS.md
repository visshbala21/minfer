# Progress

One line per phase. Status is one of: todo / in progress / done / cut.
The acceptance column records the *measured* result of the phase's acceptance test.

| Phase | Status | Acceptance result |
|---|---|---|
| 0 — Environment | done | 1× H100 80GB; three isolated venvs built from pinned lock files; versions recorded in `bench/ENV.md`; FlashInfer JIT smoke test passes (paged decode vs fp32 reference, max abs err 1.8e-3 in bf16). |
| 1 — Model | done | `pytest tests/test_model.py`: 19 passed. fp32 logits vs transformers on 5 diverse prompts: max abs diff 8.0e-5 (budget 1e-3). bf16 tolerance documented in DESIGN.md M-001 (mean abs ≤ 0.085, top-1 ≥ 0.939 vs the fp32 reference — 0.95×–1.02× transformers' own bf16 error). Open item B-001 (PyTorch bf16 SDPA nondeterminism in the test oracle) documented with a reproduction. |
| 2 — Paged KV cache + continuous batching | todo | |
| 3 — CUDA graphs + chunked prefill | todo | |
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
