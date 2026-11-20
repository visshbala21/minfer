# minfer — Build Plan

You are building `minfer`: a from-scratch LLM inference engine optimized for multi-turn AI agent workloads. This is a flagship portfolio project that senior inference engineers will scrutinize. Correctness, honest benchmarks, and clean engineering matter more than speed of delivery. Work autonomously through the phases below. Do not ask for confirmation between phases — keep going until everything is done or you are truly blocked.

## Session protocol
- At the start of every session: read this file and DESIGN.md, run `git log --oneline | head -30`, and continue from the first incomplete phase.
- Keep `PROGRESS.md` updated: one line per phase with status (todo / in progress / done / cut) and its acceptance-test result.
- Push to origin after every phase and after any meaningful sub-step.

## Ground rules (non-negotiable)
- Do NOT advance to the next phase until the current phase's acceptance test passes.
- Never fabricate, estimate, or hand-edit benchmark numbers. Every number in the README must come from a committed results file produced by a committed script.
- If a feature underperforms the baseline, keep it, report it honestly, and explain why in the README.
- If stuck on the same failure for more than ~45 minutes: stop changing code, write a minimal reproduction, state the root cause hypothesis in DESIGN.md, then fix.
- Time caps: Phase 7 (speculative decoding) and Phase 8 (Triton kernel) get ~6 hours each. If not passing by then, mark as experimental with honest numbers, or cut, and move on.
- Commit with clear, descriptive messages. No giant commits.
- Keep `DESIGN.md` updated: every design decision, tradeoff, bug found, root cause, fix, and before/after measurement.
- Pin all dependency versions in `pyproject.toml`. Record GPU, driver, CUDA, PyTorch, vLLM, SGLang, FlashInfer versions in `bench/ENV.md`.
- Code quality: type hints, docstrings on core classes, no dead code, no notebooks in the main path.
- You are the only process allowed to use the GPU. Never run two benchmarks concurrently.

## Environment
- Phase 0: run `nvidia-smi`, check CUDA/PyTorch versions, install deps with `uv` (torch, transformers, safetensors, flashinfer, triton, fastapi, uvicorn, pytest, matplotlib, ruff). Put vLLM and SGLang in their own separate virtualenvs (`.venv-vllm`, `.venv-sglang`) so their deps never conflict with minfer's.
- All model downloads go to `$HF_HOME` (persistent storage).
- Primary model: Qwen/Qwen2.5-1.5B-Instruct (single GPU). Draft model: Qwen/Qwen2.5-0.5B-Instruct. Tensor parallelism: Qwen/Qwen2.5-14B-Instruct.

## Repo structure
```
minfer/
  model/        # pure-PyTorch Qwen2 implementation, weight loading
  cache/        # block manager, radix tree prefix cache
  scheduler/    # continuous batching, admission, preemption, chunked prefill
  attention/    # FlashInfer wrappers + custom Triton kernel
  spec/         # speculative decoding
  parallel/     # tensor parallelism
  server/       # OpenAI-compatible API
  engine.py
bench/          # workload generators, runners, plotting, results/
deploy/         # Modal deployment
tests/
PLAN.md, PROGRESS.md, DESIGN.md, README.md, Makefile, pyproject.toml, LICENSE (MIT), .github/workflows/ci.yml
```

## Phase 1 — Model
Implement the Qwen2 forward pass in pure PyTorch: embeddings, RoPE, GQA attention, RMSNorm, SwiGLU MLP, lm_head. Load weights directly from safetensors.
ACCEPTANCE: pytest compares logits vs HF transformers on 5 diverse prompts (atol 1e-3 in fp32; document bf16 tolerance).

## Phase 2 — Paged KV cache + continuous batching
- Block manager: fixed-size KV blocks (configurable, default 16 tokens), free list, ref counts, per-sequence block tables, KV memory budget computed from free GPU memory.
- Scheduler: continuous batching with prefill/decode handling, admission control, preemption via recompute when out of blocks, max-batched-tokens limit.
- Attention via FlashInfer paged prefill/decode wrappers.
- Sampling: greedy, temperature, top-p, top-k, stop tokens, max_tokens.
ACCEPTANCE: 64 concurrent requests complete; greedy outputs match vLLM token-for-token on a 20-prompt test set (document any divergence and its numerical cause).

## Phase 3 — CUDA graphs + chunked prefill
- Capture CUDA graphs for decode at bucketed batch sizes (1,2,4,8,16,32,64,128,256).
- Chunked prefill: split long prompts into chunks interleaved with decode steps to protect decode latency.
- CLI/config flags to toggle each.
ACCEPTANCE: measured decode-latency improvement with CUDA graphs on vs off; p99 TPOT improvement with chunked prefill on a mixed long/short workload. Results committed.

## Phase 4 — Radix-tree prefix cache + cache-aware scheduling (HEADLINE FEATURE)
- Radix tree keyed on token sequences; nodes map to ref-counted KV blocks shared across requests.
- LRU eviction of unreferenced leaves under memory pressure.
- Cache-aware scheduling: prioritize waiting requests by cached-prefix length, with a starvation guard.
- Metrics: prefix cache hit rate, prefill tokens saved.
ACCEPTANCE: unit tests for insert / match / split / evict / ref-counting; outputs identical with cache on vs off; hit rate logged.

## Phase 5 — OpenAI-compatible server
FastAPI: `/v1/completions`, `/v1/chat/completions` (chat template applied), SSE streaming, `/v1/models`, `/health`, `/metrics` (throughput, queue depth, cache hit rate, KV usage).
ACCEPTANCE: `vllm bench serve` runs against minfer without errors; the official `openai` Python client works with streaming.

## Phase 6 — Agent workload benchmark
`bench/agent_workload.py`: deterministic (seeded) generator of realistic multi-turn agent sessions:
- A few agent types, each with a shared ~4k-token system prompt + JSON tool schemas.
- 5–15 turns per session; turns append tool calls (JSON), tool results (logs, stack traces, code, search results), and model responses; history grows each turn.
- Poisson session arrivals, think time between turns, configurable rate.
- Export JSONL traces so others can reuse them. Include a replay client that streams against any OpenAI-compatible endpoint and records TTFT, TPOT, and end-to-end latency per request.
Runner benchmarks minfer, vLLM, and SGLang on identical hardware, model, and traces at request rates [1, 2, 4, 8, 16, max], plus ShareGPT as a control workload (via `vllm bench serve`).
Metrics: output throughput (tok/s), TTFT p50/p99, TPOT p50/p99, E2E latency, prefix hit rate.
`bench/plot.py` generates all charts from results JSON.
ACCEPTANCE: results JSON committed for all three engines on both workloads; plots generated by script.

## Phase 7 — Speculative decoding (cap: ~6h)
- Prompt-lookup / n-gram drafting (agent outputs repeat tool names, JSON keys, copied code).
- Draft-model speculation (Qwen2.5-0.5B → target) with correct rejection sampling, batched verification inside continuous batching, KV rollback on rejection, per-request draft length.
- Log acceptance rate and speedup.
ACCEPTANCE: greedy outputs identical with speculation on vs off; acceptance rate and latency/throughput deltas committed on both workloads.

## Phase 8 — Custom Triton paged-attention decode kernel (cap: ~6h)
GQA-aware Triton decode kernel reading block tables; drop-in alternative to FlashInfer decode.
ACCEPTANCE: numerically matches FlashInfer (tolerance documented); microbenchmark vs FlashInfer across batch sizes and context lengths, committed with plots.

## Phase 9 — FP8
FP8 (E4M3) weights and FP8 KV cache with proper scaling on H100.
ACCEPTANCE: GSM8K (200-sample subset) accuracy bf16 vs FP8 reported; KV capacity and throughput gains measured.

## Phase 10 — Tensor parallelism
Megatron-style column/row-parallel linears, NCCL all-reduce, sharded KV cache, serving Qwen2.5-14B on 2×H100 SXM (NVLink).
If only 1 GPU is available: implement and unit-test with 2 processes on CPU (gloo backend), and leave the GPU benchmark as a clearly marked TODO in PROGRESS.md.
ACCEPTANCE: outputs match single-device reference; scaling numbers committed.

## Phase 11 — Profiling + optimization
Profile prefill and decode with torch.profiler and/or Nsight Systems. Identify the top bottleneck, fix it, and record before/after numbers in DESIGN.md with trace exports committed.

## Phase 12 — Ablations, README, CI, Modal
- Ablation table: full system vs each feature disabled (prefix cache, cache-aware scheduling, CUDA graphs, chunked prefill, speculation, FP8) on the agent workload.
- Rerun the full benchmark matrix on the final code so all README numbers are current.
- README: one-paragraph pitch (inference engine optimized for agent workloads), architecture diagram (mermaid), headline result, results tables + plots, ablations, "Where minfer loses to vLLM/SGLang and why", reproduction instructions, hardware/env details.
- Makefile: `make test`, `make bench`, `make bench-agent`, `make plots`.
- GitHub Actions: ruff + CPU-runnable unit tests (block manager, radix tree, scheduler, sampling, workload generator).
- `bench/modal_bench.py`: Modal app that provisions an H100, installs pinned deps, and runs the full benchmark suite. README: "Reproduce on Modal: `modal run bench/modal_bench.py`".
- `deploy/modal_app.py`: deploy minfer as a public OpenAI-compatible endpoint on Modal (scale-to-zero), linked from the README.
ACCEPTANCE: a fresh clone on a fresh H100 reproduces the headline numbers with `make bench`.
