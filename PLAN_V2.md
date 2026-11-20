# minfer — Plan V2 (phases 13–17)
All ground rules and the session protocol in PLAN.md still apply. Add these phases to PROGRESS.md and continue.

## Phase 13 — Benchmark fairness audit (do this before anything else)
Act as a skeptical reviewer of your own benchmarks. Compare launch flags for minfer, vLLM and SGLang: prefix caching, chunked prefill, max batch size / max-num-seqs, max batched tokens, CUDA graphs, warmup, GPU memory fraction, tokenizer and chat template, sampling params, streaming. Give the baselines their documented best-practice settings for this workload. Check whether minfer's TTFT lead survives. Also check the Triton-vs-FlashInfer microbenchmark: compare kernel time only (exclude FlashInfer plan() / host overhead) and report both kernel-only and end-to-end numbers.
ACCEPTANCE: DESIGN.md entry listing every setting compared and every change made; re-measured TTFT/TPOT/throughput on the agent workload; README updated if any claim changes. If the TTFT lead disappears, say so plainly.

## Phase 14 — Scale to Qwen2.5-7B-Instruct
Run the agent workload and ShareGPT matrix for all three engines on Qwen/Qwen2.5-7B-Instruct (single H100, bf16). Keep 1.5B results as a secondary table.
ACCEPTANCE: results + plots committed; README headline uses 7B.

## Phase 15 — Close the decode gap (cap ~8h)
Goal: match vLLM's TPOT and saturated throughput on the agent workload at 7B. Implement in order, measuring each: (a) unified mixed prefill+decode batches in a single forward pass (ragged batch, one attention plan per step); (b) fuse the KV-cache write (and RoPE) into an existing kernel; (c) anything the profiler shows next. Before/after per change.
ACCEPTANCE: before/after numbers committed for each change; matrix re-run at 7B.

## Phase 16 — Make speculative decoding pay off on agent traces (cap ~6h)
Capture the verification pass in CUDA graphs, add adaptive draft length (disable per request when acceptance drops), and tune prompt-lookup n-gram size. Target: lower TPOT than no-speculation on the agent workload at low and medium request rates at 7B.
ACCEPTANCE: greedy-equivalence tests still pass; speedup (or honest lack of one) committed with acceptance rates.

## Phase 17 — Final rerun and README
Re-run `make bench` on the final code at 7B (and 1.5B secondary), regenerate ablations and README tables, update the headline and the "where minfer loses" section.
ACCEPTANCE: README numbers all trace to committed results; CI green.
