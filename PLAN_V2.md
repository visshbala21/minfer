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

# REVISED PRIORITIES (supersedes the phase order above)
Order is now: 13 → 14 → 15 (NEW, below) → 16 → 16B (old Phase 15, renamed) → 17.
Phase 15 is the project's new headline. Old Phase 15 ("Close the decode gap") becomes Phase 16B with a ~6h cap and is optional.

## Phase 15 (NEW) — Tool-call-aware KV cache management (HEADLINE)
Insight: agent sessions pause during tool calls and then return with a longer history. Under KV memory pressure, recency-based eviction evicts paused sessions and forces full re-prefill of long histories on return.
1. Memory-pressure agent workload: extend bench/agent_workload.py with a regime where the total live KV of concurrent sessions exceeds GPU KV capacity (e.g. 7B, 16k–48k token histories, many concurrent sessions, realistic tool-call pauses of 2–30s drawn from a configurable distribution). Sweep concurrency from below to well above capacity.
2. Session-aware retention: the API accepts an optional session id (and tool-call pause hints, inferred from tool-call outputs when absent). The eviction policy prefers evicting KV from sessions unlikely to return soon over sessions mid-tool-call (e.g. expected-return-time / TTL-aware scoring instead of pure LRU).
3. Host-memory tier: during a tool-call pause, offload the session's private KV blocks to pinned CPU memory asynchronously, and prefetch them back to the GPU before the next turn arrives (overlap the copy with other work). Measure PCIe transfer cost vs recompute cost and choose per session.
4. Fair baselines: run vLLM and SGLang with their best available options for this regime (prefix caching on, and their CPU/hierarchical KV offloading features if available in the pinned versions — document exactly which). Report the case with and without their offloading.
ACCEPTANCE: plots of prefix hit rate, TTFT p50/p99, and throughput vs concurrency for all three engines (and minfer ablations: LRU only / + session-aware retention / + host tier); committed results; DESIGN.md entry explaining the policy and every tradeoff. Report honestly if baselines match or beat it.

## Phase 17 update
README headline must lead with Phase 15's result if it is a win; otherwise lead with the strongest honest result.

## Phase 17 headline metric (required)
Primary result: "full-stack minfer" (best valid combination of techniques) vs vLLM and SGLang, each in its best documented configuration for this workload, on the same real coding-agent traces at 7B under realistic memory pressure. Report p99 and p50 TTFT, prefill tokens computed, TPOT, throughput, and any quality change. Lead the README with this number (win or loss), followed by the decomposition chart explaining where the gain comes from.
