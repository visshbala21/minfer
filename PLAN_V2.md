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

# REVISION 2 — Research framing (supersedes Phase 15/17 framing above)
Project framing: minfer is a from-scratch engine used as a controlled TESTBED to measure, on REAL coding-agent traces, how agent-serving techniques contribute alone and in combination. Do not claim any mechanism is novel. Cite prior work precisely: Continuum (arXiv 2511.02230), InferCept, CacheWise (arXiv 2606.16824), TraceLab (arXiv 2606.30560), CacheBlend, EPIC, MiniPIC (arXiv 2606.13126), CacheSlide (FAST'26), Irminsul (arXiv 2605.05696), Leyline (arXiv 2606.01065). Before writing the README claim, search for any existing composed evaluation of these techniques and state honestly what is and isn't new.

## Phase 14B — Real traces (do right after Phase 14)
Build bench/real_traces.py: load the TraceLab coding-agent trace (github.com/uw-syfi/TraceLab) and one public SWE trajectory dataset (e.g. nvidia/Open-SWE-Traces or nebius/SWE-rebench-openhands-trajectories). Replay them against any OpenAI-compatible engine with real token sequences, real turn structure, and tool-call pause durations from TraceLab's timings. Report trace statistics (context lengths, turns, pause distribution, prefix / shifted-duplicate / novel token fractions via rolling-hash fingerprints).

## Phase 15 — Tool-call-aware retention (Continuum-style), as specified above, evaluated on real traces.

## Phase 15C — Position-independent reuse (CacheBlend/MiniPIC-style, cap ~8h)
Reuse KV for identical content (file contents, repeated tool outputs) that reappears at shifted positions: RoPE re-rotation of cached keys plus selective recomputation of a small fraction of tokens. Measure quality honestly: teacher-forced next-token agreement and logit KL vs full prefill on real trajectories, at several recompute ratios.

## Phase 17 (revised) — Composition study (HEADLINE)
On real traces at 7B under realistic memory pressure, run the full factorial of {prefix cache, TTL retention, host-memory tier, position-independent reuse, prompt-lookup speculation} where combinations are valid. Report: prefill tokens saved, TTFT p50/p99, TPOT, throughput, quality metrics. Produce a waterfall/decomposition chart of savings and an interaction table (which techniques compound, which compete for KV memory). Compare against vLLM and SGLang with their best options, and vllm-continuum if installable. README leads with the composition findings.

## Phase 18 — Close the decode gap + agent task completion time (FINAL)
Venvs under /opt/venvs were lost with the old pod; rebuild with scripts/setup_envs.sh first.
1. Merge wip-16b if sound (re-run its tests), else discard and document why.
2. Close the decode gap at 7B (cap ~8h): profile vs vLLM, fix, before/after per change. Target TPOT within 5% of vLLM at 1-16 req/s.
3. NEW METRIC — agent job completion time (JCT), as used by Continuum: replay whole real-trace sessions (all turns + real tool pauses) and measure per-session end-to-end completion time (p50/p90/p99) for minfer full-stack, vLLM and SGLang (best configs, host tiers on), at low / medium / high concurrency including under memory pressure.
4. Re-run the pressure sweep AND the JCT benchmark THREE times per engine. Report medians and min/max. README tables use medians, never best-of-n.
5. README headline: lead with the strongest median result that is a genuine win (JCT, prefill compute saved, engine-level TTFT, or throughput), stated with its exact workload and conditions. If none, say so.
ACCEPTANCE: decode before/after committed; 3x JCT + pressure results for all engines committed; README from medians; CI green.

## Phase 19 — Overload behavior (FINAL, cap ~6h)
Context: with the host tier, minfer beats vLLM on TTFT p50 at 32-96 sessions but loses past capacity (128 sessions: TTFT p50 ~3.3x vLLM, throughput 0.91x) despite computing ~31% fewer prefill tokens. Hypothesis: session-aware retention pins KV for paused sessions and starves arriving requests (the failure mode Continuum describes).
1. Diagnose with measurements first (queueing time vs prefill time vs decode time per request at 128 sessions); record in DESIGN.md.
2. Fix (e.g. load-adaptive retention/TTL that releases or demotes pinned KV when the waiting queue grows, admission ordering that favors returning sessions without starving new ones). Measure each change.
3. Re-run ONLY the host-tier pressure sweep at 96 and 128 sessions, 3 repetitions, minfer only (baselines unchanged), and confirm no regression at 64 and 96.
4. Update README tables (medians) and headline honestly.
ACCEPTANCE: diagnosis + before/after committed; 3x results at 96/128; README updated; CI green.

## Phase 20 — Verify the findings before publication (cap ~8h)
The project's findings will be published in a technical report and sent to the authors of the work evaluated. Act as a hostile reviewer of each claim below and try to break it. Rebuild venvs first. For each: re-run the relevant measurement (3 repetitions), check the setup is fair to the technique, test at least one alternative setting that could reverse the conclusion, and record the verdict (holds / holds with caveats / does not hold) in DESIGN.md and a new FINDINGS.md.
1. Position-independent reuse adds nothing measurable on real coding traces. Check: did the trace actually contain shifted duplicates (report the fraction)? Were recompute ratios and chunk sizes reasonable vs the CacheBlend/MiniPIC papers? Test on a second real trace dataset. Under what conditions WOULD it help?
2. Faster TTFT barely moves agent job completion time. Decompose JCT per session into tool time, queueing, prefill, decode; show the share of each; test with tool pauses scaled to 0 to show what changes.
3. Speculation helps at low load, hurts under load. Confirm the crossover load point.
4. Overload is driven by KV capacity, not retention. Confirm with a capacity-matched comparison (e.g. limit vLLM's KV to minfer's token count, or minfer's to vLLM's).
5. Part of SGLang's TTFT gap is front-end tokenization. Re-confirm with token-id prompts at 7B.
Also: package bench/ as a standalone harness usable against any OpenAI-compatible server (clear README section, one command per benchmark, documented trace sources and licenses).
ACCEPTANCE: FINDINGS.md with a verdict and evidence per claim; all results committed; harness documented; CI green.

## Phase 20 — Verify the findings before publication (cap ~8h)
The project's findings will be published in a technical report and sent to the authors of the work evaluated. Act as a hostile reviewer of each claim below and try to break it. Rebuild venvs first. For each: re-run the relevant measurement (3 repetitions), check the setup is fair to the technique, test at least one alternative setting that could reverse the conclusion, and record the verdict (holds / holds with caveats / does not hold) in DESIGN.md and a new FINDINGS.md.
1. Position-independent reuse adds nothing measurable on real coding traces. Check: did the trace actually contain shifted duplicates (report the fraction)? Were recompute ratios and chunk sizes reasonable vs the CacheBlend/MiniPIC papers? Test on a second real trace dataset. Under what conditions WOULD it help?
2. Faster TTFT barely moves agent job completion time. Decompose JCT per session into tool time, queueing, prefill, decode; show the share of each; test with tool pauses scaled to 0 to show what changes.
3. Speculation helps at low load, hurts under load. Confirm the crossover load point.
4. Overload is driven by KV capacity, not retention. Confirm with a capacity-matched comparison (e.g. limit vLLM's KV to minfer's token count, or minfer's to vLLM's).
5. Part of SGLang's TTFT gap is front-end tokenization. Re-confirm with token-id prompts at 7B.
Also: package bench/ as a standalone harness usable against any OpenAI-compatible server (clear README section, one command per benchmark, documented trace sources and licenses).
ACCEPTANCE: FINDINGS.md with a verdict and evidence per claim; all results committed; harness documented; CI green.
