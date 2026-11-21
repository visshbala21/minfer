# minfer

**minfer is a from-scratch LLM inference engine, used here as a controlled testbed: it
measures, on real coding-agent traces, what each agent-serving technique contributes alone
and in combination.** Agents resend almost everything they have already sent — the same
system prompt and tool schemas on every request, the whole conversation on every turn —
and between turns they wait for tools while their history sits in GPU memory. The
techniques that exploit this have each been published by others
([related work](#related-work-and-what-is-and-is-not-new-here)); none is new here. What
this repository adds is all of them in one engine where each is a flag — a radix-tree
prefix cache over paged KV, session-aware retention, a host-memory KV tier,
position-independent KV reuse, prompt-lookup speculative decoding — measured on the same
traces, next to vLLM and SGLang in their best configurations. The engine itself is a
pure-PyTorch implementation of Qwen2.5 with its own Triton kernels next to FlashInfer's,
continuous batching with CUDA-graph execution, and an OpenAI-compatible server; it is not
a wrapper around another serving engine.

> Phase-by-phase status with each acceptance result: [PROGRESS.md](PROGRESS.md).
> Every design decision, bug, root cause and measurement: [DESIGN.md](DESIGN.md).

## Headline result

**On real coding-agent traces at 7B, once concurrent sessions exceed GPU memory, one
technique decides the outcome — a host-memory tier for the KV cache — and vLLM with its own
offloading is the fastest system measured. minfer with every exact technique enabled
recomputes the least and is still 16–21% behind vLLM in throughput.**

Each engine in its best measured configuration, 64–128 concurrent sessions (one H100 holds
the KV of about 55–65):

<!-- BEGIN full-stack -->
| concurrent sessions | metric | minfer: prefix cache + session-aware retention + host tier | vLLM + native KV offloading | SGLang + hierarchical cache |
|---|---|---|---|---|
| 64 | TTFT p50 (ms) | 128 | 174 | **126** |
|  | TTFT p99 (ms) | 478 | **392** | 528 |
|  | prefill tokens computed per call | **962** | 1,002 | 1,053 |
|  | TPOT p50 (ms) | 25.7 | **16.5** | 16.9 |
|  | output tok/s | 604 | 717 | **729** |
|  | calls completed in 240 s | 1,101 | **1,272** | 1,264 |
|  | failed requests | 0 | 0 | 0 |
| 96 | TTFT p50 (ms) | 266 | 206 | **153** |
|  | TTFT p99 (ms) | 3,913 | **1,091** | 1,409 |
|  | prefill tokens computed per call | **985** | 1,545 | 1,702 |
|  | TPOT p50 (ms) | 46.8 | **27.2** | 29.7 |
|  | output tok/s | 641 | **809** | 772 |
|  | calls completed in 240 s | 1,194 | **1,494** | 1,446 |
|  | failed requests | 0 | 0 | 0 |
| 128 | TTFT p50 (ms) | 3,501 | **704** | 3,071 |
|  | TTFT p99 (ms) | 14,640 | **7,278** | 12,147 |
|  | prefill tokens computed per call | **1,655** | 2,472 | 2,486 |
|  | TPOT p50 (ms) | 55.0 | **45.4** | 45.8 |
|  | output tok/s | 605 | **765** | 694 |
|  | calls completed in 240 s | 1,183 | **1,484** | 1,356 |
|  | failed requests | 0 | 0 | 0 |

Real coding-agent trace, Qwen2.5-7B-Instruct, one H100, 96 GB of host memory for each engine's KV tier. No approximate technique is in any of these configurations, so output quality is that of exact bf16 inference in all three.
<!-- END full-stack -->

Where minfer's numbers come from — every valid combination of the five techniques at 96
sessions ([`bench/composition.py`](bench/composition.py)):

![What each technique adds](bench/plots/composition_7b.png)

<!-- BEGIN composition -->
| techniques | calls completed | prefill tokens computed per call | prompt tokens not computed | TTFT p50 / p99 (ms) | TPOT p50 (ms) | out tok/s |
|---|---|---|---|---|---|---|
| none | 513 | 13,281 | 0.0% | 23088 / 30311 | 74.8 | 228 |
| prefix cache | 787 | 6,850 | 55.7% | 866 / 34093 | 63.0 | 391 |
| prompt-lookup speculation | 428 | 12,614 | 0.0% | 32370 / 43931 | 84.6 | 166 |
| prefix cache + session-aware retention | 795 | 6,260 | 59.5% | 1258 / 28920 | 71.0 | 388 |
| prefix cache + position-independent reuse | 789 | 6,414 | 58.5% | 811 / 32312 | 63.8 | 387 |
| prefix cache + prompt-lookup speculation | 572 | 5,723 | 59.9% | 2047 / 46053 | 132.0 | 272 |
| prefix cache + session-aware retention + host-memory tier | 1228 | 1,002 | 93.9% | 257 / 2906 | 45.5 | 661 |
| prefix cache + session-aware retention + position-independent reuse | 792 | 5,967 | 61.4% | 978 / 28246 | 69.1 | 397 |
| prefix cache + session-aware retention + prompt-lookup speculation | 595 | 5,120 | 64.4% | 2018 / 32444 | 134.3 | 278 |
| prefix cache + position-independent reuse + prompt-lookup speculation | 575 | 5,262 | 63.3% | 1521 / 41748 | 134.9 | 264 |
| prefix cache + session-aware retention + host-memory tier + prompt-lookup speculation | 699 | 1,027 | 93.0% | 538 / 11194 | 131.3 | 350 |
| prefix cache + session-aware retention + position-independent reuse + prompt-lookup speculation | 589 | 4,877 | 66.2% | 2569 / 34261 | 133.9 | 280 |
| prefix cache + session-aware retention + host-memory tier + position-independent reuse | 1169 | 942 | 94.3% | 289 / 4563 | 48.1 | 630 |
| prefix cache + session-aware retention + host-memory tier + position-independent reuse + prompt-lookup speculation | 694 | 997 | 93.2% | 635 / 11104 | 133.7 | 352 |

96 concurrent sessions of the real trace, 240 s measured after 120 s of warm-up, whole GPU.
<!-- END composition -->

What the composition shows:

- **The prefix cache is necessary and far from sufficient under pressure.** Without it a
  call prefills 13k tokens and median time-to-first-token is 23 s. With it, at 96 sessions
  the cache thrashes: 56% of prompt tokens are served from cache where an unlimited cache
  would serve 94%.
- **The host-memory tier is the technique that matters**: prefill per call falls from
  6,260 to 1,002 tokens (the 94% ceiling), throughput rises 1.7×, median TTFT falls 5×.
  Copying a session's KV back from host memory is 18–29× cheaper than recomputing it.
- **Session-aware retention is a small gain** (9% fewer prefill tokens than LRU, a lower
  TTFT tail) that does not prevent the collapse by itself; the host tier needs its
  eviction order, so the two are measured together.
- **Position-independent reuse is nearly irrelevant on this trace**: 6% fewer prefill
  tokens with or without the host tier and no gain in latency or throughput beyond
  run-to-run noise, because only 0.8% of the trace's tokens are shifted
  duplicates — and it costs output fidelity (about 3% of next-token predictions change).
- **Prompt-lookup speculation hurts at this load**: it doubles or triples per-token
  latency and costs 27–47% of throughput in every combination. It lowers per-token latency by 8–14% on a
  lightly loaded server ([below](#component-measurements)); a saturated one has no spare
  rows to verify drafts with.
- **Interactions.** Nothing compounds. The prefill savings of retention, the host tier and
  reuse overlap (each recovers part of the same evicted history, and once the host tier
  has brought the hit rate to its ceiling there is little left for the others), and
  speculation competes with all of them for compute: with the host tier it takes back
  most of the throughput the tier gained (661 → 350 tok/s). The best valid combination for
  this workload is therefore the three exact KV techniques and nothing else.

<!-- BEGIN interactions -->
| pair (on top of what both need) | metric | without either | first alone | second alone | both | both − sum of each |
|---|---|---|---|---|---|---|
| session-aware retention × position-independent reuse | prefill tokens / call | 6,850 | -590 | -435 | -883 | +143 |
| session-aware retention × position-independent reuse | TTFT p50 (ms) | 866 | +392 | -56 | +112 | -224 |
| session-aware retention × position-independent reuse | TPOT p50 (ms) | 62.96 | +8.02 | +0.85 | +6.10 | -2.77 |
| session-aware retention × prompt-lookup speculation | prefill tokens / call | 6,850 | -590 | -1,126 | -1,730 | -13 |
| session-aware retention × prompt-lookup speculation | TTFT p50 (ms) | 866 | +392 | +1181 | +1152 | -421 |
| session-aware retention × prompt-lookup speculation | TPOT p50 (ms) | 62.96 | +8.02 | +69.07 | +71.33 | -5.75 |
| position-independent reuse × prompt-lookup speculation | prefill tokens / call | 6,850 | -435 | -1,126 | -1,587 | -26 |
| position-independent reuse × prompt-lookup speculation | TTFT p50 (ms) | 866 | -56 | +1181 | +655 | -471 |
| position-independent reuse × prompt-lookup speculation | TPOT p50 (ms) | 62.96 | +0.85 | +69.07 | +71.94 | +2.02 |
| host-memory tier × position-independent reuse | prefill tokens / call | 6,260 | -5,258 | -292 | -5,318 | +233 |
| host-memory tier × position-independent reuse | TTFT p50 (ms) | 1258 | -1001 | -279 | -968 | +312 |
| host-memory tier × position-independent reuse | TPOT p50 (ms) | 70.98 | -25.51 | -1.92 | -22.86 | +4.56 |
| host-memory tier × prompt-lookup speculation | prefill tokens / call | 6,260 | -5,258 | -1,140 | -5,233 | +1,165 |
| host-memory tier × prompt-lookup speculation | TTFT p50 (ms) | 1258 | -1001 | +760 | -719 | -479 |
| host-memory tier × prompt-lookup speculation | TPOT p50 (ms) | 70.98 | -25.51 | +63.31 | +60.34 | +22.54 |
<!-- END interactions -->

Single runs. Three configurations were measured twice (in this table and in the
[pressure sweep](#real-coding-agent-traces-under-memory-pressure)): throughput repeated to
1–3% and hit rate to half a point, but latency percentiles in overloaded configurations
moved by up to 25% between runs, and per-call figures in collapsing configurations depend
on which calls complete. Differences of that size in this table are not findings.

**Why minfer is behind vLLM with everything on.** Not KV management — it recomputes
33–42% fewer prefill tokens than either baseline at 96 and 128 sessions — but its decode
path: per-token latency is 1.56× vLLM's at 64 sessions where both are at the hit-rate
ceiling ([details](#where-minfer-loses-to-vllm-and-sglang-and-why)).

## Engine comparison on a synthetic agent workload

The same three engines on a generated agent workload that fits in GPU memory
(`bench/agent_workload.py`), where the differences are in the engines' request paths
rather than in KV management.

<!-- BEGIN headline -->
On the multi-turn agent workload at 1–16 req/s (Qwen2.5-7B-Instruct, bf16, one H100, greedy, identical traces):

| | minfer | vLLM | SGLang |
|---|---|---|---|
| TTFT p50 (ms) | 49–57 | 78–81 | 57–67 |
| TTFT p99 (ms) | 86–118 | 121–136 | 97–125 |
| TPOT p50 (ms) | 6.64–9.78 | 6.26–7.88 | 6.27–8.20 |
| prefix-cache hit rate | 93.0%–93.6% | 92.6%–93.2% | 93.1%–93.7% |
| saturated output tok/s ("max") | 1836 | 2258 | 2264 |
| TTFT p50 relative to vLLM | 0.63–0.70× | 1.00× | 0.73–0.82× |
| TPOT p50 relative to vLLM | 1.06–1.24× | 1.00× | 1.00–1.04× |
| saturated tok/s relative to vLLM | 0.81× | 1.00× | 1.00× |
| saturated TTFT p50 / p99 (ms) | 106 / 316 | 161 / 835 | 146 / 338 |
<!-- END headline -->

The same comparison on the model the engine was developed on (Qwen2.5-1.5B; re-measured
on the final code, baselines with the flags the [fairness audit](#fairness-audit)
selected):

<!-- BEGIN headline-1.5b -->
On the multi-turn agent workload at 1–16 req/s (Qwen2.5-1.5B-Instruct, bf16, one H100, greedy, identical traces):

| | minfer | vLLM | SGLang |
|---|---|---|---|
| TTFT p50 (ms) | 37–40 | 62–65 | 47–55 |
| TTFT p99 (ms) | 66–94 | 97–134 | 72–106 |
| TPOT p50 (ms) | 2.54–2.99 | 2.41–2.50 | 2.25–2.59 |
| prefix-cache hit rate | 93.0%–93.6% | 92.6%–93.2% | 93.1%–93.7% |
| saturated output tok/s ("max") | 4172 | 4890 | 5741 |
| TTFT p50 relative to vLLM | 0.59–0.62× | 1.00× | 0.75–0.85× |
| TPOT p50 relative to vLLM | 1.06–1.20× | 1.00× | 0.94–1.04× |
| saturated tok/s relative to vLLM | 0.85× | 1.00× | 1.17× |
| saturated TTFT p50 / p99 (ms) | 84 / 274 | 145 / 713 | 115 / 367 |
<!-- END headline-1.5b -->

**minfer answers first.** At 7B it has the lowest median time-to-first-token of the three
engines at every request rate on the agent workload and when saturated (24–29 ms ahead of
vLLM, 7–10 ms ahead of SGLang at 1–16 req/s), and the lowest 99th percentile — by a margin
that at 16 req/s (118 vs 121 ms for SGLang) is inside the run-to-run noise.
All three engines reuse the same share of prompt tokens (hit-rate row) — on this workload
their prefix caches are equally effective — so the lead is not a better hit rate; it is a
shorter path from socket to first token for a request that is mostly cache hits. A
[fairness audit](#fairness-audit), run at 1.5B, took that path apart: when clients send
token ids instead of text, minfer's lead over vLLM is unchanged (16 vs 36 ms) but its lead
over SGLang shrinks from about 10 ms to about 3 ms, and disappears at the median when
saturated. Against SGLang, most of the lead is therefore the HTTP front end (batched
tokenization off the event loop), not the engine; against vLLM it is the engine path
(block-sharing admission with no copy, and a CUDA-graph prefill pass for the few hundred
tokens that are actually new).

**It does not generate fastest, and at 7B the gap widens with load.** Per-token latency is
6–8% behind both baselines at 1–4 req/s, and at 16 req/s 24% behind vLLM and 19% behind
SGLang; saturated throughput is 0.81× of either. A turn's end-to-end latency is within 2–6%
of the best engine at 1–4 req/s (the head start on the first token nearly pays for the
slower tokens after it) and 16% behind at 16 req/s. See
[where minfer loses](#where-minfer-loses-to-vllm-and-sglang-and-why).

## Architecture

```mermaid
flowchart LR
    client([OpenAI client]) -->|HTTP / SSE| api

    subgraph frontend [HTTP process]
        api[FastAPI routes<br/>chat template · stop strings]
        tok[batch tokenizer<br/>incremental detokenizer]
        api <--> tok
    end

    api <-->|"pipes: token ids in,<br/>one message per step out"| loop

    subgraph engine [Engine process]
        loop[engine step loop]
        sched[scheduler<br/>admission · chunked prefill<br/>eviction → preemption]
        radix[(radix-tree<br/>prefix cache)]
        blocks[block manager<br/>ref counts · page tables]
        spec[drafters<br/>prompt lookup · draft model]
        loop --> sched
        sched <--> radix
        sched <--> blocks
        radix <--> blocks
        loop --> spec
    end

    subgraph gpu [GPU]
        prefill[prefill pass<br/>ragged spans · CUDA graph ≤ 1k tokens]
        decode[decode pass<br/>CUDA graph per batch bucket]
        model[Qwen2 layers<br/>merged projections + fused kernels]
        attn[paged attention<br/>Triton decode · FlashInfer prefill]
        kv[(paged KV store<br/>bf16 or mixed FP8)]
        prefill --> model
        decode --> model
        model <--> attn
        attn <--> kv
    end

    loop --> prefill
    loop --> decode
    blocks -. block ids .-> kv
```

How a request flows:

1. The HTTP process applies the chat template, tokenizes (batched, off the event loop) and
   sends token ids to the engine process over a pipe.
2. The scheduler looks the prompt up in the radix tree. Blocks for the longest cached
   prefix are *shared* into the request's page table (a reference count, no copy); only the
   remaining tokens are prefilled, in chunks interleaved with other requests' decode steps.
3. Each engine step runs at most one ragged prefill pass and one fixed-shape decode pass,
   both replayed from CUDA graphs when their shape fits a bucket. Blocks are published to
   the radix tree as soon as they are full, so a second request with the same prefix can
   hit while the first is still running.
4. Under memory pressure the cache gives back least-recently-used leaves before any
   running request is preempted; preempted requests are recomputed later (usually from the
   cache).
5. Tokens stream back one message per engine step and are detokenized incrementally into
   SSE chunks.

| Component | Where | Notes |
|---|---|---|
| Qwen2 forward pass, safetensors loading | `minfer/model/` | verified against `transformers` (fp32 max abs logit diff 8e-5); fused path in `fused.py` |
| Block manager, paged KV store | `minfer/cache/` | 16-token blocks, ref counts, O(1) evictable count, optional mixed-precision FP8 |
| Radix-tree prefix cache | `minfer/cache/radix_tree.py` | block-aligned edges, split on divergence, LRU leaf eviction |
| Scheduler | `minfer/scheduler/` | cache-aware admission with starvation guard, chunked prefill, preemption by recompute |
| Attention | `minfer/attention/` | Triton paged decode kernel, FlashInfer prefill, CUDA graphs for decode and prefill |
| Speculative decoding | `minfer/spec/` | prompt-lookup and draft-model drafters; exact for greedy, unbiased for sampling |
| Tensor parallelism | `minfer/parallel/` | Megatron-style shards, head-sharded KV cache (CPU-verified only) |
| FP8 | `minfer/quant/` | E4M3 weights + activations; E4M3 KV cache with calibrated scales |
| Server | `minfer/server/` | OpenAI-compatible; engine in its own process |
| Benchmarks | `bench/` | agent workload generator + replay client, three-engine matrix, ablations, plots |

## Real coding-agent traces under memory pressure

The comparison above uses a synthetic agent workload that fits in GPU memory. This section
uses real trajectories, and asks what happens when the sessions' KV does not fit.

**The trace** ([`bench/real_traces.py`](bench/real_traces.py)). No public dataset has both
the content and the timing of real agent sessions, so two are combined: token sequences and
turn structure from `nvidia/Open-SWE-Traces` (Claude Code harness: full system prompt, tool
definitions, every assistant message and tool result), and tool-call pause durations
sampled per tool name from the 141,824 real tool-call timings of TraceLab. Sessions are cut
at Qwen2.5's 32k-token context.

<!-- BEGIN real-trace -->
| real trace (Qwen2.5-7B tokenizer) | value |
|---|---|
| sessions (cut at the 32k-token context limit) | 300 (163) |
| model calls | 8,112 |
| model calls per session, p50 / p90 / max | 28 / 42 / 57 |
| prompt tokens per call, p50 / p90 / max | 17,806 / 28,482 / 31,938 |
| tokens appended per call, p50 / p90 / max | 549 / 1,916 / 26,443 |
| output tokens per call, p50 / p90 / max | 100 / 451 / 7,179 |
| tool calls | Bash 7,404, Read 1,407, Edit 1,259 |
| pause after a call (s), p50 / p75 / p90 / p99 | 0.26 / 3.5 / 30 / 120 |
| pauses longer than 1 s / 10 s / 30 s | 40% / 16% / 10% |
| prefix / shifted duplicate / novel — all prompt tokens | 94.4% / 0.8% / 4.8% |
| prefix / shifted duplicate / novel — first call of a session | 6.3% / 66.8% / 26.9% |
| prefix / shifted duplicate / novel — later calls | 94.7% / 0.6% / 4.7% |
<!-- END real-trace -->

A token is *prefix* if an exact-prefix cache of unlimited size could serve it, *shifted
duplicate* if it is not but repeats, at another position, 32 tokens seen before (what
position-independent caching could add), *novel* otherwise. Two things follow before any
engine runs: prefix caching already covers 94% of the tokens if nothing is evicted, and
position-independent reuse has 0.8% of them to find, nearly all in the system prompt of a
session's first call (Claude Code's system prompt starts with the working directory, the
date and a git status, so sessions share it only *after* a different first paragraph).

**Memory pressure** ([`bench/pressure.py`](bench/pressure.py)). The trace replayed
closed-loop at 32–128 concurrent sessions; at 7B one H100 holds the KV of about 55–65
sessions. minfer is run with LRU eviction, with session-aware retention (the eviction
order follows the expected remaining tool-call pause of each session — the idea of
Continuum and CacheWise), and with a 96 GB host-memory tier on top (a session's KV is
copied to pinned memory when it pauses and copied back when it returns); vLLM and SGLang
are run with and without the CPU-memory KV offloading each ships (96 GB each).

<!-- BEGIN pressure -->
| concurrent sessions | configuration | calls completed | cache hit rate | prefill tokens computed per call | TTFT p50 / p99 (ms) | TPOT p50 (ms) | out tok/s | failed |
|---|---|---|---|---|---|---|---|---|
| 32 | minfer, session-aware retention | 627 | 94.2% | 959 | 101 / 287 | 10.7 | 376 | 0 |
|  | vLLM | 654 | 93.9% | 1,005 | 131 / 277 | 8.6 | 403 | 0 |
|  | SGLang | 653 | 94.1% | 972 | 109 / 317 | 8.9 | 400 | 0 |
| 64 | minfer, LRU eviction | 1030 | 88.5% | 1,906 | 134 / 1350 | 27.7 | 564 | 0 |
|  | minfer, session-aware retention | 1074 | 91.0% | 1,496 | 128 / 1245 | 25.5 | 593 | 0 |
|  | minfer, session-aware retention + 96 GB host tier | 1101 | 94.2% | 962 | 128 / 478 | 25.7 | 604 | 0 |
|  | vLLM | 1238 | 89.9% | 1,695 | 165 / 875 | 17.6 | 701 | 0 |
|  | vLLM + native KV offloading (96 GB) | 1272 | 94.1% | 1,002 | 174 / 392 | 16.5 | 717 | 0 |
|  | SGLang | 1202 | 88.7% | 1,891 | 134 / 1025 | 18.5 | 670 | 0 |
|  | SGLang + hierarchical cache (96 GB host) | 1264 | 93.8% | 1,053 | 126 / 528 | 16.9 | 729 | 0 |
| 96 | minfer, LRU eviction | 788 | 55.3% | 6,913 | 1142 / 37006 | 62.9 | 388 | 0 |
|  | minfer, session-aware retention | 788 | 59.7% | 6,227 | 1204 / 28856 | 69.0 | 398 | 0 |
|  | minfer, session-aware retention + 96 GB host tier | 1194 | 94.0% | 985 | 266 / 3913 | 46.8 | 641 | 0 |
|  | vLLM | 1302 | 82.2% | 3,012 | 221 / 1703 | 38.3 | 696 | 0 |
|  | vLLM + native KV offloading (96 GB) | 1494 | 91.0% | 1,545 | 206 / 1091 | 27.2 | 809 | 0 |
|  | SGLang | 824 | 50.2% | 8,024 | 410 / 19150 | 42.9 | 406 | 0 |
|  | SGLang + hierarchical cache (96 GB host) | 1446 | 90.1% | 1,702 | 153 / 1409 | 29.7 | 772 | 0 |
| 128 | minfer, LRU eviction | 722 | 34.0% | 9,506 | 8607 / 55017 | 77.1 | 319 | 0 |
|  | minfer, session-aware retention | 736 | 42.5% | 8,287 | 8973 / 45232 | 84.8 | 338 | 0 |
|  | minfer, session-aware retention + 96 GB host tier | 1183 | 89.5% | 1,655 | 3501 / 14640 | 55.0 | 605 | 0 |
|  | vLLM | 791 | 35.2% | 9,877 | 2736 / 26787 | 138.8 | 362 | 0 |
|  | vLLM + native KV offloading (96 GB) | 1484 | 85.0% | 2,472 | 704 / 7278 | 45.4 | 765 | 0 |
|  | SGLang | 743 | 29.4% | 10,572 | 10280 / 30487 | 135.5 | 333 | 0 |
|  | SGLang + hierarchical cache (96 GB host) | 1356 | 84.7% | 2,486 | 3071 / 12147 | 45.8 | 694 | 0 |
<!-- END pressure -->

![KV memory pressure](bench/plots/pressure_7b.png)

What the sweep shows:

- **Below capacity (32 sessions) nothing differs**: all three engines sit at the 94%
  ceiling.
- **Without a host tier every engine falls off a cliff** once live KV exceeds GPU memory:
  at 128 sessions the hit rate is 29–43%, engines prefill 8–11k tokens per call instead of
  1k, and median time-to-first-token is 3–10 seconds.
- **Session-aware retention alone is a small, consistent gain** over LRU in the same engine
  (+2.5 to +8.5 points of hit rate, 10–22% fewer prefill tokens, 8–22% lower TTFT p99) and
  **does not prevent the collapse**. When the live sessions do not fit, which one is
  evicted matters little: half of this trace's pauses are under 0.3 s, so every victim is
  about to return.
- **A host-memory tier is what decides this workload, for every engine.** With one, all
  three are back at the ceiling at 64 sessions, at 90–94% at 96 and at 85–90% at 128.
  Copying a 30,000-token history back from host memory takes 32 ms on this machine;
  recomputing it takes 928 ms:

<!-- BEGIN host-tier-cost -->
| history (tokens) | KV (MB) | restore from host (ms) | GB/s | recompute (ms) | recompute / restore |
|---|---|---|---|---|---|
| 1,024 | 56 | 1.2 | 44 | 23 | 18× |
| 4,096 | 224 | 4.8 | 45 | 93 | 19× |
| 8,192 | 448 | 9.1 | 48 | 193 | 21× |
| 16,384 | 896 | 17.8 | 49 | 423 | 24× |
| 30,000 | 1,641 | 31.8 | 50 | 928 | 29× |
<!-- END host-tier-cost -->

- **With a host tier, vLLM is the fastest system here and minfer the one that recomputes
  least.** At 96 and 128 sessions minfer computes 33–42% fewer prefill tokens per call
  than vLLM or SGLang with their offloading (it offloads at every pause and demotes by
  expected return), and still delivers 0.79× of vLLM's throughput with a higher TTFT at
  the median and the tail. The cause is not KV management but the decode path described
  [below](#where-minfer-loses-to-vllm-and-sglang-and-why): on this trace every engine step
  has prefill work, which is where minfer's two passes per step cost most (per-token
  latency 1.56× vLLM's at 64 sessions, where both are at the hit-rate ceiling).

## Results

All numbers below are generated from the JSON files in [`bench/results/`](bench/results) by
[`bench/readme_tables.py`](bench/readme_tables.py); none is typed by hand
(`python -m bench.readme_tables --check` fails if this file is stale). Hardware and
software versions are in [`bench/ENV.md`](bench/ENV.md): one H100 80GB, vLLM 0.34.0,
SGLang 0.5.23, PyTorch 2.14, FlashInfer 0.7.3.

### Agent workload: minfer vs vLLM vs SGLang

**Qwen2.5-7B-Instruct** (baselines with the flags the fairness audit selected):

<!-- BEGIN agent-results -->
| rate (req/s) | engine | out tok/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | E2E p50 (ms) | E2E p99 (ms) | prefix hit rate | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | minfer | 88 | **49.0** | **89.5** | 6.64 | 7.87 | 443 | 1497 | 93.6% | 0 |
|  | vLLM | 88 | 78.2 | 120.5 | **6.26** | **7.09** | 453 | 1451 | 93.2% | 0 |
|  | SGLang | **89** | 56.8 | 96.6 | 6.27 | 7.34 | **433** | **1408** | **93.7%** | 0 |
| 2 | minfer | 143 | **53.1** | **86.1** | 6.79 | 7.56 | 476 | 1585 | 93.1% | 0 |
|  | vLLM | 143 | 80.5 | 124.5 | **6.38** | **6.95** | 475 | 1505 | 92.8% | 0 |
|  | SGLang | 143 | 60.4 | 104.3 | 6.41 | 7.21 | **455** | **1477** | **93.2%** | 0 |
| 4 | minfer | 256 | **54.5** | **97.6** | 7.27 | 10.38 | 500 | 1449 | 93.1% | 0 |
|  | vLLM | 256 | 78.4 | 127.1 | **6.72** | **7.71** | 485 | 1367 | 92.7% | 0 |
|  | SGLang | **257** | 61.6 | 104.4 | 6.78 | 8.05 | **474** | **1342** | **93.2%** | 0 |
| 8 | minfer | 415 | **56.0** | **117.9** | 7.93 | 10.74 | 546 | 1762 | 93.0% | 0 |
|  | vLLM | 417 | 81.1 | 136.4 | **7.01** | **8.83** | 512 | 1603 | 92.6% | 0 |
|  | SGLang | **418** | 65.0 | 124.7 | 7.19 | 9.26 | 512 | 1603 | **93.1%** | 0 |
| 16 | minfer | 743 | **57.3** | **118.4** | 9.78 | 13.13 | 645 | 2277 | 93.2% | 0 |
|  | vLLM | **749** | 81.4 | 136.2 | **7.88** | **10.16** | **555** | **1790** | 92.8% | 0 |
|  | SGLang | 748 | 67.0 | 121.0 | 8.20 | 11.21 | 561 | 1864 | **93.3%** | 0 |
| max | minfer | 1836 | **106.0** | **316.3** | 27.07 | 30.58 | 1684 | 5474 | 92.8% | 0 |
|  | vLLM | 2258 | 161.1 | 835.4 | **19.89** | **23.86** | 1379 | **4078** | 92.5% | 0 |
|  | SGLang | **2264** | 146.0 | 338.0 | 20.75 | 25.04 | **1368** | 4189 | **92.9%** | 0 |
<!-- END agent-results -->

![Agent workload, 7B](bench/plots/agent_vs_rate.png)

**Qwen2.5-1.5B-Instruct:**

Multi-turn agent sessions ([`bench/agent_workload.py`](bench/agent_workload.py): three
agent types, ~4k-token system prompts with JSON tool schemas, 5–15 turns, tool results
appended every turn; prompts of 4k–14k tokens, outputs of 33–240 tokens), replayed from
identical traces against each server. Greedy decoding, same model and dtype
(Qwen2.5-1.5B-Instruct, bf16), one server at a time, prefix caching on in all three; the
baselines run with the flags the [fairness audit](#fairness-audit) selected. Rate is the nominal request rate; "max" starts 64 sessions at once with
no think time. A value is bold when it is the best of the three at the printed precision.

<!-- BEGIN agent-results-1.5b -->
| rate (req/s) | engine | out tok/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | E2E p50 (ms) | E2E p99 (ms) | prefix hit rate | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | minfer | 89 | **37.3** | **66.4** | 2.54 | 2.95 | 189 | 613 | 93.6% | 0 |
|  | vLLM | 89 | 62.2 | 97.0 | 2.41 | 3.50 | 206 | 589 | 93.2% | 0 |
|  | SGLang | 89 | 47.3 | 72.0 | **2.25** | **2.53** | **182** | **551** | **93.7%** | 0 |
| 2 | minfer | 146 | **39.1** | **70.9** | 2.58 | 3.01 | 203 | 655 | 93.1% | 0 |
|  | vLLM | 146 | 64.4 | 102.7 | 2.41 | 2.98 | 218 | 604 | 92.8% | 0 |
|  | SGLang | 146 | 48.3 | 73.4 | **2.26** | **2.60** | **195** | **574** | **93.2%** | 0 |
| 4 | minfer | 269 | **38.6** | **94.2** | 2.66 | 3.31 | 201 | 555 | 93.1% | 0 |
|  | vLLM | 269 | 62.8 | 103.9 | 2.41 | 3.42 | 209 | 527 | 92.7% | 0 |
|  | SGLang | **270** | 47.3 | 100.0 | **2.35** | **2.69** | **188** | **503** | **93.2%** | 0 |
| 8 | minfer | 446 | **38.4** | **78.7** | 2.80 | 3.53 | 214 | 637 | 93.0% | 0 |
|  | vLLM | 446 | 64.8 | 128.3 | **2.44** | 3.42 | 220 | 598 | 92.6% | 0 |
|  | SGLang | **447** | 51.9 | 99.4 | 2.45 | **2.96** | **206** | **566** | **93.1%** | 0 |
| 16 | minfer | 798 | **40.4** | **85.0** | 2.99 | 4.23 | 224 | 700 | 93.2% | 0 |
|  | vLLM | 801 | 64.8 | 134.3 | **2.50** | **3.34** | 218 | 629 | 92.8% | 0 |
|  | SGLang | 801 | 55.3 | 106.4 | 2.59 | 3.35 | **216** | **625** | **93.3%** | 0 |
| max | minfer | 4172 | **84.4** | **274.2** | 10.90 | 12.94 | 723 | 2356 | 92.8% | 0 |
|  | vLLM | 4890 | 145.2 | 712.6 | 7.78 | 9.58 | 650 | 1808 | 92.5% | 0 |
|  | SGLang | **5741** | 114.7 | 367.4 | **6.83** | **7.99** | **532** | **1498** | **92.9%** | 0 |
<!-- END agent-results-1.5b -->

![Agent workload, 1.5B](bench/plots/agent_vs_rate_1.5b.png)

### ShareGPT (control workload)

**Qwen2.5-7B-Instruct:**

<!-- BEGIN sharegpt-results -->
| rate (req/s) | engine | out tok/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | E2E p50 (ms) | E2E p99 (ms) | failed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | minfer | 214 | **22.9** | **48.9** | 6.32 | 6.69 | 1051 | 4910 | 0 |
|  | vLLM | 214 | 37.6 | 58.9 | 6.21 | **6.53** | 1054 | 4883 | 0 |
|  | SGLang | 214 | 26.0 | 103.6 | **6.19** | 6.60 | **1033** | **4838** | 0 |
| 2 | minfer | 424 | **21.3** | **28.5** | 6.39 | 7.12 | 1059 | 4958 | 0 |
|  | vLLM | 424 | 35.9 | 50.8 | 6.28 | **6.72** | 1057 | 4906 | 0 |
|  | SGLang | 424 | 29.3 | 42.5 | **6.26** | 7.04 | **1042** | **4898** | 0 |
| 4 | minfer | 817 | **22.1** | **41.2** | 6.59 | 7.50 | 1073 | 5113 | 0 |
|  | vLLM | **819** | 34.6 | 55.9 | **6.28** | **6.93** | **1049** | **4891** | 0 |
|  | SGLang | 818 | 28.5 | 44.0 | 6.44 | 7.31 | 1062 | 5051 | 0 |
| 8 | minfer | 1664 | **22.5** | **39.5** | 7.01 | 8.71 | 1252 | 5435 | 0 |
|  | vLLM | **1669** | 38.4 | 63.3 | **6.50** | **7.52** | **1169** | **5061** | 0 |
|  | SGLang | 1667 | 30.8 | 45.9 | 6.77 | 8.47 | 1196 | 5263 | 0 |
| 16 | minfer | 3086 | **24.4** | **50.3** | 7.97 | 10.64 | 1240 | 6169 | 0 |
|  | vLLM | **3106** | 47.8 | 82.2 | **6.82** | **7.73** | **1084** | **5397** | 0 |
|  | SGLang | 3098 | 36.4 | 55.2 | 7.66 | 10.76 | 1193 | 5974 | 0 |
| max | minfer | 12184 | 4668.2 | 10751.1 | 15.33 | 26.65 | 7864 | 15588 | 0 |
|  | vLLM | **14737** | 3480.3 | 7599.9 | **12.50** | **23.88** | **6462** | **12532** | 0 |
|  | SGLang | 14008 | **3278.5** | **3394.7** | 33.12 | 136.83 | 8102 | 13749 | 0 |
<!-- END sharegpt-results -->

![ShareGPT, 7B](bench/plots/sharegpt_vs_rate.png)

**Qwen2.5-1.5B-Instruct:**

Single-turn chat requests with little shared prefix, driven by vLLM's own benchmark client
(`vllm bench serve`). This workload does not play to minfer's design; it shows what the
engine does when there is nothing to reuse, measured by a client this repository did not
write.

<!-- BEGIN sharegpt-results-1.5b -->
| rate (req/s) | engine | out tok/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | E2E p50 (ms) | E2E p99 (ms) | failed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | minfer | 215 | **11.2** | **20.1** | 2.24 | 2.42 | **369** | 1723 | 0 |
|  | vLLM | 215 | 25.0 | 46.3 | **2.17** | 3.77 | 393 | **1704** | 0 |
|  | SGLang | 215 | 21.1 | 95.1 | 2.18 | **2.36** | 375 | 1734 | 0 |
| 2 | minfer | 429 | **10.2** | **15.1** | 2.28 | **2.42** | **371** | 1783 | 0 |
|  | vLLM | 428 | 25.4 | 40.1 | **2.16** | 3.65 | 380 | **1718** | 0 |
|  | SGLang | 429 | 29.0 | 50.5 | 2.24 | 2.84 | 393 | 1857 | 0 |
| 4 | minfer | 857 | **11.0** | **18.0** | 2.33 | **2.63** | 384 | 1827 | 0 |
|  | vLLM | 857 | 24.3 | 45.6 | **2.17** | 3.36 | **368** | **1736** | 0 |
|  | SGLang | 857 | 24.9 | 47.8 | 2.32 | 2.86 | 399 | 1849 | 0 |
| 8 | minfer | 1740 | **11.6** | **18.0** | 2.42 | 2.73 | 432 | 1882 | 0 |
|  | vLLM | 1740 | 26.5 | 45.7 | **2.21** | **2.53** | **416** | **1757** | 0 |
|  | SGLang | 1740 | 27.4 | 46.7 | 2.43 | 3.44 | 467 | 1975 | 0 |
| 16 | minfer | 3209 | **12.1** | **20.3** | 2.55 | 3.24 | 398 | 2009 | 0 |
|  | vLLM | **3211** | 28.7 | 51.1 | **2.30** | **2.66** | **381** | **1844** | 0 |
|  | SGLang | 3209 | 29.2 | 50.1 | 2.66 | 4.91 | 430 | 2217 | 0 |
| max | minfer | **24834** | 2606.2 | 5320.2 | **6.91** | **10.21** | **4137** | **7513** | 0 |
|  | vLLM | 21466 | **2511.4** | 6829.6 | 17.08 | 26.16 | 6340 | 9362 | 0 |
|  | SGLang | 14695 | 3273.3 | **3389.0** | 35.84 | 121.79 | 8519 | 13663 | 0 |
<!-- END sharegpt-results-1.5b -->

![ShareGPT, 1.5B](bench/plots/sharegpt_vs_rate_1.5b.png)

### Fairness audit

The first version of these tables ran every engine at its defaults. That is a rule a
reviewer can fairly object to, so the comparison was audited at 1.5B
([`bench/fairness_audit.py`](bench/fairness_audit.py); every setting of the three servers
is listed side by side in DESIGN.md D-038). Each baseline was measured with the flags its
documentation offers for a latency-sensitive, prefix-heavy workload, on the same traces,
and all three engines were measured once more with the prompts sent as token ids, which
takes server-side tokenization out of the path. The tables above already use what the
audit selected; this is the audit itself, as measured at the time.

<!-- BEGIN fairness-audit -->
| engine / configuration | flags added to the defaults | 4 req/s: TTFT p50 / p99 (ms) | 4 req/s: TPOT p50 / p99 (ms) | 16 req/s: TTFT p50 / p99 (ms) | 16 req/s: TPOT p50 / p99 (ms) | max: tok/s | max: TTFT p50 / p99 (ms) | max: TPOT p50 / p99 (ms) |
|---|---|---|---|---|---|---|---|---|
| minfer / default | — | 39 / 67 | 2.64 / 3.15 | 40 / 82 | 2.99 / 4.18 | 4211 | 84 / 258 | 10.99 / 12.74 |
| vLLM / default | — | 62 / 99 | 2.38 / 2.96 | 64 / 127 | 2.54 / 3.39 | 4987 | 157 / 644 | 7.61 / 9.75 |
| vLLM / no access log | `--disable-uvicorn-access-log` | 63 / 95 | 2.40 / 3.09 | 65 / 119 | 2.49 / 3.18 | 4937 | 158 / 727 | 7.47 / 9.22 |
| vLLM / xxhash | `--prefix-caching-hash-algo` `xxhash` | 58 / 101 | 2.38 / 3.00 | 66 / 130 | 2.54 / 3.39 | 4954 | 149 / 709 | 7.58 / 9.39 |
| vLLM / interactivity | `--performance-mode` `interactivity` | 61 / 99 | 2.39 / 3.22 | 65 / 124 | 2.46 / 3.35 | 4917 | 173 / 744 | 7.39 / 9.58 |
| vLLM / throughput | `--performance-mode` `throughput` | 63 / 103 | 2.40 / 3.42 | 66 / 126 | 2.55 / 3.40 | 4899 | 133 / 737 | 7.78 / 9.54 |
| SGLang / default | — | 49 / 95 | 2.32 / 3.03 | 50 / 114 | 2.60 / 3.48 | 5071 | 103 / 453 | 8.33 / 10.77 |
| SGLang / lpm | `--schedule-policy` `lpm` | 49 / 94 | 2.34 / 3.07 | 48 / 130 | 2.60 / 3.55 | 5021 | 122 / 465 | 8.11 / 10.60 |
| SGLang / batch tokenizer | `--enable-dynamic-batch-tokenizer` | 48 / 86 | 2.36 / 2.72 | 55 / 107 | 2.59 / 3.18 | 5796 | 116 / 402 | 6.83 / 8.27 |
| SGLang / mixed chunk | `--enable-mixed-chunk` | 49 / 95 | 2.33 / 3.05 | 48 / 122 | 2.48 / 3.43 | 5131 | 308 / 540 | 5.69 / 7.73 |
| SGLang / torch compile | `--enable-torch-compile` | 141 / 193 | 69.87 / 70.87 | 132 / 214 | 40.10 / 72.23 | 1225 | 120 / 472 | 9.00 / 70.78 |
| minfer / token ids | (prompts sent as token ids) | 16 / 34 | 2.65 / 3.23 | 17 / 32 | 2.98 / 4.28 | 4214 | 33 / 216 | 11.50 / 13.56 |
| vLLM / token ids | (prompts sent as token ids) | 36 / 68 | 2.39 / 3.14 | 36 / 87 | 2.52 / 3.26 | 5207 | 88 / 406 | 8.34 / 9.93 |
| SGLang / token ids | (prompts sent as token ids) | 19 / 30 | 2.35 / 2.77 | 20 / 35 | 2.60 / 3.33 | 5569 | 31 / 346 | 8.61 / 10.09 |
| vLLM / tuned | `--disable-uvicorn-access-log` | 62 / 100 | 2.40 / 3.03 | 64 / 126 | 2.56 / 3.55 | 4979 | 125 / 651 | 7.75 / 9.63 |
| SGLang / tuned | `--enable-dynamic-batch-tokenizer` | 48 / 83 | 2.36 / 2.72 | 53 / 105 | 2.61 / 3.28 | 5780 | 113 / 373 | 6.83 / 8.20 |
<!-- END fairness-audit -->

What the audit changed:

- **vLLM's defaults are already its best configuration here.** No flag moved its numbers
  beyond the spread of its own repeated runs. It was the only server writing an access-log
  line per request; that is now off, for parity, with no measurable effect.
- **SGLang gains 14% saturated throughput from its batch tokenizer**
  (`--enable-dynamic-batch-tokenizer`), which the first tables did not give it. With it
  SGLang is the fastest of the three at saturation at 1.5B and minfer's deficit there is
  0.73×, not the 0.83× first published. Every benchmark since runs SGLang with that flag. Its `torch.compile` mode,
  the other documented candidate, was several times slower on this workload.
- **minfer's TTFT lead survives against the tuned baselines** (39–40 ms vs 62–64 ms and
  48–53 ms at 4–16 req/s) **but it is not all engine.** With token-id prompts the three
  engines' TTFT is 16–17, 36 and 19–20 ms, and when saturated SGLang's median (31 ms) is
  level with minfer's (33 ms). About two thirds of the lead over SGLang on text prompts
  comes from how the front ends tokenize a 25 KB prompt; the lead over vLLM is in the
  engine path either way.
- **The Triton decode kernel's published speed-up over FlashInfer was against the wrong
  baseline**; see [Component measurements](#component-measurements).

### Where minfer loses to vLLM and SGLang, and why

minfer is behind in three places, all visible in the tables above.

1. **Time per output token, at every rate, and more so under load at 7B.** At 1.5B a
   decode step was bound by the number of kernels it launches, and Phase 11 closed most of
   a 2× gap by fusing them. At 7B the matmuls dominate, and the gap grows with the request
   rate: 6% at 1 req/s, 24% at 16 req/s. The likely cause — consistent with that
   load-dependence, but not isolated by a measurement at 7B — is the structural difference
   that is left: in a step that has both prefill and decode work, minfer runs two passes
   over 15 GB of weights (a ragged prefill pass and a fixed-shape decode pass, the design
   that keeps the decode graphs simple) where vLLM runs one mixed batch, and the busier
   the server, the more steps are mixed.
2. **Saturated throughput on the agent workload: 0.81× of both baselines at 7B**,
   presumably for the same reason: with 64 sessions running flat out almost every step is
   mixed. Prefill passes above 1,024 tokens also still run eagerly rather than from a graph.
3. **Throughput at extreme concurrency on ShareGPT at 7B.** With 1,000 requests submitted
   at once vLLM and SGLang deliver 14–15k tok/s and minfer 12k, and minfer's median TTFT is
   the worst of the three (4.7 s against 3.3–3.5 s): it caps the number of running
   sequences at 256 and queues the rest. At 1.5B the same cap gave it the highest
   throughput; at 7B it does not. That is a default (`--max-num-seqs`), not a measured
   optimum.

Unifying the two passes was the next item of the original plan; the plan was revised to
study KV management on real traces first (below), and the decode path is as it was at
Phase 12. One more caveat: the agent traces use scripted assistant turns so that every
engine sees identical prompts; a live agent loop re-sends the model's own output, which
this benchmark cannot credit any engine for caching.

### Ablations

The synthetic agent workload against `minfer serve` at 1.5B with one engine feature changed
at a time ([`bench/ablations.py`](bench/ablations.py)); the KV-management techniques are
ablated on real traces at 7B in the [headline](#headline-result) instead. The first row is the default system; the
last rows constrain the engine so that the scheduling features have decisions to make.

<!-- BEGIN ablations -->
| configuration | flags | 16 req/s: tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | hit rate | max: tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | hit rate |
|---|---|---|---|---|---|---|---|---|---|
| full system (defaults) | — | 800 | 37 / 86 | 2.98 / 4.24 | 93.2% | 4208 | 83 / 234 | 11.08 / 12.83 | 92.8% |
| − prefix cache | `--no-prefix-cache` | 762 | 109 / 438 | 6.21 / 17.72 | — | 1204 | 1715 / 2486 | 18.39 / 20.32 | — |
| − cache-aware admission (FCFS) | `--no-cache-aware-scheduling` | 797 | 38 / 79 | 2.96 / 4.24 | 93.2% | 4298 | 76 / 147 | 10.97 / 12.18 | 92.8% |
| − CUDA graphs (decode and prefill) | `--no-cuda-graphs` | 752 | 47 / 338 | 7.90 / 17.48 | 93.2% | 3361 | 73 / 270 | 13.07 / 14.64 | 92.8% |
| − prefill graphs only | `--no-prefill-graphs` | 798 | 40 / 86 | 3.13 / 4.64 | 93.2% | 4042 | 75 / 270 | 11.64 / 13.78 | 92.8% |
| − chunked prefill | `--no-chunked-prefill` | 800 | 37 / 80 | 2.98 / 4.25 | 93.2% | 4257 | 76 / 223 | 11.08 / 12.64 | 92.8% |
| − fused kernels | `--no-fused-kernels` | 780 | 43 / 89 | 5.05 / 6.71 | 93.2% | 3207 | 78 / 266 | 14.81 / 16.85 | 92.8% |
| − Triton decode (FlashInfer decode) | `--decode-kernel` `flashinfer` | 798 | 38 / 77 | 3.21 / 4.45 | 93.2% | 3485 | 69 / 241 | 14.03 / 16.93 | 92.8% |
| + speculation: prompt lookup | `--speculation` `ngram` | 779 | 43 / 87 | 5.55 / 10.49 | 93.2% | 3297 | 76 / 241 | 13.19 / 21.90 | 92.8% |
| + speculation: draft model | `--speculation` `draft` | 731 | 75 / 124 | 11.72 / 18.97 | 93.2% | 1852 | 141 / 424 | 22.29 / 39.75 | 92.8% |
| + FP8 KV cache | `--kv-cache-dtype` `fp8` | 791 | 43 / 87 | 4.19 / 6.00 | 93.2% | 2646 | 78 / 264 | 19.46 / 23.32 | 92.8% |
| + FP8 weights | `--weight-dtype` `fp8` | 765 | 49 / 97 | 6.88 / 11.27 | 93.2% | 2572 | 80 / 254 | 18.91 / 21.66 | 92.8% |
| 16 running sequences max | `--max-num-seqs` `16` | 800 | 37 / 81 | 2.97 / 4.25 | 93.2% | 2848 | 1081 / 1649 | 5.17 / 7.80 | 92.8% |
| 16 running sequences max, FCFS | `--max-num-seqs` `16` `--no-cache-aware-scheduling` | 799 | 40 / 79 | 2.96 / 4.19 | 93.2% | 2942 | 970 / 1204 | 5.08 / 7.21 | 92.8% |
| KV cache 8% of GPU memory | `--kv-memory-fraction` `0.08` | 800 | 39 / 85 | 3.04 / 4.23 | 93.2% | 4054 | 81 / 252 | 11.21 / 15.18 | 92.1% |
| KV cache 8%, FCFS | `--kv-memory-fraction` `0.08` `--no-cache-aware-scheduling` | 799 | 38 / 78 | 3.01 / 4.36 | 93.2% | 4128 | 74 / 228 | 11.42 / 14.69 | 92.3% |
| KV cache 8%, FP8 KV | `--kv-memory-fraction` `0.08` `--kv-cache-dtype` `fp8` | 791 | 43 / 91 | 4.22 / 6.37 | 93.2% | 2679 | 76 / 293 | 19.09 / 21.12 | 92.8% |
| KV cache 8%, no prefix cache | `--kv-memory-fraction` `0.08` `--no-prefix-cache` | 764 | 106 / 317 | 6.33 / 17.82 | — | 1201 | 1714 / 2512 | 18.39 / 20.25 | — |
<!-- END ablations -->

Reading the table:

- **The prefix cache is what the engine is for.** Turning it off costs a factor of several
  in saturated throughput and in time-to-first-token (second row), and the same again when
  memory is tight (last row).
- **The execution-path work matters next**: fused kernels, CUDA graphs and the Triton
  decode kernel each account for a visible share of per-token latency and throughput.
- **Cache-aware admission shows no benefit in any scenario measured** — within noise with
  the default limits, and slightly worse than FCFS when a queue is forced. Every waiting
  request on this workload already has a long cached prefix, so the policy has little to
  choose between. It stays as an option; this result is the honest status of the feature.
- **Chunked prefill is neutral on this workload** (cache hits leave only a few hundred new
  tokens per request, below the chunk size). Its trade-off on long uncached prompts is in
  the chunked-prefill table below.
- **Speculation and FP8 make this engine slower per token** (sections below).

### Component measurements

**Decode step, one optimization at a time** (wall-clock time of an engine step in
steady-state decode; [`bench/decode_latency.py`](bench/decode_latency.py)):

<!-- BEGIN decode-latency -->
| batch | context (tokens) | eager (ms) | + CUDA graphs (ms) | + Triton decode (ms) | + fused kernels (default) (ms) | eager → default |
|---|---|---|---|---|---|---|
| 1 | 256 | 10.95 | 3.74 | 3.52 | 2.17 | 5.0× |
| 2 | 256 | 10.97 | 3.95 | 3.71 | 2.23 | 4.9× |
| 4 | 256 | 11.04 | 4.00 | 3.80 | 2.25 | 4.9× |
| 8 | 256 | 11.07 | 4.21 | 3.95 | 2.33 | 4.7× |
| 16 | 256 | 11.06 | 4.34 | 4.07 | 2.37 | 4.7× |
| 32 | 256 | 11.20 | 4.67 | 4.29 | 2.51 | 4.5× |
| 64 | 256 | 11.35 | 5.11 | 4.67 | 2.80 | 4.1× |
| 128 | 256 | 11.55 | 5.92 | 5.33 | 3.35 | 3.4× |
| 256 | 256 | 11.80 | 7.54 | 6.70 | 4.54 | 2.6× |
| 1 | 2048 | 10.89 | 3.84 | 3.61 | 2.27 | 4.8× |
| 2 | 2048 | 11.01 | 4.05 | 3.81 | 2.34 | 4.7× |
| 4 | 2048 | 11.09 | 4.13 | 3.94 | 2.39 | 4.6× |
| 8 | 2048 | 11.06 | 4.47 | 4.17 | 2.52 | 4.4× |
| 16 | 2048 | 11.18 | 4.99 | 4.44 | 2.72 | 4.1× |
| 32 | 2048 | 11.27 | 5.53 | 4.94 | 3.19 | 3.5× |
| 64 | 2048 | 11.48 | 7.00 | 5.98 | 4.07 | 2.8× |
| 128 | 2048 | 11.99 | 9.84 | 7.93 | 5.98 | 2.0× |
| 256 | 2048 | 17.90 | 15.97 | 12.24 | 10.33 | 1.7× |
<!-- END decode-latency -->

![Decode latency](bench/plots/decode_latency.png)

**What the profiler showed** ([`bench/profile_engine.py`](bench/profile_engine.py), traces
in `bench/results/profiles/`): the decode step was bound by kernel count. "Before" is the
engine without fused kernels, Triton decode or prefill graphs.

<!-- BEGIN profile -->
| scenario | CUDA kernels/step before | after | GPU ms/step before | after |
|---|---|---|---|---|
| decode graph batch1 | 1219 | 398 | 3.63 | 2.03 |
| decode graph batch64 | 1219 | 399 | 5.11 | 2.58 |
| decode eager batch1 | 1219 | 398 | 3.75 | 2.04 |
| prefill 2048 tokens | 1248 | 399 | 15.56 | 10.98 |
<!-- END profile -->

**One prefill pass** ([`bench/prefill_latency.py`](bench/prefill_latency.py)): an eager
pass costs about the same Python dispatch time whatever its size; fused kernels, and a
CUDA graph for the smaller passes, remove most of it.

<!-- BEGIN prefill-latency -->
| prompt tokens | eager (ms) | eager + fused kernels (ms) | default (ms) | speedup |
|---|---|---|---|---|
| 16 | 10.91 | 6.86 | 2.38 | 4.6× |
| 64 | 10.86 | 6.93 | 2.61 | 4.2× |
| 256 | 10.73 | 6.93 | 3.53 | 3.0× |
| 512 | 10.69 | 6.93 | 4.47 | 2.4× |
| 1024 | 11.39 | 7.19 | 7.09 | 1.6× |
| 2048 | 17.74 | 12.29 | 12.34 | 1.4× |
<!-- END prefill-latency -->

**Triton paged-attention decode kernel vs FlashInfer's**
([`bench/triton_kernel.py`](bench/triton_kernel.py)). *Kernel only* is GPU time per layer
when the kernel is replayed from a CUDA graph (no Python launcher, no host-side planning);
*step* is one decode step's worth of attention in wall-clock time, host preparation
included (FlashInfer's `plan()`, or the Triton path's page-table copy) plus 28 layers.
FlashInfer has two decode configurations; the comparison is against the faster one.

<!-- BEGIN triton -->
| batch | context | FlashInfer, plain (µs/layer) | FlashInfer, tensor cores (µs/layer) | Triton (µs/layer) | kernel only: Triton vs best FlashInfer | step: FlashInfer plan + 28 layers (ms) | step: Triton prep + 28 layers (ms) | end to end: Triton vs FlashInfer | max abs difference |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 11.1 | 6.6 | 7.0 | 0.94× | 0.22 | 0.21 | 1.04× | 0.0020 |
| 8 | 128 | 11.4 | 7.1 | 8.0 | 0.88× | 0.24 | 0.24 | 0.97× | 0.0020 |
| 32 | 128 | 15.5 | 10.7 | 5.6 | 1.92× | 0.34 | 0.17 | 1.95× | 0.0039 |
| 64 | 128 | 15.7 | 11.3 | 5.9 | 1.90× | 0.35 | 0.18 | 1.91× | 0.0039 |
| 256 | 128 | 29.8 | 17.6 | 8.4 | 2.09× | 0.53 | 0.25 | 2.09× | 0.0039 |
| 1 | 1,024 | 11.6 | 7.5 | 7.2 | 1.05× | 0.25 | 0.22 | 1.14× | 0.0010 |
| 8 | 1,024 | 15.7 | 11.6 | 9.0 | 1.29× | 0.36 | 0.27 | 1.33× | 0.0010 |
| 32 | 1,024 | 28.7 | 15.5 | 11.7 | 1.33× | 0.47 | 0.35 | 1.36× | 0.0010 |
| 64 | 1,024 | 43.7 | 33.5 | 29.1 | 1.15× | 0.98 | 0.83 | 1.17× | 0.0010 |
| 256 | 1,024 | 157.2 | 97.7 | 93.0 | 1.05× | 2.79 | 2.63 | 1.06× | 0.0020 |
| 1 | 4,096 | 15.5 | 11.2 | 10.9 | 1.03× | 0.35 | 0.32 | 1.10× | 0.0005 |
| 8 | 4,096 | 28.0 | 15.2 | 14.2 | 1.08× | 0.46 | 0.42 | 1.12× | 0.0005 |
| 32 | 4,096 | 78.2 | 54.9 | 52.0 | 1.06× | 1.58 | 1.48 | 1.07× | 0.0005 |
| 64 | 4,096 | 151.8 | 99.5 | 96.5 | 1.03× | 2.83 | 2.72 | 1.04× | 0.0005 |
| 256 | 4,096 | 586.9 | 359.6 | 357.7 | 1.01× | 10.13 | 10.02 | 1.01× | 0.0005 |
| 1 | 16,384 | 23.6 | 13.4 | 24.7 | 0.54× | 0.41 | 0.71 | 0.58× | 0.0002 |
| 8 | 16,384 | 79.2 | 55.1 | 54.9 | 1.00× | 1.59 | 1.56 | 1.02× | 0.0002 |
| 32 | 16,384 | 294.2 | 188.3 | 187.3 | 1.01× | 5.32 | 5.27 | 1.01× | 0.0002 |
| 64 | 16,384 | 581.3 | 364.3 | 366.8 | 0.99× | 10.26 | 10.30 | 1.00× | 0.0002 |

Kernel only, against the faster FlashInfer configuration, the Triton kernel is 0.54–2.09× as fast: more than 5% faster at 9 of 19 points, more than 5% slower at 3, within 5% at 7. End to end (host preparation included) it is 0.58–2.09×.
<!-- END triton -->

![Triton kernel](bench/plots/triton_kernel.png)

An earlier version of this table reported the Triton kernel as 1.35–3.5× faster at 18 of
19 points. That compared it with FlashInfer's plain decode path; for this head layout
FlashInfer's tensor-core path (the one SGLang selects) is faster everywhere, and against
it the honest summary is the sentence under the table: a clear win only for many
sequences with short contexts, parity from 4k-token contexts up, and a loss at batch 1
with a very long context. minfer's own FlashInfer decode option now uses the tensor-core
path too.

**Chunked prefill** on a mixed long/short workload
([`bench/chunked_prefill.py`](bench/chunked_prefill.py)): smaller chunks bound the stalls
that long prompts inflict on running streams (ITL = inter-token latency), and cost
time-to-first-token.

<!-- BEGIN chunked-prefill -->
| chunk (tokens) | ITL p99 (ms) | ITL max (ms) | TPOT p50 (ms) | TPOT p99 (ms) | long-prompt TTFT p50 (ms) | out tok/s |
|---|---|---|---|---|---|---|
| off | 5.6 | 114 | 2.45 | 5.00 | 69 | 1703 |
| 4096 | 25.0 | 41 | 2.45 | 5.31 | 82 | 1702 |
| 2048 | 17.7 | 26 | 2.45 | 5.47 | 100 | 1700 |
| 1024 | 12.8 | 26 | 2.51 | 5.89 | 121 | 1697 |
| 512 | 11.9 | 14 | 2.58 | 9.29 | 208 | 1687 |
| 256 | 10.8 | 14 | 3.04 | 9.54 | 462 | 1653 |
<!-- END chunked-prefill -->

**Speculative decoding** ([`bench/speculation.py`](bench/speculation.py)), engine-level,
greedy:

<!-- BEGIN speculation -->
| workload | speculation | out tok/s | vs off | TPOT p50 (ms) | tokens/step | accepted | outputs identical to off |
|---|---|---|---|---|---|---|---|
| agent | off | 3199 | — | 6.54 | 16.6 | — | — |
| agent | ngram | 2483 | 0.78× | 5.21 | 30.4 | 75% | 234/356 |
| agent | draft model | 1413 | 0.44× | 12.23 | 54.3 | 74% | 230/356 |
| sharegpt | off | 35777 | — | 5.05 | 186.8 | — | — |
| sharegpt | ngram | 14832 | 0.41× | 12.25 | 189.3 | 56% | 91/256 |
| sharegpt | draft model | 9809 | 0.27× | 14.89 | 366.2 | 66% | 83/256 |
<!-- END speculation -->

That table is the Phase 7 design, in which a verification span ran through the eager
prefill pass: most drafts are accepted, steps emit twice as many tokens, and it is still a
net slowdown. Verification is now replayed from a CUDA graph that returns every row's
argmax, with the sequences that have no draft riding in the same pass. Through the server
at 7B on the agent workload ([`bench/speculation_serve.py`](bench/speculation_serve.py)):

<!-- BEGIN speculation-serve -->
| configuration | TPOT p50 at 1 req/s (ms) | TPOT p50 at 4 req/s (ms) | TPOT p50 at 8 req/s (ms) | E2E p50 at 1 req/s (ms) | E2E p50 at 4 req/s (ms) | E2E p50 at 8 req/s (ms) | drafted tokens accepted | output tokens from drafts |
|---|---|---|---|---|---|---|---|---|
| no speculation | 6.63 (1.00×) | 7.26 (1.00×) | 7.88 (1.00×) | 441 | 493 | 545 | — | — |
| prompt lookup, n=3, up to 4 drafted | 5.76 (0.87×) | 6.84 (0.94×) | 8.21 (1.04×) | 391 | 472 | 596 | 62–64% | 24–28% |
| prompt lookup, n=2, up to 4 drafted | 5.91 (0.89×) | 6.99 (0.96×) | 9.01 (1.14×) | 401 | 498 | 635 | 54–55% | 23–26% |
| prompt lookup, n=4, up to 4 drafted | 6.03 (0.91×) | 6.81 (0.94×) | 8.52 (1.08×) | 387 | 488 | 609 | 64–67% | 21–26% |
| prompt lookup, n=3, up to 8 drafted | 5.71 (0.86×) | 6.71 (0.92×) | 7.93 (1.01×) | 384 | 463 | 570 | 52–54% | 26–29% |
| prompt lookup, n=5, up to 8 drafted | 5.96 (0.90×) | 6.77 (0.93×) | 8.09 (1.03×) | 386 | 457 | 587 | 55–58% | 22–25% |
| prompt lookup, n=3, up to 4 drafted, never paused | 5.59 (0.84×) | 6.72 (0.93×) | 8.27 (1.05×) | 389 | 474 | 594 | 61–63% | 25–29% |
| prompt lookup, n=3, up to 4 drafted, eager verification (Phase 7) | 6.34 (0.96×) | 8.36 (1.15×) | 11.17 (1.42×) | 422 | 619 | 774 | 62–63% | 24–28% |
<!-- END speculation-serve -->

With prompt lookup (n=3, up to 8 drafted tokens) the median time per output token is 14%
lower than without speculation at 1 req/s and 8% lower at 4 req/s, and level at 8 req/s.
The last row is the same drafts verified the old way: slower than no speculation at 4 and
8 req/s. Two costs remain: the 99th-percentile TPOT is worse in every speculative
configuration (a rejected draft pays for rows that produced nothing), and the gain fades
as load rises, because a step with drafts carries several rows per sequence and at 7B a
step costs what its rows cost. It stays off by default. "Outputs identical to off" in the
engine-level table is below 100% because bf16 argmax near-ties flip when the same position
is evaluated by a different kernel (see Correctness); on the CPU in float32 the outputs are
exactly identical for any draft.

**FP8** ([`bench/gsm8k.py`](bench/gsm8k.py)): GSM8K accuracy on the first 200 test
problems (standard error ±3.3 points), KV capacity in the same memory budget, weight
memory, and decode-step latency.

<!-- BEGIN fp8 -->
| setting | GSM8K (200) | KV capacity | weights | decode step ms (batch 1 / 64 / 256) |
|---|---|---|---|---|
| bf16 | 69.0% | 1.00× | 2.89 GB | 2.16 / 2.95 / 5.31 |
| FP8 KV cache | 68.5% | 1.92× | 2.89 GB | 2.74 / 4.15 / 8.63 |
| FP8 weights | 70.5% | 1.00× | 1.67 GB | 4.42 / 6.92 / 9.62 |
| FP8 weights + KV | 71.0% | 1.91× | 1.67 GB | 5.04 / 8.12 / 12.82 |
<!-- END fp8 -->

FP8 costs no measurable accuracy (all four settings are within one standard error of each
other) and **nearly doubles KV capacity**, but it makes decode steps **slower**, not
faster: a 1.5B model's decode step is bound by kernel count rather than by the bytes its
matmuls read, and FP8 adds kernels (quantizing activations and K/V) while the FP8-weight
path cannot use the fused kernels at all. FP8 is therefore a memory feature here and off by
default — and in the one memory-constrained scenario measured (last ablation rows) the
extra capacity restored the cache hit rate but did not make up for the slower steps.

One finding worth knowing about: an *all*-FP8 KV cache destroyed the model's output
(GSM8K accuracy fell to almost nothing). Layer 0's keys have a few components two orders of
magnitude larger than the rest, and E4M3's 3-bit mantissa turns its 6% relative rounding
error on those into attention-logit errors far above anything a softmax tolerates. No
scale fixes a relative error, so calibration measures each layer's logit error and keeps
the offending layers (for this model: layer 0 only) in bf16 — which is why capacity grows
a little less than 2×. Root-cause analysis and measurements: DESIGN.md B-005.

### Correctness

- **Model.** Float32 logits match Hugging Face `transformers` on five diverse prompts
  (max abs difference 8e-5, `bench/results/model_numerics.json`); the fused-kernel path is
  tested against the reference path kernel by kernel and end to end.
- **Engine vs a cache-free oracle (exact).** On a small float32 model on the CPU, greedy
  engine output is *identical* to recomputing the whole sequence at every step — under
  serial and batched execution, chunked prefill, memory pressure with preemption and
  recompute, the prefix cache with eviction, speculation with drafts of any quality, and
  two-process tensor parallelism. That is what the CPU test-suite (run in CI) asserts.
- **On the GPU in bf16, bit-identical output is not available to any engine.** Greedy
  decoding takes an argmax, and when float32 separates the top two tokens by less than
  bf16 rounding noise the winner depends on kernel and batch shape. vLLM disagrees with
  *itself* between a batched and an unbatched run on about half of 20 prompts. The table
  shows minfer against vLLM, and the prefix cache on against off, next to those noise
  floors; every difference was traced to a float32 near-tie.

<!-- BEGIN equivalence -->
| comparison (greedy, bf16) | result | largest float32 margin at a difference |
|---|---|---|
| minfer vs vLLM, 20 prompts, one request at a time | 11/20 outputs identical | 0.21 |
| minfer (batched) vs vLLM (one at a time) | 10/20 outputs identical | 0.21 |
| **vLLM (batched) vs vLLM (one at a time)** — the noise floor | 9/20 outputs identical | — |
| minfer fed vLLM's tokens (teacher-forced), per position | 1985/2001 tokens equal | 0.30 |
| prefix cache on vs off, 64 agent-style requests | 46/64 outputs identical | 0.13 |
| **cache off vs cache off, different batching** — the noise floor | 46/64 outputs identical | 0.20 |

In the prefix-cache run the cache served 74.7% of prompt tokens and cut the tokens run through the model from 60,572 to 14,302.
<!-- END equivalence -->

- **Tests.** `make test` runs the whole suite on the GPU; the CPU subset (block manager,
  radix tree, scheduler, sampling, engine vs oracle, speculation, tensor parallelism, FP8
  storage, workload generator, server bridge) runs in CI on every push.

## Related work, and what is and is not new here

Nothing in this repository is a new mechanism. Each technique it implements was published
by others; the citations below were checked against the papers' own pages (details and
caveats in DESIGN.md, "Related work").

| technique here | where it comes from |
|---|---|
| Radix-tree prefix cache over paged KV | SGLang's RadixAttention; vLLM's automatic prefix caching |
| Session-aware retention (keep the KV of a session paused in a tool call) | **Continuum** (Li et al., arXiv 2511.02230): KV time-to-live from per-tool durations; **InferCept** (Abhyankar et al., ICML 2024): tool calls as interceptions; **CacheWise** (Tiwari et al., arXiv 2606.16824): eviction by predicted time to next reuse |
| Host-memory KV tier | LMCache; vLLM's native KV offloading; SGLang's hierarchical cache; sized for agents by **EfficientAgent** (Shao et al., arXiv 2609.33762) |
| Position-independent KV reuse with selective recomputation | **CacheBlend** (Yao et al., EuroSys 2025); **EPIC** (Hu et al., ICML 2025); **MiniPIC** (Ordonez and Parnell, arXiv 2606.13126); for agents: **CacheSlide** (Liu et al., FAST '26), **Irminsul** and **Leyline** (Ma et al., arXiv 2605.05696, 2606.01065) |
| Prompt-lookup speculative decoding | prompt lookup decoding as shipped in transformers and vLLM |
| The workload's timings | **TraceLab** (Zhu et al., arXiv 2606.30560) |

What this repository contributes is a measurement: these techniques in **one** engine
where each is a flag, run alone and in every valid combination on the **same** real
coding-agent trace, under memory pressure, next to vLLM and SGLang with their own
offloading. A literature search (five queries, listed in DESIGN.md) found no published
evaluation that composes prefix caching, tool-call-aware retention, a host-memory tier,
position-independent reuse and speculative decoding on real agent traces and reports how
they interact. That is a negative search result, not proof: the closest work found —
Continuum (retention + DRAM/SSD offload), CacheWise (eviction + DRAM swap on real Claude
Code traces), "Stateful Inference for Low-Latency Multi-Agent Tool Calling"
(arXiv 2605.26289: persistent KV + prefix cache + prompt lookup on generated workflows)
and the vLLM × AgentX write-up of September 2026 — each covers a subset, and four more
recent preprints that might overlap (arXiv 2610.13060, 2610.01108, 2611.18459, 2605.00528)
were found by title only and not read.

Two limits on what the measurement can claim. The trace is assembled from two public
sources because no public dataset has both content and timing (real token sequences from
Open-SWE-Traces, real tool-call latencies from TraceLab, paired by tool name), and it is
cut at this model's 32k-token context, a fraction of real Claude Code sessions. And it is
one model on one GPU.

## What is not done

- **Tensor parallelism has never run on two GPUs.** The development machine has one H100.
  The implementation is verified with two CPU processes exchanging all-reduces over gloo
  (outputs identical to a single process, KV memory per rank halved); the NCCL path and any
  scaling number for Qwen2.5-14B on 2×H100 do not exist.
- **Modal.** `bench/modal_bench.py` and `deploy/modal_app.py` are written against the
  pinned Modal SDK and import cleanly, but were never run: no credentials. There is no
  hosted endpoint.
- **Reproduction on a second machine was not performed.** A fresh clone on the *same*
  machine, with empty caches and re-downloaded weights, was: see
  [Reproduction check](#reproduction-check).
- **The decode gap to vLLM and SGLang at 7B is open** (6% per token at 1 req/s, 24% at
  16 req/s, 0.81× saturated throughput). Unified prefill+decode passes were implemented
  and measured; they lose at saturation and are off by default (DESIGN.md M-025).
- **Speculative decoding** helps only on a lightly loaded server and hurts a saturated one.
- **One model family** (Qwen2/2.5), text only, no tool-call parsing on the response side,
  no `n > 1`, no penalties, no logprobs.

## Reproducing

```bash
git clone https://github.com/visshbala21/minfer && cd minfer
scripts/setup_envs.sh all      # three pinned virtualenvs: minfer, vLLM, SGLang (needs uv)
make test                      # full test suite (GPU + weights); `make test-cpu` is the CI subset
make bench                     # every benchmark above, the plots, and the tables in this file
```

`make bench` needs one H100-class GPU, nvcc on `PATH` (FlashInfer compiles its kernels on
first use), about 100 GB of free `/dev/shm` for vLLM's KV offloading, and most of a day:
it runs one server at a time, and nothing else may run on the machine meanwhile (a test
suite running next to it visibly corrupted a baseline's numbers once; DESIGN.md B-006).
Individual pieces: `make bench-micro` (engine-level, 1.5B), `make bench-agent` (the
three-engine matrix at 7B and 1.5B), `make bench-ablations`, `make bench-audit`,
`make bench-real` (real traces: pressure sweep, host tier, reuse quality, speculation,
composition), `make plots`. Every results file written since the 7B phase records the
commit it was measured at. To run it on
a fresh H100 without a local GPU: `modal run bench/modal_bench.py` (written but not
executed from the development machine, which has no Modal credentials).

### Reproduction check

This check was made at the end of Phase 12, on the 1.5B results as they were then; the
tables it refers to as "the README run" have since been re-measured.

The commands above were run once more from a fresh `git clone` in a new directory, with an
empty `HOME` (no JIT caches for FlashInfer, Triton, vLLM or SGLang) and an empty `HF_HOME`
(weights and datasets downloaded again): the three environments built from the lock files,
`make test` passed (272 tests) and `make bench` ran to completion. It was the same physical
H100, so this checks that the repository is self-contained and how far the numbers move
between two complete runs — not that they carry over to another machine. The tables above
are from the original run; `bench/reproduction.py` compares the two
(`bench/results/reproduction.json`, rerun results in `bench/results/reproduction_rerun/`).

<!-- BEGIN reproduction -->
| agent workload | engine | README run | rerun | rerun / README run |
|---|---|---|---|---|
| TTFT p50, 1–16 req/s (ms) | minfer | 38–41 | 37–41 | 0.94–1.01× |
|  | vLLM | 59–63 | 55–67 | 0.93–1.09× |
|  | SGLang | 44–47 | 47–51 | 1.03–1.10× |
| TTFT p99, 1–16 req/s (ms) | minfer | 68–82 | 65–82 | 0.92–1.06× |
|  | vLLM | 95–122 | 97–140 | 0.95–1.14× |
|  | SGLang | 78–114 | 84–114 | 0.99–1.09× |
| TPOT p50, 1–16 req/s (ms) | minfer | 2.54–2.99 | 2.55–3.02 | 1.00–1.01× |
|  | vLLM | 2.25–2.40 | 2.26–2.39 | 0.98–1.01× |
|  | SGLang | 2.25–2.57 | 2.25–2.57 | 0.99–1.00× |
| TTFT p50, "max" (ms) | minfer | 83 | 79 | 0.95× |
|  | vLLM | 144 | 139 | 0.96× |
|  | SGLang | 119 | 122 | 1.03× |
| TPOT p50, "max" (ms) | minfer | 10.71 | 11.15 | 1.04× |
|  | vLLM | 7.40 | 7.60 | 1.03× |
|  | SGLang | 8.08 | 8.26 | 1.02× |
| output tok/s, "max" | minfer | 4219 | 4215 | 1.00× |
|  | vLLM | 5106 | 5031 | 0.99× |
|  | SGLang | 5049 | 4995 | 0.99× |

- minfer has the lowest TTFT p50 at every rate from 1 to 16 req/s: holds in both runs
- minfer has the lowest TTFT p99 at every rate from 1 to 16 req/s: holds in both runs
- minfer has the lowest TTFT p50 and p99 at saturation: holds in both runs
- minfer's TPOT p50 is higher than vLLM's at every rate: holds in both runs
- minfer's saturated throughput is below vLLM's and SGLang's: holds in both runs
- no request failed for any engine on either workload: **differs** (README run: True, rerun: False)
<!-- END reproduction -->

The one difference in kind: in the rerun SGLang dropped one connection out of 120 in its
ShareGPT run at 1 req/s (the first request of that run took 15 s; nothing in its server
log explains it), where the original run had no failed request for any engine. The
saturated ShareGPT run ("max", 1,000 requests at once) is the least repeatable
measurement here, for the baselines in particular: vLLM's median TTFT was 1.5 s in the
original run and 4.0 s in the rerun, SGLang's throughput 16.1k and 13.6k tok/s; minfer's
moved by 6% and 4%. Treat that row as an order of magnitude.

The agent traces are deterministic: `python -m bench.agent_workload generate --sessions 96
--rate 16 --seed 4 --out trace.jsonl` writes the trace used at 16 req/s, and
`python -m bench.agent_workload replay trace.jsonl --base-url http://host:port` replays it
against any OpenAI-compatible server.

## Serving

```bash
.venv/bin/minfer serve --model Qwen/Qwen2.5-7B-Instruct --port 8000 --host-kv-gb 96
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi."}],"max_tokens":16}'
```

Endpoints: `/v1/chat/completions`, `/v1/completions` (SSE streaming on both), `/v1/models`,
`/health`, `/metrics` (Prometheus: throughput, queue depth, prefix-cache hit rate, KV
usage), `/tokenize`, `/detokenize`. Every feature has a flag (`minfer serve --help`):
`--[no-]prefix-cache`, `--[no-]cache-aware-scheduling`, `--[no-]cuda-graphs`,
`--[no-]prefill-graphs`, `--prefill-chunk-tokens N`, `--[no-]fused-kernels`,
`--decode-kernel {auto,triton,flashinfer}`, `--speculation {none,ngram,draft}`,
`--kv-cache-dtype {auto,fp8}`, `--weight-dtype {auto,fp8}`, `--eviction-policy
{session,lru}`, `--host-kv-gb N`, `--[no-]position-independent-reuse`,
`--[no-]unified-batches`.

A request can name the agent session it belongs to, which is what session-aware retention
and the host tier act on: `"session_id": "..."`, and optionally `"session_tools": ["Bash"]`
(the tools the client is about to run) or `"session_final": true`. Without the last two
the server reads the tool calls off its own response.

`modal deploy deploy/modal_app.py` puts the same server behind a scale-to-zero endpoint on
an H100 (written, not deployed: no Modal credentials on the development machine, so there
is no public endpoint to link to).

## License

MIT — see [LICENSE](LICENSE).
