# minfer — Design log

This file is the engineering notebook for the build described in [PLAN.md](PLAN.md): every
design decision and its tradeoff, every bug with root cause and fix, and every before/after
measurement. Entries are append-only and numbered (`D-nnn` decisions, `B-nnn` bugs,
`M-nnn` measurements) so the README and commit messages can cite them.

Rule for numbers: nothing in this file is estimated. A measurement entry names the script
that produced it and, where applicable, the committed results file.

---

## Phase 0 — Environment

### D-001 Dependency versions: match the baselines' torch and FlashInfer

- **Decision.** minfer pins `torch==2.14.0` (CUDA 13.0 build), `triton==3.8.0` and
  `flashinfer-python==0.7.3`.
- **Why.** These are exactly the versions `vllm==0.34.0` pins (SGLang 0.5.23 pins torch 2.14.1
  and FlashInfer 0.7.2). Running the same PyTorch/cuBLAS build and the same FlashInfer release
  as the primary baseline removes library versions as a confounder: a throughput difference
  between minfer and vLLM is then attributable to the engine.
- **Tradeoff.** Not the newest releases (torch 2.15.0 and FlashInfer 0.7.5 exist). Accepted:
  comparability beats recency for a benchmark project.
- Direct dependencies are pinned with `==` in `pyproject.toml`; transitive pins are in
  `uv.lock`. The baselines are pinned by full `uv pip freeze` output in
  `bench/envs/{vllm,sglang}.txt` (top-level specs: `vllm[bench]==0.34.0`, `sglang==0.5.23`).

### D-002 Three isolated virtualenvs, built on local disk

- **Decision.** `.venv` (minfer), `.venv-vllm`, `.venv-sglang` are separate environments
  created by `scripts/setup_envs.sh`. On this machine they are built under `/opt/venvs`
  (`MINFER_VENV_ROOT`) and symlinked into the repo; a plain clone builds them in place.
- **Why separate.** vLLM and SGLang pin conflicting versions of torch, transformers and
  FlashInfer; isolating them means neither can perturb minfer's environment, or each other's.
- **Why local disk.** The checkout lives on an NFS volume. Measured with the same wheel set
  (torch 2.14.0 + FlashInfer 0.7.3):

  | | NFS venv | local venv |
  |---|---|---|
  | `import torch`, cold | 22.3 s | 3.0 s |
  | `import torch`, warm | 5.6 s | 1.3 s |
  | `import flashinfer`, cold | 26.8 s | 6.2 s |

  (`time python -c "import ..."`, one run each, 2026-11-20.) The test loop pays the warm
  import cost on every invocation, so the 4× matters. The uv cache stays on the persistent
  volume, so rebuilding an environment after a machine reset is a copy, not a download.

### D-003 The environment record is generated, not written

`bench/ENV.md` is produced by `bench/collect_env.py`, which queries `nvidia-smi` and each
virtualenv's own interpreter. Hand-maintained version tables go stale; a generated one
cannot disagree with what was actually installed.

### D-004 FlashInfer kernels are JIT-compiled with the system toolkit

- FlashInfer 0.7.3 ships sources and compiles each kernel specialization on first use with
  `nvcc` + `ninja`, caching the result under `~/.cache/flashinfer/0.7.3/`. Here that is the
  system CUDA 12.8 toolkit, while PyTorch carries its own CUDA 13.0 runtime. The two coexist:
  the smoke test (batch paged decode, bf16, 12 query heads / 2 KV heads / head_dim 128,
  page size 16) compiles in ~7 s and matches an fp32 PyTorch reference with max abs error
  1.8e-3 on outputs of mean magnitude 0.089 — bf16-level agreement.
- **Gotcha.** FlashInfer invokes `ninja` by bare name. It is installed into the venv's `bin/`,
  which is only on `PATH` when the venv is activated; calling `.venv/bin/python` directly
  fails with `FileNotFoundError: 'ninja'`. minfer therefore puts the interpreter's `bin/`
  on `PATH` before importing FlashInfer (see `minfer/attention`), and the Makefile exports it.

### Constraints of the build machine

- One H100. Phase 10 follows the plan's single-GPU fallback.
- No Modal credentials. The Modal apps can be written and linted here but not deployed.
- "You are the only process allowed to use the GPU": benchmarks and GPU tests are always run
  strictly one at a time; baseline servers are started and torn down per run.

---

## Phase 1 — Model

### D-005 One forward pass; attention sits behind a seam

- **Decision.** `Qwen2Model.forward(token_ids, positions, attn)` runs over a *flattened* batch
  of `n` tokens with explicit positions — no batch dimension, no padding. Each layer computes
  rotated `q`/`k` and `v` and calls `attn.attend(layer, q, k, v)`; the `AttentionContext`
  decides where K/V live and which kernel runs.
- **Why.** The engine needs three attention regimes — ragged prefill batches, single-token
  decode batches, and (in tests) cache-free full sequences — and later a second decode kernel.
  Putting the variation behind one method means there is exactly one implementation of the
  layer math to verify against the reference, and the model knows nothing about paging.
- `forward` returns hidden states and `logits()` is separate: the unembedding is a
  151,936 × 1,536 matmul and the engine only needs it for the rows it will sample from.

### D-006 Numerics mirror transformers op-for-op

RMSNorm is evaluated in float32 with the weight applied after casting back; RoPE is the
rotate-half formulation with tables computed in float32 and then cast to the model dtype;
the MLP is SwiGLU. Matching the reference's operation order (rather than a mathematically
equivalent rearrangement) is what makes the bf16 path bit-identical to transformers in the
common case (M-001) instead of merely close.

### D-007 RoPE tables are precomputed on the CPU

- **Decision.** `cos`/`sin` tables of shape `[max_positions, head_dim]` are built once on the
  CPU in float32 and follow the model's device/dtype; the forward pass gathers rows by position.
- **Tradeoff, measured** (`rope_tables_cpu_vs_gpu` in `bench/results/model_numerics.json`):
  CPU and GPU `cos` disagree in the last float32 bit for 752,938 of 4,194,304 entries
  (`sin`: 670,524). After the cast to bf16 only 16 entries per table differ; the first
  affected position is 593 for `cos` and 2,402 for `sin`. transformers evaluates the same
  formula on the GPU per forward pass, so beyond position 593 a bf16 run can differ from it
  by one bf16 ulp in a handful of rotation coefficients.
- **Why accept it.** CPU tables are device-independent (the CPU reference tests and, later,
  every tensor-parallel rank see identical coefficients), and it is the construction vLLM
  uses for its `cos_sin_cache`, which is the engine minfer has to match token-for-token in
  Phase 2. At fp32 the one-ulp difference is far inside the 1e-3 budget.

### D-008 Weights load straight onto the device

`load_model` reads each tensor with `safetensors.safe_open(device=...)` and installs it with
`load_state_dict(assign=True)`; placeholder weights are never initialized (`skip_init`). Host
memory therefore never holds a second copy of the model. The checkpoint's tensor names are
checked against the architecture in both directions and any mismatch raises — a silently
half-loaded model is the worst failure mode a weight loader can have.

### M-001 Logit equivalence with transformers

Script: `bench/model_numerics.py` → `bench/results/model_numerics.json`. Qwen2.5-1.5B-Instruct,
five prompts (English prose, Python, Chinese, a chat-templated multi-turn conversation,
JSON with unicode), every position compared. "fp32 ref" is transformers in float32.

| prompt | tokens | fp32: max abs vs HF | bf16: max abs vs HF bf16 | minfer bf16 vs fp32 ref (max / mean / top-1 / mean KL) | HF bf16 vs fp32 ref (max / mean / top-1 / mean KL) |
|---|---|---|---|---|---|
| english | 112 | 4.67e-05 | 0.000 | 1.150 / 0.034 / 0.9464 / 7.0e-04 | 1.150 / 0.034 / 0.9464 / 7.0e-04 |
| code | 161 | 6.15e-05 | 0.469 | 1.365 / 0.038 / 1.0000 / 2.5e-04 | 1.365 / 0.037 / 1.0000 / 2.3e-04 |
| chinese | 78 | 4.86e-05 | 0.000 | 2.067 / 0.047 / 0.9744 / 1.6e-03 | 2.067 / 0.047 / 0.9744 / 1.6e-03 |
| chat | 115 | 8.01e-05 | 0.000 | 2.523 / 0.085 / 0.9739 / 3.5e-03 | 2.523 / 0.085 / 0.9739 / 3.5e-03 |
| structured | 147 | 5.15e-05 | 0.453 | 3.048 / 0.039 / 0.9388 / 1.5e-03 | 3.048 / 0.041 / 0.9524 / 1.5e-03 |

- **fp32 (the acceptance criterion): worst case 8.0e-5 against a 1e-3 budget.**
- **bf16 tolerance.** There is no useful *elementwise* bf16 tolerance: bf16 itself lands up to
  3.05 logits from the fp32 answer at isolated positions, for transformers exactly as for
  minfer (logit standard deviation is about 3, so that is a large error on a few logits, and a
  small one on average — mean abs error is at most 0.085). The documented tolerance is
  therefore stated against the fp32 reference with aggregate metrics, asserted in
  `tests/test_model.py` with headroom over the measured worst case:

  | metric (minfer bf16 vs fp32 reference) | measured worst | asserted |
  |---|---|---|
  | mean abs logit error | 0.085 | ≤ 0.15 |
  | mean KL(ref ‖ minfer) per position | 3.5e-3 nats | ≤ 1e-2 |
  | top-1 agreement | 0.939 | ≥ 0.90 |
  | max abs logit error | 3.05 | ≤ 5.0 |
  | mean abs error relative to HF's own bf16 run | 0.95×–1.02× | ≤ 1.25× |

  Top-1 disagreements only occur where the fp32 margin between the two best tokens is small;
  the median fp32 top-1 margin on these prompts is 0.85–4.3 logits.
- Against transformers' *own* bf16 run, minfer is bit-identical on three of the five prompts
  in the committed run and differs by up to 0.47 logits on the other two. That is B-001.

### B-001 bf16 scaled-dot-product attention returns two different results for identical inputs (open; PyTorch-level, not a minfer bug)

- **Symptom.** With transformers and minfer both loaded in one process, their bf16 logits are
  usually bit-identical, but for some prompts in some processes they differ by 0.34–0.87.
- **Reproduction.** `bench/repro/sdpa_bf16_divergence.py` (12 fresh processes × 5 prompts) →
  `bench/results/repro_sdpa_bf16_divergence.json`: 9 of 60 prompt-runs diverged, in 7 of the
  12 processes. In all 9, the layer-0 inputs to
  `F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)` were
  bit-identical between the two implementations and the outputs were not. In all 60 runs,
  calling the function a second time on the same tensors reproduced the first result
  (0 unstable repeats). Each prompt has exactly one alternative value (e.g. the Chinese
  prompt diverges by 0.438 every time it diverges), so this is a choice between two
  computations, not noise.
- **Ruled out** (scratch experiments, not committed): the individual operations of the SDPA
  math path — `matmul`, `bmm`, `softmax`, `linear`, the scaled q/k products — are bit-stable
  over thousands of repeats, over hundreds of relocations of their inputs to different
  addresses, and on the real layer-0 tensors replayed in a fresh process; RoPE tables are
  identical across processes; no cuBLAS fallback warning is emitted; a single model per
  process never diverged (6 processes).
- **Hypothesis (unconfirmed).** GQA forces PyTorch's "math" SDPA path, which is a batched bf16
  `matmul` through cuBLAS. Something outside the tensor values — the leading candidate is the
  placement of the temporaries the path allocates, which differs between the two call sites —
  selects between two GEMM code paths with different accumulation order. I stopped here: the
  time box for the investigation was spent and the result does not gate anything (below).
- **Why it does not block.** (1) The acceptance criterion is fp32, where the worst error is
  8.0e-5. (2) Both bf16 variants are legitimate bf16 evaluations: they differ from each other
  by at most 0.87 logits while each differs from fp32 by up to 3.05. (3) The dense SDPA path
  is the test oracle only; the engine's attention runs through FlashInfer.
- **Consequences adopted.** bf16 tests compare against the fp32 reference with aggregate
  metrics rather than against another bf16 run elementwise; later phases compare paged
  attention against the dense oracle in fp32 wherever a tight tolerance is wanted.

---

## Phase 2 — Paged KV cache + continuous batching

### D-009 Block manager: integers only, reference counted, block 0 reserved

- `BlockManager` (`minfer/cache/block_manager.py`) owns a LIFO free list, a reference count
  per block, and the block table of every live sequence. It holds no tensors, so it is
  unit-testable on the CPU and its invariants (a block is free iff its count is zero; every
  table entry is backed by a reference) are asserted by `check_invariants` under a
  4,000-operation randomized test.
- Blocks are reference counted from the start, although nothing shares them until Phase 4:
  `adopt()` lets a new sequence begin its table with another owner's blocks, and
  `replace_block()` exists for copy-on-write of a partially filled shared block. Designing the
  sharing in now avoids retrofitting ownership rules into the scheduler later.
- Block 0 is never allocated. Fixed-shape (CUDA-graph) batches are padded with dummy rows, and
  those rows need somewhere harmless to read from and write to.
- `ensure_capacity` is all-or-nothing: on `OutOfBlocks` the table is unchanged, so the
  scheduler can react (evict, preempt) without undoing a partial allocation.
- Budget: `kv_bytes_per_block` × blocks ≤ a fraction of the device memory free *after* the
  weights are loaded. For Qwen2.5-1.5B in bf16 with 16-token blocks a block is
  28 layers × 2 (K,V) × 16 × 2 heads × 128 × 2 bytes = 458,752 bytes, i.e. 28.7 KB per token.

### D-010 KV layout: one tensor, addressed by slot

- `PagedKVStore` holds a single `[layers, blocks, 2, block_size, kv_heads, head_dim]` tensor;
  `layer(i)` is the 5-D "NHD" paged cache FlashInfer takes directly (keys at index 0, values
  at index 1 of the third axis).
- A token's **slot** is `block · 2·block_size + offset`, the row of its key in the layer's
  flattened view; its value is `block_size` rows later. A KV write is therefore two
  `index_copy_` calls whose index tensor has shape `[n_tokens]` and nothing else — no
  per-sequence structure, which is the property CUDA-graph capture will need in Phase 3.
- The store is zero-initialized so padded rows that read block 0 see finite values.

### D-011 One batch shape for prefill, verification and decode

- `PagedBatch` describes a pass as, per sequence, a run of new tokens at positions
  `[start, start+len)` with everything before `start` already cached. A prefill chunk, a
  single decode token and (Phase 7) a run of speculative tokens are the same shape, so there
  is one builder (`build_paged_batch`, vectorized in numpy) and one model entry point.
- Per-token tensors (ids, positions, slots) go to the device; the CSR page table
  (`kv_indptr`, `kv_indices`, `kv_last_page_len`) and `qo_indptr` stay on the CPU.
  FlashInfer's `plan()` begins with `indptr.to("cpu")` to do its work partitioning on the
  host, so handing it device tensors would force a device-to-host sync every step.

### D-012 A plain-PyTorch paged backend is the oracle (and the CPU backend)

- FlashInfer 0.7.3 kernels exist for fp16, bf16 and fp8 only (`flashinfer/jit/utils.py`
  `dtype_map`); there is no float32 and no CPU path. `ReferencePagedBackend` implements the
  same contract with a gather per sequence and a position mask
  (`kv_position <= query_position`), in any dtype on any device.
- It is deliberately slow and obvious. Its jobs are (1) to let the whole engine — scheduler,
  prefix cache, speculation — run in CI on a tiny random model, and (2) to give the
  accelerated backends a float32 reference, sidestepping the bf16 ambiguity of B-001.
- Verified on the CPU (`tests/test_paged_attention.py`, tiny random 3-layer model, block
  size 4): hidden states from ragged prefill chunks, mixed-length decode batches,
  interleaved (non-contiguous) block tables, and a sequence that *adopts* another sequence's
  blocks for a shared prefix all match one cache-free pass to 2e-5.

### D-013 Separate prefill and decode passes per step

- **Decision.** An engine step runs at most one prefill pass (ragged spans, FlashInfer
  `BatchPrefillWithPagedKVCacheWrapper`, `causal=True`) and one decode pass (one token per
  sequence, `BatchDecodeWithPagedKVCacheWrapper`), SGLang-style, instead of one mixed ragged
  batch, vLLM-style.
- **Why.** The decode pass has a shape that depends only on the batch size, which is what a
  CUDA graph can capture per bucket (Phase 3); prefill stays eager with dynamic shapes.
- **Cost.** Two forward passes in steps that have both kinds of work, so the weights are
  read twice. Accepted for now; revisited with measurements in Phase 3/11.
- Both wrappers share one 256 MB workspace and are planned once per pass (the page tables
  are identical for every layer). RoPE is applied by the model (`pos_encoding_mode="NONE"`),
  so the kernels see already-rotated q/k and the cache stores rotated keys.

### D-014 The pending-token invariant

An unfinished sequence always has ≥ 1 token whose KV is not cached (`Sequence.num_pending`);
running the model over all pending tokens yields the logits for the next one. First
prefill, a prefill chunk, a decode step, recompute after preemption and (Phase 4) a
prefix-cache hit are the same operation with a different pending count. The engine has
exactly two execution paths — "span of n tokens" and "one token" — and no per-case state
machine. A one-token prompt simply starts in the decode path.

### D-015 Scheduler policy

`minfer/scheduler/scheduler.py`, pure Python over integers, unit-tested on the CPU
(`tests/test_scheduler.py`, including a randomized workload under memory pressure with the
block-accounting invariants checked every step).

- **Admission** is FCFS with no overtaking: a waiting sequence is admitted when a running
  slot is free (`max_num_seqs`) and the pool can hold its whole pending context *on top of*
  what earlier admissions have not allocated yet (a long prompt being prefilled in chunks
  has claimed its blocks only partially), plus a watermark (1% of the pool) so running
  sequences can grow without immediately preempting the newcomer. With nothing running the
  head of the queue is always admitted, otherwise a request larger than
  `pool − watermark` would wait forever.
- **Token budget.** `max_batched_tokens` bounds decode + prefill tokens per step; a prompt
  larger than what is left is prefilled across steps. This is already chunked prefill in
  mechanism; Phase 3 adds the policy knob and measures it.
- **Preemption by recompute.** Blocks for the whole step are allocated before the forward
  pass. Sequences are visited oldest first; when one cannot get a block, the *youngest*
  running sequence loses its KV and goes to the *front* of the waiting queue with its
  generated tokens kept. Visiting oldest-first gives two properties for free: a victim is
  never a sequence already selected for this step, and the oldest sequence always makes
  progress (no livelock; a sequence that cannot fit in the whole pool is rejected at
  submission).
- **Tradeoff.** Recompute wastes the victim's prefill work (vs swapping KV to host memory),
  but needs no second memory tier and the victim's re-prefill is a single batched pass.
  vLLM made the same choice.

### D-016 Sampling: one uniform per row, drawn on the host

`minfer/sampling.py` is inverse-CDF sampling over the sorted distribution with filters in
the transformers/vLLM order (temperature → top-k → top-p, top-p measured after top-k). The
single uniform each row needs comes from the request's own `random.Random(seed)` on the
host. A seeded request therefore produces the same tokens whatever else is in its batch —
asserted on the CPU model across full batching, serial execution and preemption-heavy
runs — which a shared device generator cannot provide. Greedy rows skip the sort entirely.
A stop token is reported to the caller with `finish_reason="stop"` (vLLM's convention for
token ids; the server strips it from text).

### D-017 KV budget

With `num_blocks=None` the pool is `kv_memory_fraction` (default 0.9) of the device memory
free *after* the weights are loaded, divided by `kv_bytes_per_block`. The remaining 10%
covers activations, the FlashInfer workspace and, later, CUDA-graph pools. Tests and
equivalence runs pass an explicit `num_blocks` so their memory use is fixed.

### M-002 Greedy equivalence with vLLM (Phase 2 acceptance)

Scripts: `bench/vllm_greedy_fixture.py` (run in `.venv-vllm`; vLLM 0.34.0, bf16, FLASH_ATTN
backend, explicit greedy sampling parameters, prompts passed as token ids) →
`tests/fixtures/vllm_greedy.json`; `bench/vllm_equivalence.py` →
`bench/results/vllm_greedy_equivalence.json`. 20 chat prompts, `max_tokens=128`,
2,001 generated tokens in the vLLM reference. minfer runs bf16 through FlashInfer.

**Result: the outputs are *not* all token-identical, and cannot be for any bf16 engine.**

| comparison (free-running greedy, 20 prompts) | identical outputs |
|---|---|
| minfer (one request at a time) vs vLLM (one at a time) | 8 / 20 |
| minfer (20 batched) vs vLLM (one at a time) | 11 / 20 |
| **vLLM (20 batched) vs vLLM (one at a time)** | **9 / 20** |

- The third row is the noise floor: vLLM disagrees with *itself* across batch compositions
  on 11 of 20 prompts. minfer's agreement with vLLM (8–11) is at that floor.
- **Every divergence is a float32 near-tie.** At the first differing token of each of the
  21 diverging outputs (12 serial + 9 batched), a cache-free float32 pass over the common
  prefix shows that the two engines picked float32's top-1 and top-2 tokens, separated by
  at most 0.330 logits (median 0.051; smallest 7e-5).
- **Teacher-forced view** (minfer's prefill + decode kernels fed vLLM's own tokens, so all
  2,001 positions are compared rather than only those before the first flip):

  | metric over 2,001 positions | value |
  |---|---|
  | minfer bf16 argmax = vLLM's token | 1,977 (98.8%) |
  | float32 reference argmax = vLLM's token | 1,985 (99.2%) |
  | minfer bf16 argmax = float32 reference argmax | 1,979 (98.9%) |
  | minfer bf16 vs float32, mean abs logit error | 0.049 |
  | minfer bf16 vs float32, max abs logit error | 1.16 |
  | largest float32 margin at a minfer/vLLM disagreement | 0.330 |

  All 24 disagreements are between float32's two best tokens; float32 sides with vLLM in
  15 and with minfer in 9. Against float32 as the arbiter, vLLM is "wrong" at 16 positions
  and minfer at 22 — the same order, as expected from two bf16 evaluations whose individual
  logits are ~0.05 (mean) and up to ~1.2 away from the float32 value.
- **Numerical cause.** Greedy decoding takes an argmax; when float32 separates the two best
  tokens by less than bf16 rounding noise, the winner depends on accumulation order, which
  differs between attention kernels (vLLM: FlashAttention; minfer: FlashInfer), between
  prefill and decode kernels, and between GEMM shapes at different batch sizes. After the
  first flip the two continuations are different texts, which is why free-running identity
  falls so far below per-position agreement.
- **What the test asserts** (`tests/test_engine_gpu.py`): every free-running divergence has
  a first differing token (never a silent length difference) at which both tokens are
  float32's top-2 with margin < 0.5; teacher-forced agreement > 97%; mean abs logit error
  < 0.15. The 0.5 bound is ~1.5× the measured worst margin and well under the measured
  max single-logit bf16 error.
- **Honest reading of the acceptance criterion.** "Match token-for-token" holds for 8–11
  of 20 prompts; the rest are documented divergences with the cause above. I count the
  criterion as met through its "document any divergence and its numerical cause" clause,
  and report the raw 8/20 rather than a filtered number.

### M-003 64 concurrent requests (Phase 2 acceptance)

`tests/test_engine_gpu.py`: 64 chat requests submitted at once, mixed greedy and seeded
temperature/top-k/top-p sampling, 8–96 max tokens. All complete with valid finish reasons
and every block returned to the pool, (a) with an 8,192-block pool and no preemption,
(b) with a 200-block pool (3,200 tokens for a ~5,500-token peak demand), where the run
completes through preemption and recompute, and (c) with `max_batched_tokens=48`,
`max_num_seqs=3`, forcing multi-step prefill. On the CPU model the same scenarios are
asserted to give outputs *identical* to the cache-free oracle (`tests/test_engine.py`),
which the bf16 GPU path cannot promise (M-002).
