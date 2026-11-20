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

---

## Phase 3 — CUDA graphs + chunked prefill

### D-018 Decode passes replay CUDA graphs, one per batch-size bucket

- **Decision.** `minfer/attention/decode_graphs.py` captures the whole decode pass (embedding
  → 28 layers → final norm → unembedding) at batch sizes 1, 2, 4, …, 256. A batch of `n`
  runs in the smallest bucket ≥ `n`; larger batches fall back to the eager path. Toggle:
  `EngineConfig.cuda_graphs` / `--[no-]cuda-graphs`.
- **How.** Each bucket owns static input buffers (token ids, positions, KV slots — the
  slot-addressed store of D-010 exists for exactly this), a FlashInfer decode wrapper in
  CUDA-graph mode with static page-table buffers, and a static logits output. Per step the
  engine copies the inputs in, calls `plan()` (host-side; it rewrites the wrapper's
  buffers), and replays. Padding rows are a one-token sequence at position 0 whose K/V goes
  to slot 0 of the reserved block 0 (D-009).
- **Memory.** Buckets are captured largest-first into one shared graph pool; they are never
  replayed concurrently and each output is consumed before the next replay. Capturing all
  nine takes ~0.6 s at start-up.
- **Correctness.** At exact bucket sizes the replayed graph is *bit-identical* to the eager
  pass over 36 consecutive steps with changing page tables
  (`test_graph_decode_is_bit_identical_to_eager_at_bucket_sizes`, n = 1, 4, 16, 256). With
  padding rows the logits differ by bf16 rounding only (max abs < 1.5, mean < 0.05 asserted;
  measured max 0.37): padding changes the row count of every GEMM, and cuBLAS does not
  round identically across shapes — the same effect as M-002, not a property of graphs.

### M-004 Decode-step latency, CUDA graphs on vs off (Phase 3 acceptance)

Script: `bench/decode_latency.py` → `bench/results/decode_latency.json`. Wall-clock time of
`Engine.step()` in steady-state decode (scheduling + batch build + attention plan + forward
+ sampling + host sync), 200 steps per point, p50 in ms, NVIDIA H100 80GB HBM3.

| batch | ctx 256: off | on | speedup | ctx 2048: off | on | speedup |
|---|---|---|---|---|---|---|
| 1 | 11.54 | 3.75 | 3.08× | 11.52 | 3.85 | 3.00× |
| 2 | 11.61 | 3.93 | 2.95× | 11.58 | 4.04 | 2.87× |
| 4 | 11.58 | 4.00 | 2.90× | 11.67 | 4.13 | 2.83× |
| 8 | 11.68 | 4.20 | 2.78× | 11.70 | 4.46 | 2.62× |
| 16 | 11.69 | 4.34 | 2.69× | 11.72 | 4.94 | 2.37× |
| 32 | 11.78 | 4.65 | 2.53× | 11.84 | 5.52 | 2.15× |
| 64 | 11.89 | 5.11 | 2.33× | 12.09 | 6.88 | 1.76× |
| 128 | 12.10 | 5.87 | 2.06× | 12.51 | 9.65 | 1.30× |
| 256 | 12.31 | 7.38 | 1.67× | 17.01 | 15.16 | 1.12× |

- Eager decode costs ~11–12 ms regardless of batch size: it is bound by Python/PyTorch
  dispatch of ~1,000 small kernels, not by the GPU. Graphs remove that floor
  (3.1× at batch 1).
- The gain shrinks as the GPU becomes the bottleneck: 1.1× at batch 256 with 2,048-token
  contexts, where attention over 524k cached tokens dominates.
- Remaining per-step cost at batch 1 is ~3.7 ms, of which the graph replay itself is the
  GPU part; host-side work (scheduling, numpy batch build, `plan()`, sampling sync) is the
  rest. Breaking that down is Phase 11.

### D-019 Chunked prefill is a per-step prefill token budget

- **Decision.** `EngineConfig.prefill_chunk_tokens` (`--prefill-chunk-tokens N` /
  `--no-chunked-prefill`) caps the prefill tokens of a step *while at least one sequence is
  decoding*. No new mechanism was needed: under the pending-token invariant (D-014) a
  partially prefilled sequence is just a sequence with more than one pending token.
- The cap is lifted when nothing is decoding: chunking protects the gap between decode
  steps, and with no decoders small chunks would only multiply per-pass overhead.
- The budget is shared FCFS among prefilling sequences, so one long prompt cannot starve
  the prefill of short ones for more than its own chunks.

### M-005 Chunked prefill on a mixed long/short workload (Phase 3 acceptance)

Script: `bench/chunked_prefill.py` → `bench/results/chunked_prefill.json`. Engine-level
(no HTTP), real-time replay of one seeded trace per setting: Poisson 12.0 req/s, prompt 32-256 tok, output 96-192 tok
(346 requests) mixed with Poisson 1.5 req/s, prompt 4000-16000 tok, output 16-48 tok (55
requests); 592,225 prompt tokens in 30 s. CUDA graphs on.
All latencies in ms, measured on the **short** requests unless noted.

| chunk | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ITL max | TTFT short p50 / p99 | TTFT long p50 / p99 | out tok/s |
|---|---|---|---|---|---|---|---|---|
| off | 5.70 | 10.38 | 4.36 | 50.0 | 148 | 19 / 136 | 100 / 212 | 1681 |
| 4096 | 5.70 | 10.57 | 4.36 | 41.5 | 52 | 18 / 147 | 115 / 258 | 1680 |
| 2048 | 5.78 | 11.40 | 4.36 | 27.8 | 32 | 19 / 205 | 147 / 324 | 1677 |
| 1024 | 6.16 | 13.11 | 4.39 | 20.0 | 105 | 19 / 429 | 210 / 537 | 1670 |
| 512 | 7.52 | 16.14 | 4.73 | 17.5 | 22 | 111 / 1478 | 496 / 1696 | 1638 |
| 256 | 15.99 | 16.42 | 15.88 | 16.9 | 95 | 4454 / 9512 | 4526 / 9503 | 1372 |

*TPOT* is the per-request mean time per output token after the first (vLLM's definition);
*ITL* is the gap between consecutive tokens, pooled over all tokens.

**What chunking does, honestly:**

- **It bounds stalls.** Per-token p99 latency falls from 50.0 ms to 27.8 ms at chunk 2048
  and 20.0 ms at 1024; the worst stall falls from 148 ms to 32 ms at 2048.
  This is the "p99 TPOT improvement" the phase asks for, read as time per output token
  at the token level.
- **It does not improve the per-request mean TPOT; it makes it worse.** p99 of
  per-request TPOT goes from 10.38 ms (off) to 11.40 (2048) and 16.14 (512). Chunking does not
  remove prefill work, it spreads it — a request's mean TPOT pays for all prefill that
  overlaps its lifetime either way — and it *adds* work: each prefill pass is eager and
  carries the ~11 ms dispatch floor of M-004, so a 16k-token prompt in 512-token chunks
  pays that floor 32 times instead of once.
- **It costs TTFT**, for both classes, because prefill throughput is capped at
  chunk / step-time. At chunk 256 the cap (≈16k tokens/s) is at the workload's prefill
  demand and the system saturates (TTFT in seconds, throughput down 18%).
- **Default: 2048.** It removes most of the tail (ITL p99 −44%, max stall −78%) for +1% p50
  TPOT, +10% p99 TPOT and ~50 ms of long-prompt TTFT. Smaller chunks buy little extra tail
  and cost steeply.
- **Open observation.** Isolated ITL maxima of 67–105 ms appear in single settings in two
  runs of this benchmark (chunk 1024 and 256 above) although the chunk bound should prevent
  them; p99 is unaffected. Not yet explained — to be chased with a profiler in Phase 11.
- **Follow-up for Phase 11.** The eager prefill pass's fixed cost is the reason chunking is
  expensive here; removing it (graph capture for bucketed prefill shapes, or fusing the
  decode rows into the prefill pass) would change this table.

---

## Phase 4 — Radix-tree prefix cache + cache-aware scheduling

### D-020 Block-aligned radix tree

- **Decision.** `minfer/cache/radix_tree.py`: a radix tree keyed on token sequences in which
  every edge is a whole number of KV blocks. A node owns its edge's tokens and the blocks
  that hold them; children are indexed by the tokens of their first block, so descending
  costs one dictionary lookup per node plus a slice comparison along the edge.
- **Why block-aligned.** A block is the unit a page table can share. Token-granular reuse
  would need copy-on-write of a partially matching block for every hit; aligning to blocks
  makes cached blocks immutable and shared by reference with no copying at all.
- **Cost.** A prefix is reused up to its last full block: at most `block_size − 1` = 15
  matching tokens are recomputed per request (and one token is always left pending,
  D-014). Against multi-thousand-token agent prefixes that is noise; the measured hit rates
  below include it.
- **Shape.** A sequence that keeps generating lengthens its leaf's edge in place instead of
  growing a chain of one-block nodes, so the depth of the tree is proportional to the
  number of *divergences* on a path, not to its length. Edges split only when a sequence
  with blocks of its own diverges inside one.

### D-021 Ownership: the cache holds one reference; "evictable" is counted, not searched

- The tree holds exactly one reference to each of its blocks
  (`BlockManager.cache_block`); sequences hold theirs through their block tables. A cached
  block with reference count 1 is in use by nobody and is *evictable*.
- The block manager maintains `num_evictable` incrementally on every retain/release, so
  admission treats evictable blocks as available memory in O(1), with no tree scan.
- Invariant that makes that count meaningful: a sequence always references the *tree's*
  copy of every block on its path (below), so an unreferenced node never has a referenced
  descendant, and everything counted evictable really can be evicted leaf-first.

### D-022 One operation in both directions: `sync`

`PrefixCache.sync(tokens, own_blocks, limit)` walks the tree along a sequence's tokens. Where
the tree has a block, that block is canonical; where it does not and the sequence has
filled one, the sequence's block is inserted; the walk continues past the sequence's own
blocks while the tree has more. The sequence's table is then pointed at the canonical
blocks (`BlockManager.adopt`). That single operation is used at three moments:

1. **Admission** (table empty): a pure lookup — the classic prefix-cache hit.
2. **After every step** that filled a block (`Scheduler.commit`): insertion. Blocks are
   published as soon as they exist, not when the request finishes, so a request can hit on
   the prompt of another request that is still running. If two sequences computed the same
   block in the same step, the second one's copy is dropped in favour of the tree's.
3. **Before each prefill chunk** of a sequence that is still prefilling: catch-up. If
   another sequence has meanwhile cached more of this prompt, it is adopted instead of
   computed. With chunked prefill this gives in-batch deduplication for free: of N
   identical prompts arriving together, the first computes chunk 1, and by the next step
   the other N−1 find it in the tree (`test_identical_concurrent_requests_...`).

Outputs of the model are cached too (they are in the sequence's blocks like anything else),
which is what makes turn *k+1* of a session hit on all of turn *k*.

### D-023 Eviction: LRU leaves, trimmed from the tail, before any preemption

- When an allocation fails, `Scheduler._reserve` first asks the cache to evict; only if
  nothing evictable is left does it preempt a running sequence (D-015).
- Victims are least-recently-used leaves (a min-heap of `(last_access, node)` with lazy
  invalidation; a path's nodes all get the same access time, so an ancestor is never older
  than its descendants). A leaf is trimmed block by block from its tail, exactly as many as
  needed: the tokens furthest from the root are the least likely to be shared, and the head
  of a long edge (a system prompt) survives the eviction of its tail (old history).
- A preempted sequence's published blocks stay in the tree, so its recompute is a cache
  hit unless memory pressure evicts them first.

### D-024 Cache-aware admission with a starvation guard

- Among the oldest `admission_window` (32) waiting requests, the one with the longest
  cached prefix is admitted first (`PrefixCache.match_length`, a read-only walk).
  It is the cheapest to start and uses cached blocks before pressure evicts them.
- Overrides, in arrival order: a preempted sequence (it already held a place), and any
  request that has waited `starvation_seconds` (default 1 s) — a cold request cannot be
  overtaken indefinitely by a stream of hot ones (`test_starvation_guard_...`).
- `--no-cache-aware-scheduling` restores plain FCFS. Whether the preference pays off is
  an empirical question for the agent benchmark (Phase 6) and the ablations (Phase 12).

### M-006 Prefix cache: correctness and effect (Phase 4 acceptance)

**Unit tests** (`tests/test_radix_tree.py`, 22 tests): insert, match, block alignment,
split (including grandchildren), leaf extension, duplicate merging, catch-up, LRU order,
tail trimming, pinned leaves, reference counts through a full lifecycle, and a randomized
run against a brute-force list-of-prefixes model with the tree and block-manager
invariants checked after every operation.

**Outputs identical with the cache on vs off — exact, on the CPU float32 model**
(`tests/test_prefix_cache_engine.py`): six concurrent four-turn sessions with two shared
system prompts give the same tokens as the cache-off engine *and* as the cache-free
oracle, under serial execution, FCFS, chunked prefill, a tight token budget, block sizes 1
and 16, a pool small enough to force eviction, and a pool small enough to force preemption;
seeded sampling is unchanged by a hit; an aborted request leaves the tree consistent.
The accounting is exact too: tokens through the model (off) − tokens through the model
(on) = `prefill_tokens_saved`.

**On the real model in bf16** (`bench/prefix_cache_equivalence.py` →
`bench/results/prefix_cache_equivalence.json`): 16 concurrent sessions × 4 turns
(64 requests, 56,978 prompt tokens), two agent types with JSON tool schemas, greedy,
identical prompts replayed with the cache off and on.

| | cache off | cache on |
|---|---|---|
| tokens through the model | 60,622 | 14,170 (−77%) |
| prefix-cache hit rate (tokens) | — | 75.0% (42,688 / 56,914) |
| requests with a hit | — | 48 / 64 |
| tokens adopted while prefilling (catch-up) | — | 3,568 |
| wall time for the workload | 1.84 s | 1.37 s |

- The 16 first turns arrive together and miss at admission (nothing is cached yet); their
  shared system prompts are then partly recovered by catch-up.
- **Outputs: 45 of 64 are token-identical; 19 differ.** Every difference starts at a
  float32 near-tie between float32's top-2 tokens (largest margin 0.252 logits). The
  noise floor, measured in the same script by running the cache-*off* engine twice with
  different concurrency limits (only the batching differs): 54 of 64 identical,
  largest margin 0.129. So bf16 outputs are not bit-stable against *any* change in
  how a token's inputs were computed (M-002); the cache is one such change — a cached block
  was filled by a different pass than a recomputed one — and flips somewhat more near-ties
  than rebatching alone (19 vs 10), all of the same kind. The GPU test asserts
  that every divergence is such a near-tie (margin < 0.5, both tokens in float32's top 2).
- **Honest reading of the acceptance criterion.** "Outputs identical with cache on vs off"
  is established exactly where arithmetic is deterministic (float32, CPU) and up to bf16
  near-ties on the GPU, with the noise floor measured alongside.
- **Hit rate is logged**: `PrefixCacheStats` (lookups, hits, hit rate, tokens saved,
  catch-up tokens, inserted/evicted blocks), exported by the server's `/metrics` in Phase 5.

---

## Phase 5 — OpenAI-compatible server

### D-025 Two processes: HTTP front end and engine

- **Decision.** `minfer serve` runs the HTTP server (FastAPI on uvicorn/uvloop/httptools)
  in one process and the engine in a child process (`minfer/server/async_engine.py`,
  `spawn` start method because CUDA does not survive a fork). The front end tokenizes,
  applies the chat template, detokenizes incrementally and writes SSE; the engine process
  only ever sees token ids.
- **IPC.** Two one-way pipes. Front end → engine: `add` (request id, prompt ids, sampling
  params), `abort`, `stop`. Engine → front end: one `tokens` message *per engine step*
  carrying every `(request id, token id, finish reason, cached tokens)` the step produced,
  `error` for a rejected request, a `metrics` snapshot every 250 ms and whenever the engine
  goes idle, and `dead` before exiting on a failure. The front end reads the pipe from the
  event loop (`loop.add_reader`), so there are no extra threads on either side.
- **Why not a thread.** That was the first implementation; see B-002.
- **Failure handling.** If the engine process dies, every in-flight request fails with a
  server error and `/health` turns 503 (`test_engine_process_death_...`). A client that
  disconnects, or a stop string that matches, aborts the request in the engine and frees
  its KV.

### B-002 Engine thread and event loop fought over the GIL (fixed by D-025)

- **Symptom.** First server version: the engine loop on a thread inside the uvicorn
  process. Under `vllm bench serve` (random dataset, 512 prompt / 128 output tokens,
  300 requests at 30 req/s, `--temperature 0`) median TPOT was 38.5 ms and median TTFT
  140 ms, with peak concurrency 230.
- **Isolation.** The same arrival process replayed in-process with `bench/driver.py`
  (no HTTP) gave median TPOT 9.0 ms. The server's own counters showed the engine doing
  447 steps at 26.7 ms each, against ~1,400 steps at 8.5 ms in-process: the engine *itself*
  was 3× slower when an event loop streaming ~3,300 chunks/s lived in the same interpreter.
  Greedy vs sampled requests made no difference (so not the sampler), and shrinking the GIL
  switch interval from 5 ms to 0.1 ms only moved median TPOT to 33 ms (so not just handoff
  latency).
- **Root cause.** Every PyTorch operator call releases and re-acquires the GIL. An eager
  pass makes several hundred such calls per step, and at each one the event-loop thread —
  which always has chunks to serialize and write under load — can take the lock. The
  engine's per-step Python time and the front end's per-step Python time add up serially,
  and slower steps raise concurrency, which raises the front end's work per step further.
- **Fix.** Separate processes (D-025). Same benchmark, same machine: 1,550 steps at 6.7 ms,
  median TPOT 7.0 ms, median TTFT 36 ms, peak concurrency 75.
- These before/after numbers are development measurements (the thread-based version was
  never committed); the committed evidence for the current design is M-007.

### D-026 Protocol details

- Routes: `/v1/completions` (string, token-id, and batched prompts), `/v1/chat/completions`
  (chat template applied; `tools` passed to the template; multi-part text content
  accepted), `/v1/models`, `/health`, `/metrics`, and vLLM-compatible `/tokenize` and
  `/detokenize` (vLLM's benchmark client uses them to align prompt lengths).
- Streaming is SSE, one chunk per token that produces text, a final chunk with the finish
  reason, an optional usage chunk (`stream_options.include_usage`), then `[DONE]`.
- Detokenization uses `tokenizers.decoders.DecodeStream` (incremental, UTF-8 safe,
  ~0.85 µs per token measured) rather than re-decoding a growing suffix.
- Stop strings are applied on the detokenized stream with hold-back: text that could
  still become the start of a stop string is not sent, so a stop string split across
  tokens never leaks (`test_stop_string_split_across_pieces_is_never_leaked`).
- `usage.prompt_tokens_details.cached_tokens` reports the request's prefix-cache hit.
- Unknown request fields are ignored; fields that would change the output but are not
  implemented (`n > 1`, penalties, `echo`, non-text content parts) are rejected with a 400
  in OpenAI's error envelope rather than silently dropped.
- `/metrics` (Prometheus text): running/waiting requests (queue depth), token counters and
  output tokens/s, prefill/decode token counters, engine step count and time, preemptions,
  KV pool usage, and the prefix cache's hit rate, hit/lookup/catch-up tokens, tokens saved,
  tree size and evictions.

### Known limit: the sampler sorts the whole vocabulary

Requests with temperature > 0 go through a full descending sort of 151,936 logits per row
(D-016). Measured on the H100 with random logits: 1.2 ms per step at batch 32 and 5.6 ms at
batch 256, against 0.06 ms for argmax at batch 256. Benchmarks in this repository use greedy decoding
for all engines, so this is not in their hot path; a top-k pre-filter is the obvious fix
and is left for Phase 11.

### M-007 `vllm bench serve` against minfer; `openai` client (Phase 5 acceptance)

- Script: `bench/serve_compat.sh` → `bench/results/serve_compat/` (vLLM's result JSON plus
  the server's `/metrics` afterwards). vLLM 0.34.0's benchmark client, random dataset,
  512 prompt / 128 output tokens, 500 requests at 30 req/s, greedy:

  | endpoint | completed | failed | out tok/s | median TTFT | median TPOT | p99 TPOT |
  |---|---|---|---|---|---|---|
  | `/v1/completions` | 500 | 0 | 3704 | 33.4 ms | 6.81 ms | 8.12 ms |
  | `/v1/chat/completions` | 500 | 0 | 3704 | 33.6 ms | 6.91 ms | 8.43 ms |

  This is a compatibility result (the client is load-limited at 30 req/s), not a
  performance comparison; that is Phase 6.
- The first chat run failed all 500 requests: the client sends `content` as a list of
  typed parts, which the Qwen chat template does not accept. Fixed by flattening text
  parts (and rejecting non-text ones); covered by `test_chat_accepts_multi_part_text_content`.
- `tests/test_server.py` (10 tests) drives a real `minfer serve` process with the official
  `openai` Python client: chat and completions, streaming and not, streamed text equal to
  the non-streamed text, usage chunks, token-id and batched prompts, stop strings, seeded
  sampling, error envelopes, prefix-cache hits visible in `usage` and `/metrics`, and the
  raw SSE wire format.

---

## Phase 6 — Agent workload benchmark

### D-027 The workload: scripted histories, identical prompts for every engine

`bench/agent_workload.py` generates multi-turn sessions of three agent types (coder,
analyst, researcher). Each type has a ~4k-token system prompt (numbered guidelines plus
4–5 JSON tool schemas rendered through the model's chat template), shared by all of its
sessions. A session is a task followed by 5–15 turns; each turn is an assistant message
(a thought and a `<tool_call>` JSON, or the final answer) and a tool result (file contents
with line numbers, pytest logs with stack traces, grep hits, SQL result tables, search
results, fetched pages). Every request resends everything so far.

- **Scripted assistant messages.** The history uses the trace's assistant messages, not
  what the server under test generated, and each request asks for exactly the scripted
  number of output tokens (`ignore_eos`). Two servers replaying a trace therefore get
  byte-identical prompts and do the same amount of decoding, whatever their sampling noise
  (which M-002 showed is unavoidable in bf16).
- **What that costs.** A server cannot reuse the KV of its *own* generated tokens on the
  next turn (the history holds the scripted text). Real agent loops do get that reuse, so
  the benchmark understates prefix-cache benefit — for every engine equally.
- **Client-side rendering.** The replay client renders prompts with the model's chat
  template and sends text to `/v1/completions`, so all servers tokenize the same string;
  differences in server-side chat-template handling cannot leak into the comparison.
- **Timing.** Sessions arrive open-loop (Poisson); the next turn of a session is sent
  `think` seconds (exponential, mean 1 s) after the previous response completes. "Rate" is
  the nominal request rate: session arrival rate × mean turns (10). "max" starts 64
  sessions at once with no think time.
- **Determinism.** Contents depend only on (seed, session index); timing on its own
  streams. Each result records the trace's SHA-256, and the matrix uses a different seed
  per rate so one rate's sessions cannot be cache hits for the next (the system prompts
  are shared by construction — that is the workload — and are warm after the warm-up).
- Traces are JSONL (a header with the agent definitions, then one line per session) and can
  be regenerated bit-for-bit; they are not committed (several MB each).

### D-028 Fairness of the three-engine comparison

- One server at a time on the same GPU; each engine in its own pinned virtualenv; the same
  model, bf16, greedy decoding with explicit parameters, same context length.
- Every tuning choice is left at the engine's default: batching limits, attention backend,
  CUDA graphs, chunked prefill, and prefix caching (on by default in all three).
- vLLM gets `--generation-config vllm` so that it does not adopt the checkpoint's sampling
  defaults (the benchmark sends explicit greedy parameters regardless), and
  `--enable-prompt-tokens-details`; SGLang gets `--enable-metrics --enable-cache-report`.
  These only add reporting.
- SGLang runs with `SGLANG_ENABLE_JIT_DEEPGEMM=0`: its DeepGEMM JIT (an FP8 GEMM library a
  bf16 model never calls) aborts at start-up unless nvcc ≥ 12.9, and this machine has
  12.8. Its FlashInfer kernels compile with the system toolkit, as minfer's do.
- The agent workload is measured by this repository's client; the ShareGPT control workload
  is measured by vLLM's own client (`vllm bench serve`), so one of the two workloads is
  not measured by code written for minfer.
- Sample sizes: each agent run has at least 60 s of arrivals and ≥ ~230 requests (so "p99"
  at the low rates is the 3rd-worst request); ShareGPT runs have max(120, 60 × rate)
  requests and 1,000 at "max".

### B-003 Front-end tokenization serialized on the GIL (fixed)

- **Symptom.** First agent run at "max" rate through the server: TTFT p50 1,385 ms and
  1,829 output tok/s, while the same sessions driven in-process gave TTFT p50 28 ms and
  2,600 tok/s.
- **Root cause.** The server tokenized each prompt with `tokenizer.encode` in a thread
  pool. Encoding a 6k-token prompt takes 6.9 ms and holds the GIL; with 4–16 threads
  contending, the same work took 25–29 ms *per prompt* (measured in isolation). With 64
  sessions resending 25 KB prompts, requests queued for over a second before reaching
  the engine.
- **Fix.** `BatchTokenizer`: concurrent requests are encoded together with
  `encode_batch_fast` on one worker thread. It releases the GIL and parallelizes in Rust:
  0.57 ms per prompt in a batch of 64. Requests arriving during a batch form the next one,
  so there is no added wait. Same run after the fix: TTFT p50 79 ms, 2,627 tok/s.

---

## Phase 7 — Speculative decoding

### D-029 Verification rides in the prefill pass

- A sequence with a draft of `k` tokens is verified by processing the span *(pending token,
  d1 … dk)* — a ragged span like any prefill chunk (D-011, D-014). Row *i* of the span is
  the target's prediction after *i* draft tokens. Sequences without a draft take the
  ordinary (CUDA-graph) decode pass in the same step.
- **Greedy** requests accept `d_{i+1}` iff it equals the target's argmax at row *i*, then
  emit the target's own token at the first mismatch: the emitted sequence is exactly the
  non-speculative greedy output, for any draft whatsoever. This is what the CPU tests
  assert, with drafts that are all right, all wrong, and mixed.
- **Sampling** requests use speculative sampling: accept with probability
  `min(1, p(d)/q(d))`, otherwise resample from the normalized residual `max(p − q, 0)`.
  Prompt-lookup drafts are a point mass (`q = 1` on the drafted token); draft-model drafts
  carry the warped distribution they were sampled from. Tests check the emitted second and
  third tokens against the target's exact conditional distributions.
- **KV rollback is logical.** Rejected tokens' K/V was written beyond what `num_cached`
  covers afterwards; it is never read (attention is bounded by the cached length) and is
  overwritten when real tokens reach those positions. Rejected K/V can never enter the
  prefix cache because only full blocks below `num_cached` are published.
- **Per-request draft length.** `Sequence.draft_len` doubles after a fully accepted draft
  (up to `num_draft_tokens`) and drops to `accepted + 1` after a rejection, so a request
  whose drafts keep failing stops paying for long verifications.
- **Cost model.** A verification span goes through the eager prefill pass, not the
  CUDA-graph decode pass, so on this engine a speculative step is *slower* than a plain
  decode step (M-004: ~11 ms vs ~4 ms at small batch). Speculation pays only when it
  emits enough tokens per step to cover that; the numbers are in M-009.

### D-030 Drafters

- **Prompt lookup** (`minfer/spec/ngram.py`): if the last `n` tokens (default 3) occurred
  earlier in the sequence, propose what followed them. An incremental per-sequence
  dictionary of n-gram → continuation position makes a lookup O(1) amortized; no model, no
  extra forward pass.
- **Draft model** (`minfer/spec/draft_model.py`): Qwen2.5-0.5B with its own block pool,
  page tables and radix tree — the target's KV machinery instantiated a second time, so
  the draft model also reuses shared prefixes. Per step it catches up on tokens the
  target confirmed (one ragged prefill pass, whose last row is the first draft token), then
  runs one CUDA-graph decode pass per further draft token. K/V it wrote for drafts that
  were then rejected is discarded by comparing its speculative tail with the confirmed
  tokens.
- Seeded sampling stays reproducible with speculation but is not the *same* stream as
  without it (speculative sampling consumes a different number of uniforms per token).

---

## Phase 10 — Tensor parallelism (implemented; multi-GPU run not possible here)

### D-031 Megatron-style shards, driver/worker execution, head-sharded KV

- **Shards.** A rank holds `1/N` of the attention heads (query and KV heads together, so
  grouped-query attention stays local to a rank) and `1/N` of the MLP's hidden units.
  q/k/v and gate/up projections are column-parallel (no communication); o_proj and
  down_proj are row-parallel: each rank computes a partial sum of the full-width output
  and an all-reduce adds them. That is two all-reduces of one hidden-state vector per
  token per layer. Embeddings, norms and the unembedding are replicated.
  `shard_state_dict` slices a full checkpoint; `Qwen2Model(config, shard)` is the same
  module code with narrower projections and a `reduce` hook.
- **Execution.** Rank 0 runs the unchanged engine (scheduler, block manager, prefix cache,
  sampling). Before each pass its attention backend broadcasts the batch layout to the
  workers over pipes; every rank then runs the same stack on its shard and they meet in
  the all-reduces (`torch.distributed`: NCCL between GPUs, gloo between CPU processes).
- **Sharded KV cache.** Block ids are allocated only by the driver, so one block id
  addresses the same positions in every rank's store; each store holds only its rank's KV
  heads. Per-rank KV memory is `1/N` of the single-device figure, with no coordination
  beyond the batch layout.
- **Limits.** `num_kv_heads` must divide by `N` (Qwen2.5-1.5B: N = 2; 14B: N ∈ {2, 4, 8}).
  All-reduces cannot be captured in a CUDA graph, so TP runs the eager decode path; and
  draft-model speculation is not wired through TP.

### M-008 Tensor parallelism: what could be verified on one GPU

- `tests/test_tensor_parallel.py`: per-layer shard algebra (Σ over ranks of each
  row-parallel output equals the unsharded layer, to 1e-5) and a two-process gloo group
  whose engine output is identical to the single-process cache-free oracle — also with
  the prefix cache, chunked prefill, a pool small enough to force preemption, and
  prompt-lookup speculation.
- `bench/tensor_parallel_check.py` → `bench/results/tensor_parallel_check.json`: 16
  requests / 768 generated tokens identical between one process and two; KV store bytes
  per rank exactly half; and the communication the design implies for the target models
  (Qwen2.5-14B: 96 all-reduces and 0.98 MB all-reduced per token at bf16; KV 196.6 KB per
  token on one device, 98.3 KB per rank at N = 2).
- **Not measured: any speed-up.** The plan's target (Qwen2.5-14B on 2×H100 over NVLink)
  needs a second GPU. The CPU timings in the results file are labelled as not a benchmark:
  two CPU processes on one host say nothing about two H100s. The multi-GPU benchmark is an
  open TODO in PROGRESS.md.

### M-009 Agent workload and ShareGPT: minfer vs vLLM vs SGLang (Phase 6 acceptance)

Script: `bench/run_matrix.py` → `bench/results/matrix/{minfer,vllm,sglang}.json` (plus
vLLM's raw result files for ShareGPT); charts by `bench/plot.py` in `bench/plots/`; the
README tables are generated from the same JSON by `bench/readme_tables.py`.
Qwen2.5-1.5B-Instruct, bf16, greedy, one H100, one server at a time, all engines at
defaults (D-028). Every run of every engine completed with **0 failed requests**, and all
three replayed traces with identical SHA-256 per rate.

This is the state of the engine *at the end of Phase 6*, before the profiling and
optimization phase; the matrix is re-run on the final code in Phase 12.

**Agent workload, rates 1–16 req/s (ranges over the five rates):**

| | minfer | vLLM | SGLang |
|---|---|---|---|
| TTFT p50 (ms) | 38.5–45.2 | 46.1–64.5 | 45.6–49.3 |
| TTFT p99 (ms) | 66.7–82.5 | 101.4–135.0 | 83.4–133.2 |
| TPOT p50 (ms) | 4.09–5.78 | 2.22–2.37 | 2.25–2.58 |
| TPOT p99 (ms) | 4.87–8.00 | 2.73–3.22 | 2.62–3.49 |
| E2E p50 (ms) | 294.4–393.7 | 192.6–210.4 | 182.1–207.5 |
| prefix hit rate | 93.0%–93.6% | 92.6%–93.2% | 93.1%–93.7% |

**Agent workload, "max" (64 sessions at once):**

| | minfer | vLLM | SGLang |
|---|---|---|---|
| output tok/s | 2654 | 5417 | 5017 |
| TTFT p50 / p99 (ms) | 79 / 300 | 109 / 747 | 134 / 471 |
| TPOT p50 / p99 (ms) | 18.8 / 21.0 | 7.2 / 8.4 | 8.0 / 10.4 |

**ShareGPT (control), measured by `vllm bench serve`:** TTFT p50 at 1–16 req/s is
21.1–23.6 ms (minfer), 22.8–27.5 (vLLM), 19.8–28.3 (SGLang); TPOT p50 is
3.99–5.51, 2.15–2.27 and 2.18–2.65 ms; saturated throughput (1,000 requests at once) is
13678, 20669 and 13484 output tok/s.

**Reading it honestly:**

- **Where minfer is ahead: time to first token on the agent workload**, at every rate,
  p50 and p99. All three engines reuse ~93% of prompt tokens (their prefix caches are
  equally effective on this workload), so the difference is not cache hit rate; it is the
  cost of getting a mostly-cached request from the socket to its first token.
- **Where minfer is behind: time per output token, by about 2×**, at every rate and on
  both workloads, and therefore end-to-end latency (agent turns emit 33–240 tokens, so
  TPOT dominates E2E) and saturated throughput (about half of vLLM's and SGLang's on the
  agent workload; two thirds of vLLM's on ShareGPT).
- **Why.** A single-token decode step in minfer replays a CUDA graph of roughly 1,200
  kernels (every PyTorch operator of every layer is its own kernel: 8 for each RMSNorm,
  ~12 for RoPE, …), and each costs a few microseconds whatever its size: 3.7 ms at batch 1
  (M-004). vLLM (torch.compile + custom ops) and SGLang (fused norm/activation kernels)
  launch a fraction of that. Under load the eager prefill pass adds its ~11 ms dispatch
  floor to every step that admits a request, which is what drives TPOT to 19 ms at "max".
- Both are engineering gaps in the forward pass, not in the design the project is about
  (the cache, the scheduler); they are the subject of Phase 11.
- SGLang's ShareGPT run at 1 req/s has a TTFT p99 of 245 ms against a p50 of 20 ms:
  with 120 requests that is one or two slow requests, reported as measured.
- One caveat on the minfer runs: a 28 GB model download (B-004 below) overlapped the end of
  minfer's agent "max" run and the start of its ShareGPT runs. The "max" result matches an
  earlier undisturbed run of the same trace (2,627 vs 2,654 tok/s, TTFT p50 79.0 vs
  78.8 ms), so it was left in; the whole matrix is re-run in Phase 12 regardless.

### B-004 A benchmark helper downloaded a 28 GB checkpoint mid-benchmark (process bug)

`bench/tensor_parallel_check.py` resolved the Qwen2.5-14B model directory just to read its
`config.json`, which fetched the weights too — while the matrix was measuring minfer. Fixed
by fetching only `config.json`. Rule adopted: nothing that touches the network or the disk
at scale runs while a benchmark holds the GPU.

### M-010 Speculative decoding on the GPU (Phase 7 acceptance)

Script: `bench/speculation.py` → `bench/results/speculation.json`. Engine-level, greedy,
requests stop at EOS; agent = 32 sessions closed-loop (356 requests), ShareGPT = 256
single-turn requests submitted at once; `num_draft_tokens = 4`, n-gram size 3, draft
model Qwen2.5-0.5B. "tok/step" is tokens emitted per engine step over all sequences.

| workload | speculation | out tok/s | vs off | TPOT p50 (ms) | TPOT p99 (ms) | tok/step | acceptance | outputs identical to off |
|---|---|---|---|---|---|---|---|---|
| agent | off | 1869 | — | 11.04 | 15.3 | 16.1 | — | — |
| agent | ngram | 1730 | 0.93× | 7.58 | 21.7 | 31.9 | 75.7% | 231/356 |
| agent | draft model | 1346 | 0.72× | 12.81 | 28.6 | 56.5 | 74.2% | 232/356 |
| sharegpt | off | 23366 | — | 7.82 | 21.5 | 188.7 | — | — |
| sharegpt | ngram | 10180 | 0.44× | 17.70 | 24.7 | 189.1 | 56.7% | 93/256 |
| sharegpt | draft model | 8659 | 0.37× | 16.35 | 27.6 | 362.2 | 66.0% | 76/256 |

**Speculation works and, on this engine today, makes things slower.**

- *It works:* 57–76% of drafted tokens are accepted, and steps emit about twice as many
  tokens (agent: 16 → 32 per step with prompt lookup, 57 with the draft model).
- *It is slower:* throughput drops to 0.93× (prompt lookup) and 0.72× (draft model) on the
  agent workload and to 0.44× / 0.37× on ShareGPT. The reason is the cost model of D-029:
  a verification span runs in the eager prefill pass (~11 ms of dispatch per pass, M-004)
  while the tokens it replaces would have come from ~4–7 ms CUDA-graph decode steps; the
  draft model adds its own passes on top. With 256 concurrent ShareGPT requests the plain
  decode graph already produces 188 tokens per step, so there is little left to win and a
  full eager pass to lose.
- The one improvement is median TPOT on the agent workload with prompt lookup
  (11.0 → 7.6 ms): agent outputs do copy from their prompts, as the design assumed. p99
  gets worse (15 → 22 ms).
- **Kept, off by default, reported as measured.** The fix is not in the speculation logic
  but in the pass it rides on: a CUDA-graph capture of the verification/prefill pass
  (Phase 11) changes the denominator. The ablation in Phase 12 re-measures it.

**Greedy outputs with speculation on vs off.** Exactly identical on the CPU float32 model
for any draft quality (`tests/test_speculation.py`: scripted drafts that are all right,
all wrong and mixed; prompt lookup; a draft model equal to and unrelated to the target;
under chunked prefill, memory pressure, and with the prefix cache). On the GPU in bf16 the
right-hand column applies: verification evaluates the same positions with the prefill
kernel instead of the decode kernel, and near-ties flip as in M-002. At the first differing
token of each diverging output, float32 separates the two tokens by at most 0.97 logits
(per setting: 0.43, 0.52, 0.97, 0.32); in 3–7 outputs per setting the pair is not float32's exact
top-2 (three-way near-ties). ShareGPT diverges more often than the agent workload because
its outputs are up to 256 tokens long: more positions, more chances for one flip, and
everything after a flip differs. The GPU test asserts a float32 gap below 1.5 logits at
every first divergence (single bf16 logits are up to ~1.2 off, M-002).

**Bug found on the way.** FlashInfer's plain decode kernel is specialized for GQA group
sizes 1, 2, 3, 4, 6, 8, 16; Qwen2.5-0.5B has 14 heads over 2 KV heads (group 7) and failed
with "Unsupported group_size: 7". The backend now selects FlashInfer's tensor-core decode
path for other group sizes (`needs_tensor_cores`).

---

## Phase 8 — Custom Triton paged-attention decode kernel

### D-032 Kernel design

`minfer/attention/triton_decode.py`, a drop-in alternative to FlashInfer's decode kernel
(`decode_kernel="triton"` / `--decode-kernel triton`); prefill stays on FlashInfer.

- **Grid `(batch, kv_heads, splits)`; GQA handled inside a program.** The G = 6 query
  heads that share a KV head are stacked into one 16-row matrix (padded: `tl.dot` needs
  ≥ 16 rows), so each key/value block is read once and scored against all six queries
  with one matmul. A program walks its part of the sequence four blocks (64 tokens) per
  iteration: block ids from the block table, a gather of K and V, scores, and an online
  softmax (running max / sum / accumulator), so nothing the size of the context is
  materialized.
- **Reads block tables directly.** The kernel takes a padded `[batch, max_pages]` table
  and the cached lengths. There is no host-side planning step: FlashInfer needs a
  `plan()` call per engine step (~0.07 ms measured), this needs two small buffer copies.
- **Split-KV.** With few sequences a sequential walk leaves the GPU idle on long contexts
  (first version, batch 1 × 16k context: 0.30 ms per layer vs FlashInfer's 0.02). Each
  sequence is cut into `SPLITS` chunks run by separate programs that store unnormalized
  partials; a tiny second kernel merges them with the same recurrence. `SPLITS` depends
  only on the batch size (`256 / (batch · kv_heads)`, capped at 16), so the grid is static
  per batch bucket and the kernel captures cleanly in the decode CUDA graphs.
- **A measurement that changed the design.** With `SPLITS` uncapped (128 at batch 1), the
  merge kernel — two programs looping serially over 128 chunks — cost 30 µs per layer by
  itself, at every context length. Capping at 16 brought batch 1 from 0.37× to 1.6× of
  FlashInfer at short contexts.

### M-011 Triton kernel vs FlashInfer (Phase 8 acceptance)

**Numerics** (`bench/triton_kernel.py`, `tests/test_engine_gpu.py`). Random bf16 queries and
caches with shuffled block tables, 19 (batch, context) points: the largest absolute
difference between the two kernels is 0.0039 on outputs of mean magnitude 0.01–0.11, and
each kernel is equally far from a float32 PyTorch reference (max abs error: Triton
0.0038, FlashInfer 0.0038). **Documented tolerance, asserted in the tests: max abs
difference < 0.02 from FlashInfer and from float32**, i.e. bf16 rounding of O(1)
quantities; ragged batches (lengths 1 to 2,000 in one batch, including shorter than a
chunk) are checked row by row against float32. In the engine, the kernel passes the same
teacher-forced check against the float32 model as FlashInfer (M-002), and graph replay is
bit-identical to the eager call.

**Microbenchmark** (`bench/results/triton_kernel.json`, plot `bench/plots/triton_kernel.png`):
GPU time per layer in µs when 28 calls are replayed from a CUDA graph — the way the engine
runs decode — as *Triton / FlashInfer (speed-up)*:

| batch | ctx 128 | ctx 1,024 | ctx 4,096 | ctx 16,384 |
|---|---|---|---|---|
| 1 | 7.0 / 11.1 (1.59×) | 7.3 / 12.0 (1.65×) | 10.8 / 16.0 (1.48×) | 25.2 / 23.2 (0.92×) |
| 8 | 8.0 / 11.3 (1.42×) | 9.2 / 15.9 (1.74×) | 14.8 / 27.3 (1.85×) | 55.5 / 75.1 (1.35×) |
| 32 | 5.5 / 15.3 (2.76×) | 11.6 / 27.3 (2.35×) | 52.7 / 74.5 (1.41×) | 187.3 / 288.5 (1.54×) |
| 64 | 6.0 / 15.9 (2.66×) | 29.3 / 42.1 (1.44×) | 96.6 / 150.2 (1.55×) | 366.6 / 575.1 (1.57×) |
| 256 | 8.3 / 28.7 (3.46×) | 92.9 / 154.9 (1.67×) | 356.9 / 582.0 (1.63×) | — |

The Triton kernel is faster at 18 of 19 points (1.35–3.46×) and slower at one (batch 1 with a
16k-token context, 0.92×: sixteen chunks of 1,024 tokens each are still walked
serially, while FlashInfer spreads one sequence over more of the GPU). The results file
also has timings of the kernels called from Python; below ~0.05 ms those measure the
launchers, not the kernels, and are not used here.

**In the engine** (`bench/decode_latency.py`, same file as M-004, re-run for this phase):
whole decode step, p50 ms, CUDA graphs on.

| batch | ctx 256: FlashInfer | Triton | speedup | ctx 2048: FlashInfer | Triton | speedup |
|---|---|---|---|---|---|---|
| 1 | 3.73 | 3.53 | 1.06× | 3.83 | 3.61 | 1.06× |
| 2 | 3.93 | 3.72 | 1.06× | 4.04 | 3.82 | 1.06× |
| 4 | 4.00 | 3.82 | 1.05× | 4.11 | 3.94 | 1.04× |
| 8 | 4.21 | 3.97 | 1.06× | 4.44 | 4.18 | 1.06× |
| 16 | 4.33 | 4.07 | 1.07× | 4.96 | 4.46 | 1.11× |
| 32 | 4.65 | 4.29 | 1.09× | 5.50 | 4.96 | 1.11× |
| 64 | 5.10 | 4.65 | 1.10× | 6.93 | 5.99 | 1.16× |
| 128 | 5.92 | 5.34 | 1.11× | 9.76 | 7.93 | 1.23× |
| 256 | 7.51 | 6.68 | 1.12× | 15.75 | 12.29 | 1.28× |

A 1.04–1.28× faster decode step: attention is one of ~45 kernels per layer, so a much faster
attention kernel moves the step by single digits at small batch and by a quarter at
batch 256 with long contexts, where attention dominates. (The M-004 table predates this
re-run; the numbers in `decode_latency.json` are the current ones and agree with it to
within 3%.)

---

## Phase 9 — FP8

### D-033 FP8 linear layers: static weight scale, dynamic activation scale

`minfer/quant/fp8.py`, `--weight-dtype fp8`. Every linear layer of the transformer blocks
(7 per layer; the tied unembedding stays bf16) holds its weight in E4M3 with one scale per
tensor and quantizes its input on the fly with a scale from the batch's own largest
magnitude; the product runs on the H100's FP8 tensor cores through `torch._scaled_mm`
with a bf16 result. Layers that read the same input (q/k/v; gate/up) share one activation
quantization. No calibration data is needed. One tensor-wide scale suffices because E4M3
is a floating format: relative precision holds over four decades below the scale's
maximum (unit-tested with a 2,500× outlier dimension).

### D-034 FP8 KV cache is mixed precision, chosen by calibration

`--kv-cache-dtype fp8`. Keys and values are divided by a calibrated per-layer scale
(largest observed magnitude × 1.25 → 448) and stored as E4M3; FlashInfer's FP8 kernels
multiply the scales back in. **Layers whose keys FP8 would break keep bf16**, in a second
tensor; a FlashInfer wrapper per storage dtype is planned each pass. Which layers those
are is measured, not configured: calibration (eight short passages through a scratch
bf16 store, before the real store is sized) records for each layer the mean absolute
change of the attention logits `q·k/√d` when its keys are rounded to E4M3, and layers
above 1.0 stay bf16. For Qwen2.5-1.5B that is exactly layer 0.

### B-005 An all-FP8 KV cache destroyed the model's output (fixed by D-034)

- **Symptom.** First GSM8K run with the FP8 KV cache: accuracy **1%** (bf16: 64%), 143 of
  200 answers running to the token limit, text degenerating into repetition ("is is a …
  to to to to"). FP8 *weights* alone were fine (70%).
- **Not the kernel.** A single-layer reproduction (random data, FlashInfer FP8 decode and
  prefill with `k_scale`/`v_scale`) matched a float32 computation on the dequantized
  values to 4e-3. Not the CUDA graphs either: the eager path at batch 1 was equally broken
  (teacher-forced mean abs logit error 3.66, against 0.05 for bf16).
- **Root cause.** Measured layer by layer on real keys: layer 0's keys have components of
  magnitude ~320 against a median of 1.4, and its queries reach ~100 in the same
  dimensions. E4M3's rounding error is *relative* (≤ 1/16), so those components are off by
  up to ~20 in absolute terms, which the large queries turn into a mean attention-logit
  error of **13–15** (max 465) in layer 0. Every other layer sits at 0.05–0.25. No choice
  of scale helps: scaling moves the exponent, not the 3-bit mantissa.
- **Fix.** Keep the offending layer in bf16 (D-034). Simulated before implementing
  (float32 attention with keys/values rounded per layer): all layers FP8 → logit error
  1.53; layer 0 in bf16 → 0.17; layers 0–1 → 0.16; values-only FP8 → 0.07. In the engine
  after the fix: 0.19, GSM8K 69%.
- **Cost.** One layer of 28 at 2 bytes: a token's KV is 14.5 KB instead of the 14.0 KB of
  an all-FP8 cache (and 28.0 KB in bf16), so capacity grows 1.92× rather than 2×.

### M-012 FP8: accuracy, capacity, speed (Phase 9 acceptance)

Script: `bench/gsm8k.py` → `bench/results/gsm8k_fp8.json`. First 200 problems of the GSM8K
test split, zero-shot chat prompt, greedy, max 512 tokens, all 200 submitted at once.
"Same answer" counts problems whose final answer equals the bf16 run's. Decode step is
p50 ms at batch 1 / 64 / 256 with 512-token contexts.

| setting | GSM8K correct | same answer as bf16 | KV per token | KV capacity (tokens) | weights | decode step (ms) | eval out tok/s |
|---|---|---|---|---|---|---|---|
| bf16 (baseline) | 128/200 (64.0%) | 200/200 | 28.0 KB | 2.55 M (1.00×) | 2.89 GB | 3.76 / 5.41 / 8.60 | 18233 |
| FP8 KV cache | 138/200 (69.0%) | 143/200 | 14.5 KB | 4.88 M (1.92×) | 2.89 GB | 4.15 / 6.03 / 10.79 | 15732 |
| FP8 weights | 140/200 (70.0%) | 142/200 | 28.0 KB | 2.49 M (0.98×) | 1.67 GB | 4.65 / 7.50 / 10.76 | 13907 |
| FP8 weights + KV | 142/200 (71.0%) | 138/200 | 14.5 KB | 4.79 M (1.88×) | 1.67 GB | 5.02 / 8.09 / 12.84 | 12302 |

- **Accuracy: no measurable loss.** 64.0% (bf16) vs 69.0% / 70.0% / 71.0%. With 200 problems the
  standard error of each figure is ±3.3 points, so the FP8 settings are indistinguishable
  from bf16 and from each other; the differences (in FP8's favour) are noise, not a
  benefit. About 70% of final answers are identical to the bf16 run's; the rest are
  different reasoning paths after a near-tie flip, right about as often as wrong.
- **KV capacity: 1.92×** (2.55 M → 4.88 M tokens in the same memory budget).
- **Weights: 2.89 GB → 1.67 GB** (embeddings and the unembedding stay bf16).
- **Throughput: FP8 is slower, not faster.** Decode steps take +10% to +26% longer with the FP8
  KV cache, +24% to +39% with FP8 weights, and +33% to +49% with both. This engine's
  decode step is bound by the number of kernels it launches (M-009), and FP8 adds kernels:
  a cast and a divide for every K/V write, four activation quantizations per layer, and a
  second attention plan for the bf16 layer. The FP8 matmuls themselves are not the
  bottleneck for a 1.5B model at these batch sizes, so halving the bytes they read buys
  nothing here. A larger model, or a decode path with far fewer kernels, would change the
  balance; neither was measured.
- **What FP8 is good for in this engine:** memory. Twice the KV in the same GPU means
  twice the cached prefixes before eviction and twice the concurrent context — which is
  the resource the agent workload consumes — at a 10–50% cost in per-token latency.
  It is therefore off by default.

---

## Phase 11 — Profiling + optimization

### M-013 Profile: the decode step is bound by kernel count

Script: `bench/profile_engine.py` (torch.profiler, CPU + CUDA activities) →
`bench/results/profiles/before/` and `after/`: a Chrome trace per scenario plus
`summary.json`. "Before" is the engine as benchmarked in Phase 6 (`--no-fused-kernels
--decode-kernel flashinfer`, no prefill graphs).

- A batch-1 decode step launches **1219 CUDA kernels** and the GPU is busy for 3.63 ms of
  them — essentially the whole step. The matmuls (cuBLAS `nvjet_*`) and attention account
  for under half; the rest is several hundred element-wise kernels a few microseconds
  each: the eight operators of every RMSNorm, the dozen of every RoPE application, casts,
  residual adds.
- The count is the same at batch 64 (1219): the step's cost is per *kernel*, not per
  token, until the batch is large. That is the 2× TPOT gap to vLLM and SGLang (M-009),
  which both fuse these stages.
- The eager prefill pass launches the same kernels from Python: ~11 ms of dispatch for a
  pass whether it holds 16 tokens or 1,024 (M-004, and `prefill_latency.json` below).

### D-035 Fix 1: merged projections and fused Triton kernels (`Qwen2Model.fuse`)

- q/k/v projections merged into one matmul, gate/up into another (weights concatenated at
  load; the originals are released).
- Three Triton kernels (`minfer/model/fused.py`): **add + RMSNorm** (residual add and the
  next norm in one launch, float32 statistics, weight applied after the cast, as the
  reference does), **RoPE over the merged projection** (splits q/k/v and rotates q and k,
  reading the cos/sin tables by position), and **SiLU·mul** over the merged gate/up
  output.
- A layer is now 3 matmuls + 4 fused kernels + the K/V write and attention:
  398 kernels per step instead of 1219.
- **Numerics.** In bf16 the norm and activation kernels are bit-identical to the PyTorch
  modules (they round where the modules round); the rotation rounds once instead of three
  times and differs by rounding only. In float32 everything matches to ~1e-6, and the
  whole fused model matches the unfused one to 5e-5 in logits
  (`test_fused_kernels_match_the_reference_modules`, `test_fused_model_matches_...`).
  The reference (unfused) path stays as the CPU / float32 / tensor-parallel / FP8-weights
  implementation and as the oracle.
- On by default on CUDA (`--no-fused-kernels` to disable).

### D-036 Fix 2: Triton decode attention by default

`decode_kernel="auto"` selects the Phase 8 kernel (M-011) unless the KV cache is FP8.

### D-037 Fix 3: CUDA graphs for small prefill passes

`minfer/attention/prefill_graphs.py`: prefill passes are captured at bucketed
`(spans, tokens)` shapes — spans ∈ {2, 4, 8, 16, 32}, tokens ∈ {32 … 1,024} in steps of
≤ 25% — and replayed with FlashInfer's prefill wrapper in graph mode. A batch is padded to
its bucket with spans that live in the reserved block 0; logits are computed only for the
last row of each completing span. 55 graphs, captured in ~2 s at start-up.

- **Why only up to 1,024 tokens.** An eager pass overlaps its Python dispatch with GPU
  execution, so it costs max(dispatch, compute). A graph removes the dispatch but pays for
  the padding. First version (buckets to 4,096, powers of two): a 1,024-token pass took
  8.8 ms from the graph against 7.4 ms eagerly, and 2,048 tokens 16.2 vs 12.3 ms. Passes
  above 1,024 tokens now run eagerly.
- Speculative verification spans are not captured (they need logits for every row);
  speculation therefore still pays the eager pass (M-010).

### M-014 Before / after (Phase 11 acceptance)

**Profile** (`bench/results/profiles/{before,after}/summary.json`):

| scenario | kernels/step before | after | GPU kernel ms/step before | after |
|---|---|---|---|---|
| decode graph batch1 | 1219 | 398 | 3.63 | 2.02 |
| decode graph batch64 | 1219 | 399 | 5.11 | 2.59 |
| decode eager batch1 | 1219 | 398 | 3.75 | 2.04 |
| prefill 2048 tokens | 1248 | 399 | 15.56 | 11.03 |

**Decode step, p50 ms** (`bench/decode_latency.py` → `bench/results/decode_latency.json`,
which now measures the four stages in one run; plot `bench/plots/decode_latency.png`):

| batch | context | eager | + CUDA graphs (Phase 3) | + Triton decode (Phase 8) | + fused kernels (default) | Phase 3 → default |
|---|---|---|---|---|---|---|
| 1 | 256 | 11.35 | 3.74 | 3.53 | 2.18 | 1.72× |
| 8 | 256 | 11.42 | 4.21 | 3.96 | 2.33 | 1.81× |
| 64 | 256 | 11.66 | 5.10 | 4.65 | 2.81 | 1.82× |
| 256 | 256 | 12.13 | 7.56 | 6.69 | 4.55 | 1.66× |
| 1 | 2048 | 11.32 | 3.84 | 3.62 | 2.27 | 1.69× |
| 8 | 2048 | 11.42 | 4.45 | 4.17 | 2.52 | 1.77× |
| 64 | 2048 | 11.86 | 6.96 | 6.00 | 4.08 | 1.71× |
| 256 | 2048 | 17.73 | 15.89 | 12.35 | 10.18 | 1.56× |

The default engine's decode step is **1.56–1.85× faster than the Phase 3 engine** at every
batch size and context measured, and 5.2× faster than eager at batch 1.

**One prefill pass, p50 ms** (`bench/prefill_latency.py` → `bench/results/prefill_latency.json`):

| prompt tokens | eager, unfused (before) | eager, fused | default (graph ≤ 1,024 tokens) |
|---|---|---|---|
| 16 | 10.83 | 6.89 | 2.38 |
| 64 | 11.26 | 6.96 | 2.59 |
| 256 | 11.08 | 6.98 | 3.50 |
| 512 | 10.81 | 6.99 | 4.47 |
| 1024 | 11.43 | 7.23 | 7.11 |
| 2048 | 17.77 | 12.36 | 12.35 |

**End to end**, the saturated agent workload driven in-process (64 sessions, the "max"
trace; development measurement with a scratch script, not a committed result): 2,600 →
4,300 output tok/s and TTFT p50 28 → 17 ms. The committed end-to-end numbers are the
Phase 12 re-run of the full matrix.

**What is left in the step.** After the fixes a saturated step is mostly GPU time (about
9 of 11.7 s in that run were spent waiting on the device). Remaining known costs, in
order: two passes per step when a step has both prefill and decode work (D-013) where
vLLM runs one; ~400 kernels where a compiled graph would have fewer (the K/V write and
the unmerged o_proj/down_proj tail); and ~0.2 ms of host work per step in the scheduler
and the radix tree.
