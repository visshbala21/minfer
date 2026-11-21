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

---

## Phase 12 — Final re-run, ablations

A note on reading this file: result files under `bench/results/` are regenerated by
`make bench`, so the numbers quoted in the entries above are the ones measured when each
entry was written; several of those files now hold newer numbers. This section records the
final run on the final code (`make test`: 272 passed; then `make bench`, which produced
every number in the README) and what changed.

### M-015 The three-engine matrix on the final code

`bench/results/matrix/*.json`, same method as M-009, all engines re-measured in one
`make bench-agent` run; 0 failed requests.

| agent workload, 1–16 req/s | minfer (Phase 6) | minfer (final) | vLLM | SGLang |
|---|---|---|---|---|
| TTFT p50 (ms) | 38.5–45.2 | 38.3–41.1 | 59.3–62.9 | 43.7–46.6 |
| TTFT p99 (ms) | 66.7–82.5 | 68.5–82.3 | 95.4–122.2 | 77.5–114.4 |
| TPOT p50 (ms) | 4.09–5.78 | 2.54–2.99 | 2.25–2.40 | 2.25–2.57 |
| E2E p50 (ms) | 294–394 | 191–225 | 196–213 | 177–208 |
| "max": output tok/s | 2654 | 4219 | 5106 | 5049 |
| "max": TTFT p50 / p99 (ms) | 79 / 300 | 83 / 241 | 144 / 570 | 119 / 460 |
| "max": TPOT p50 (ms) | 18.83 | 10.71 | 7.40 | 8.08 |

(The Phase 6 column is M-009's minfer measurement, kept for comparison. vLLM and SGLang,
whose code did not change, moved by a few percent between the two runs at 1–16 req/s and
by more at saturation — vLLM's "max" TTFT p50 was 109 ms in M-009 and 144 ms here — which
is the run-to-run noise of this setup and the reason single-digit-percent differences in
these tables should not be read as rankings.)

- **TTFT: minfer leads at every rate**, median and p99, and at saturation.
- **TPOT: the 2× gap of Phase 6 is now 1.06–1.24× of vLLM** at 1–16 req/s and
  1.45× at "max". Still behind; the reasons are in M-014's last paragraph.
- **Saturated throughput: 83% of vLLM's** (was 49%).
- **ShareGPT** (vLLM's client): TTFT p50 9.9–11.8 ms vs 21.5–29.3 (vLLM) and
  19.3–28.8 (SGLang); TPOT p50 2.24–2.55 vs 2.16–2.28 and 2.18–2.63 ms. With 1,000
  requests at once: 24607 / 22248 / 16102 output tok/s and TPOT p50 6.7 / 24.4 / 35.3 ms,
  but median TTFT 2764 / 1505 / 2233 ms: minfer runs at most 256 sequences at once and
  queues the rest, trading first-token latency for per-token latency. That is a default,
  not a tuned choice.

### M-016 Ablations on the agent workload

`bench/ablations.py` → `bench/results/ablations.json`: a fresh `minfer serve` per
configuration, the same two traces as the matrix (16 req/s and "max").

| configuration | 16 req/s: tok/s | TTFT p50 / p99 | TPOT p50 / p99 | max: tok/s | TTFT p50 / p99 | TPOT p50 / p99 | hit rate (max) |
|---|---|---|---|---|---|---|---|
| full system (defaults) | 800 | 37 / 86 | 2.98 / 4.24 | 4208 | 83 / 234 | 11.08 / 12.83 | 92.8% |
| − prefix cache | 762 | 109 / 438 | 6.21 / 17.72 | 1204 | 1715 / 2486 | 18.39 / 20.32 | — |
| − cache-aware admission (FCFS) | 797 | 38 / 79 | 2.96 / 4.24 | 4298 | 76 / 147 | 10.97 / 12.18 | 92.8% |
| − CUDA graphs (decode and prefill) | 752 | 47 / 338 | 7.90 / 17.48 | 3361 | 73 / 270 | 13.07 / 14.64 | 92.8% |
| − prefill graphs only | 798 | 40 / 86 | 3.13 / 4.64 | 4042 | 75 / 270 | 11.64 / 13.78 | 92.8% |
| − chunked prefill | 800 | 37 / 80 | 2.98 / 4.25 | 4257 | 76 / 223 | 11.08 / 12.64 | 92.8% |
| − fused kernels | 780 | 43 / 89 | 5.05 / 6.71 | 3207 | 78 / 266 | 14.81 / 16.85 | 92.8% |
| − Triton decode (FlashInfer decode) | 798 | 38 / 77 | 3.21 / 4.45 | 3485 | 69 / 241 | 14.03 / 16.93 | 92.8% |
| + speculation: prompt lookup | 779 | 43 / 87 | 5.55 / 10.49 | 3297 | 76 / 241 | 13.19 / 21.90 | 92.8% |
| + speculation: draft model | 731 | 75 / 124 | 11.72 / 18.97 | 1852 | 141 / 424 | 22.29 / 39.75 | 92.8% |
| + FP8 KV cache | 791 | 43 / 87 | 4.19 / 6.00 | 2646 | 78 / 264 | 19.46 / 23.32 | 92.8% |
| + FP8 weights | 765 | 49 / 97 | 6.88 / 11.27 | 2572 | 80 / 254 | 18.91 / 21.66 | 92.8% |
| 16 running sequences max | 800 | 37 / 81 | 2.97 / 4.25 | 2848 | 1081 / 1649 | 5.17 / 7.80 | 92.8% |
| 16 running sequences max, FCFS | 799 | 40 / 79 | 2.96 / 4.19 | 2942 | 970 / 1204 | 5.08 / 7.21 | 92.8% |
| KV cache 8% of GPU memory | 800 | 39 / 85 | 3.04 / 4.23 | 4054 | 81 / 252 | 11.21 / 15.18 | 92.1% |
| KV cache 8%, FCFS | 799 | 38 / 78 | 3.01 / 4.36 | 4128 | 74 / 228 | 11.42 / 14.69 | 92.3% |
| KV cache 8%, FP8 KV | 791 | 43 / 91 | 4.22 / 6.37 | 2679 | 76 / 293 | 19.09 / 21.12 | 92.8% |
| KV cache 8%, no prefix cache | 764 | 106 / 317 | 6.33 / 17.82 | 1201 | 1714 / 2512 | 18.39 / 20.25 | — |

What the table says, feature by feature:

- **Prefix cache — the feature the engine is built around.** Without it: TTFT p50
  37 → 109 ms at 16 req/s; at saturation throughput 4208 → 1204 tok/s
  (3.5×) and TTFT p50 83 → 1715 ms.
- **Fused kernels, CUDA graphs, Triton decode** each account for a visible share of TPOT
  and saturated throughput (rows 4–8); prefill graphs matter mostly at saturation.
- **Cache-aware admission: no benefit in any scenario measured.** FCFS is within noise at
  16 req/s and slightly better at saturation. With 256 running slots and a pool of 2.5 M
  tokens, requests are admitted on arrival, so there is nothing to reorder. In the
  scenario built to give it a queue (16 running sequences, "max") it is slightly *worse*
  than FCFS: TTFT p50 1081 vs 970 ms, p99 1649 vs 1204 ms, 2848 vs 2942 tok/s.
  Every waiting request on this workload already has a long cached prefix (the system
  prompt at least), so "longest cached prefix first" mostly reorders requests of similar
  cost and lengthens the tail for the ones it passes over. The feature stays (default on,
  `--no-cache-aware-scheduling` to disable) with this result attached; a workload that
  mixes cached and uncached requests under memory pressure is where it could matter, and
  none was measured.
- **Chunked prefill: neutral here.** Prefix hits leave only a few hundred new tokens per
  request, below the chunk size, so nothing gets chunked. Its effect on a workload with
  long uncached prompts is M-017.
- **Speculation: slower** (as M-010), in both forms.
- **FP8: slower per token, and its capacity did not pay for itself here.** With the KV
  cache cut to 8% of GPU memory, the bf16 engine starts evicting (hit rate at "max"
  92.1% vs 92.8%) and loses 4% of its throughput; FP8 KV restores the hit rate
  (92.8%) but its slower steps cost far more than the evictions did (2679 vs 4054
  tok/s). The prefix cache degrades gracefully under this pressure — a cache this small
  still keeps the shared system prompts — so the scenario was not harsh enough to make
  capacity the binding constraint.

### M-017 Micro-benchmarks re-measured on the final code

Numbers that moved enough to change a statement made earlier in this file:

- **Chunked prefill (revises M-005).** `bench/results/chunked_prefill.json`, short
  requests, ms:

  | chunk | ITL p99 | ITL max | TPOT p50 | TPOT p99 | long-prompt TTFT p50 |
  |---|---|---|---|---|---|
| off | 5.6 | 114 | 2.45 | 5.00 | 69 |
| 4096 | 25.0 | 41 | 2.45 | 5.31 | 82 |
| 2048 | 17.7 | 26 | 2.45 | 5.47 | 100 |
| 1024 | 12.8 | 26 | 2.51 | 5.89 | 121 |
| 512 | 11.9 | 14 | 2.58 | 9.29 | 208 |
| 256 | 10.8 | 14 | 3.04 | 9.54 | 462 |

  With the engine about twice as fast, the rare unchunked long prefill no longer reaches
  the 99th percentile of token gaps: **turning chunking off now gives the best ITL p99**,
  and chunking makes it worse by turning one long stall into many medium ones. What
  chunking still buys is the **worst-case stall** (114 ms unchunked vs 26 ms at 2,048).
  The Phase 3 acceptance ("p99 TPOT improvement") therefore held for the Phase 3 engine at
  the token level (M-005) and does **not** hold for the final engine at either level.
  The default stays at 2,048 as a bound on the worst stall; on the agent workload it is
  neutral (M-016).
- **Speculation (updates M-010).** The baseline got faster and speculation did not (its
  verification pass is still eager): agent 3199 tok/s → 2483 tok/s (0.78×, 75% accepted) with prompt lookup,
  1413 tok/s (0.44×, 74% accepted) with the draft model; ShareGPT 35777 tok/s → 14832 tok/s (0.41×, 56% accepted) and
  9809 tok/s (0.27×, 66% accepted).
- **FP8 (updates M-012).** GSM8K: bf16 69.0%, FP8 KV 68.5%, FP8 weights 70.5%, both
  71.0%. The bf16 figure itself moved from 64.0% to 69.0% between the unfused and the fused
  engine — the same model with different rounding — which is a direct measurement of how
  little a 5-point difference on 200 problems means. Decode step at batch 1 / 64 / 256:
  bf16 2.16 / 2.95 / 5.31 ms; FP8 KV 2.74 / 4.15 / 8.63; FP8 weights 4.42 / 6.92 / 9.62
  (the FP8-weight path cannot use the fused kernels, so its relative cost grew).
- **Equivalence (updates M-002, M-006).** Against vLLM: 11/20 and 10/20 outputs identical
  (vLLM vs itself: 9/20); teacher-forced 1985/2001 positions equal, mean abs logit
  error vs float32 0.047. Prefix cache on vs off: 46/64 identical, against a
  re-batching noise floor of 46/64.

### Phase 12 acceptance, honestly

"A fresh clone on a fresh H100 reproduces the headline numbers with `make bench`."
`make bench` is the command that produced every number in the README, run once end to end
on the development machine after `make test`. **A fresh clone on a second machine was not
tried** (a fresh clone on the same machine was, later: M-018) (there is no second machine, and no Modal credentials to run
`bench/modal_bench.py`), so reproduction elsewhere is untested; the environments are
pinned by lock files and built by `scripts/setup_envs.sh`.

### M-018 Fresh-clone reproduction on the development machine

Script: `bench/reproduction.py` → `bench/results/reproduction.json`; the rerun's own result
files are in `bench/results/reproduction_rerun/` (profiler traces omitted).

- **What was done.** `git clone` of commit 5227474 into a new directory; `HOME` pointed at
  an empty directory (so no FlashInfer, Triton, vLLM or SGLang JIT caches) and `HF_HOME` at
  an empty directory (weights and datasets downloaded again); `scripts/setup_envs.sh all`
  with the environments built in place (3 min with a warm uv cache); `make test`
  (272 passed in 21 min); `make bench` (2 h 49 min, exit status 0 — the README's "about two
  hours" was an underestimate and now says three).
- **What it is not.** The same physical H100, driver and system CUDA toolkit. The plan's
  acceptance test asks for a fresh H100; that still needs a second machine or Modal
  credentials.
- **Result, agent workload.** Every qualitative claim of the README holds in both runs
  (minfer lowest TTFT p50 and p99 at every rate and at saturation; higher TPOT p50 than
  vLLM at every rate; lower saturated throughput than both). minfer's rerun / original
  ratios: TTFT p50 0.94–1.01, TTFT p99 0.92–1.06, TPOT p50 1.00–1.04, saturated throughput
  1.00 (4219 → 4215 tok/s). The baselines moved by similar amounts (TTFT p50 0.93–1.10,
  TTFT p99 up to 1.14 for vLLM). Trace hashes are identical in both runs.
- **Ablations.** Saturated throughput of all 18 configurations is within 0.98–1.03 of the
  original run. One statement in M-016 does not survive: "with the KV cache cut to 8% the
  bf16 engine loses 4% of its throughput" (4208 → 4054) was run-to-run noise — in the rerun
  the small-cache configuration is at 4195 against 4163 for the full system.
- **What did not reproduce.**
  - SGLang, ShareGPT at 1 req/s: 119 of 120 requests completed; one connection was dropped
    (`ServerDisconnectedError` in vLLM's client), and the first request of that run took
    15 s. SGLang's server log shows nothing at that time; cause not determined. The
    original run had 0 failures everywhere.
  - ShareGPT "max" for the baselines: vLLM's median TTFT 1505 → 4006 ms and median TPOT
    24.4 → 13.4 ms; SGLang's throughput 16102 → 13589 tok/s. minfer: TTFT 2764 → 2596 ms,
    throughput 24607 → 25627 tok/s. With 1,000 requests arriving at once the outcome
    depends on how each server's admission happens to interleave with tokenization, and
    one run of it is not a stable measurement. The README says so.

---

## Phase 13 — Benchmark fairness audit

The question for this phase: read the benchmarks of Phases 6, 8 and 12 as a reviewer who
does not believe them. Two comparisons were audited — the three-engine matrix and the
Triton-vs-FlashInfer microbenchmark — and one of the two did not survive as published.

### M-019 The Triton-vs-FlashInfer comparison was against FlashInfer's slower configuration (revises M-011)

- **What was wrong.** `bench/triton_kernel.py` built FlashInfer's decode wrapper with its
  default `use_tensor_cores=False`. For grouped-query attention with a group size of 4 or
  more, SGLang selects FlashInfer's tensor-core decode path, and that path is the faster one
  for this head layout (12 query heads over 2 KV heads) at **every** batch size and context
  measured. M-011's "faster at 18 of 19 points (1.35–3.46×)" was therefore a comparison with
  a configuration nobody tuning FlashInfer would use. It was not a host-overhead artefact:
  M-011's headline numbers were already kernel-only (replayed from a CUDA graph, no
  `plan()`); the baseline itself was the problem.
- **Re-measured** (`bench/triton_kernel.py` → `bench/results/triton_kernel.json`), GPU time
  per layer in µs replayed from a CUDA graph (kernel only), then one decode step's worth of
  attention in ms of wall-clock time (host preparation + 28 replayed layers; end to end):

  | batch | context | FlashInfer plain | FlashInfer tensor cores | Triton | Triton vs plain | **Triton vs best** | step: FlashInfer (ms) | step: Triton (ms) | **end to end** |
  |---|---|---|---|---|---|---|---|---|---|
  | 1 | 128 | 11.1 | 6.6 | 7.0 | 1.58× | 0.94× | 0.22 | 0.21 | 1.04× |
  | 8 | 128 | 11.4 | 7.1 | 8.0 | 1.42× | 0.88× | 0.24 | 0.24 | 0.97× |
  | 32 | 128 | 15.5 | 10.7 | 5.6 | 2.79× | 1.92× | 0.34 | 0.17 | 1.95× |
  | 64 | 128 | 15.7 | 11.3 | 5.9 | 2.65× | 1.90× | 0.35 | 0.18 | 1.91× |
  | 256 | 128 | 29.8 | 17.6 | 8.4 | 3.54× | 2.09× | 0.53 | 0.25 | 2.09× |
  | 1 | 1,024 | 11.6 | 7.5 | 7.2 | 1.62× | 1.05× | 0.25 | 0.22 | 1.14× |
  | 8 | 1,024 | 15.7 | 11.6 | 9.0 | 1.75× | 1.29× | 0.36 | 0.27 | 1.33× |
  | 32 | 1,024 | 28.7 | 15.5 | 11.7 | 2.46× | 1.33× | 0.47 | 0.35 | 1.36× |
  | 64 | 1,024 | 43.7 | 33.5 | 29.1 | 1.50× | 1.15× | 0.98 | 0.83 | 1.17× |
  | 256 | 1,024 | 157.2 | 97.7 | 93.0 | 1.69× | 1.05× | 2.79 | 2.63 | 1.06× |
  | 1 | 4,096 | 15.5 | 11.2 | 10.9 | 1.43× | 1.03× | 0.35 | 0.32 | 1.10× |
  | 8 | 4,096 | 28.0 | 15.2 | 14.2 | 1.98× | 1.08× | 0.46 | 0.42 | 1.12× |
  | 32 | 4,096 | 78.2 | 54.9 | 52.0 | 1.50× | 1.06× | 1.58 | 1.48 | 1.07× |
  | 64 | 4,096 | 151.8 | 99.5 | 96.5 | 1.57× | 1.03× | 2.83 | 2.72 | 1.04× |
  | 256 | 4,096 | 586.9 | 359.6 | 357.7 | 1.64× | 1.01× | 10.13 | 10.02 | 1.01× |
  | 1 | 16,384 | 23.6 | 13.4 | 24.7 | 0.96× | 0.54× | 0.41 | 0.71 | 0.58× |
  | 8 | 16,384 | 79.2 | 55.1 | 54.9 | 1.44× | 1.00× | 1.59 | 1.56 | 1.02× |
  | 32 | 16,384 | 294.2 | 188.3 | 187.3 | 1.57× | 1.01× | 5.32 | 5.27 | 1.01× |
  | 64 | 16,384 | 581.3 | 364.3 | 366.8 | 1.58× | 0.99× | 10.26 | 10.30 | 1.00× |

- **What is true.** Against FlashInfer's best configuration the Triton kernel is
  **0.54–2.09×** kernel-only: more than 5% faster at 9 of 19 points, more than 5% slower at
  3, and within 5% at 7. It wins clearly only with short contexts and many sequences
  (≥ 32 sequences × 128 tokens: 1.9–2.1×) and moderately at 1,024-token contexts
  (1.05–1.33×); once the context reaches 4,096 tokens the two kernels are the same speed
  (0.99–1.08×), and at batch 1 with a 16k context the Triton kernel takes almost twice as
  long (0.54×). End to end the picture is the same (0.58–2.09×): host preparation is
  0.05–0.09 ms for `plan()` and 0.03–0.13 ms for the Triton path's table copy, small next
  to 28 kernel launches, so "no host planning" was never a material advantage either.
- **Numerics** are unaffected: the tensor-core path is as close to float32 as the plain one
  (max abs error 0.0038 over the 19 points).
- **Changes made.** (1) `needs_tensor_cores` now selects the tensor-core path for every
  group size ≥ 4, so minfer's own FlashInfer decode option (`--decode-kernel flashinfer`,
  and the FP8-KV path, which can only use FlashInfer) runs FlashInfer's faster kernel; the
  "− Triton decode" ablation row is re-measured on that basis in Phase 17. (2) The
  microbenchmark measures both FlashInfer configurations, takes the faster as the baseline,
  and reports kernel-only and end-to-end numbers separately; the README table and plot show
  all three kernels. (3) The Triton kernel stays the default decode kernel: at the batch
  sizes and contexts of the agent workload it is equal or somewhat faster, never much
  slower. Whether that default survives at 7B (group size 7) is re-checked in Phase 14.

### D-038 Every setting of the three servers, side by side

Collected from each server's start-up log (`bench/logs/matrix_*_server.log`, and one
SGLang start at `--log-level info`, which prints its resolved `server_args`) and from the
sources of vLLM 0.34.0 and SGLang 0.5.23 where the logs do not say.

| setting | minfer | vLLM 0.34.0 | SGLang 0.5.23 | same? |
|---|---|---|---|---|
| model, dtype, context | Qwen2.5-1.5B, bf16, 32,768 | same (`--dtype bfloat16`) | same (`--dtype bfloat16`) | yes |
| prefix caching | on: radix tree over 16-token blocks | on: SHA-256 hash chain over 16-token blocks | on: radix tree, page size 1 | all on; measured hit rate 92.5–93.3% for all three |
| chunked prefill | on: 2,048 prefill tokens per step while anything decodes | on: bounded by `max_num_batched_tokens` 8,192 | on: `chunked_prefill_size` 8,192 | all on; minfer's chunk is the smallest (costs it TTFT on long uncached prompts, M-017) |
| max running sequences | 256 | 1,024 (H100 default) | 4,096 (derived) | differs; irrelevant at ≤ 64 concurrent sessions, decisive for ShareGPT "max" (1,000 at once) — see M-015 |
| max batched tokens per step | 8,192 | 8,192 | `max_prefill_tokens` 16,384 | comparable |
| CUDA graphs | decode: 9 batch buckets to 256; prefill: 55 (spans, tokens) buckets to 1,024 tokens | decode and prefill: 51 sizes to 512 (`FULL_AND_PIECEWISE`) | decode: 36 batch sizes to 256; prefill: 58 token shapes to 8,192 | all on |
| model compilation | hand-written fused Triton kernels | `torch.compile` (inductor), optimization level 2 | none by default (`--enable-torch-compile` is opt-in and experimental) | each engine's default |
| attention kernels | Triton decode, FlashInfer prefill | FlashAttention 3 | FlashAttention 3 (`fa3`) | each engine's default |
| KV memory | 90% of memory free after weights: 2.55 M tokens | `gpu_memory_utilization` default: 2.64 M tokens | `mem_fraction_static` 0.83: 2.33 M tokens | comparable; none evicts on this workload |
| scheduling policy | cache-aware admission (no effect measured, M-016) | FCFS | FCFS (`lpm` is opt-in) | — |
| warm-up | one "max" agent trace (64 sessions, seed 10,000) before any measured run | same trace | same trace (plus its own built-in start-up warm-up) | yes |
| tokenizer and chat template | client renders the chat template; server tokenizes the text | same | same | yes: `prompt_tokens` totals are identical for all three at every rate |
| sampling | `temperature=0`, `ignore_eos`, `max_tokens` from the trace | same request | same request | yes: `completion_tokens` totals identical |
| streaming | SSE, one chunk per token | SSE, `stream_interval` 1 | SSE, `stream_interval` 1 | yes |
| access log | off | **on** (one uvicorn line per request) | off at `--log-level warning` | **no** → fixed below |
| processes | HTTP front end + engine | API server + engine core | HTTP server + tokenizer/detokenizer managers + scheduler | each engine's default |
| client | `bench/agent_workload.py`, one process, same machine | same | same | yes |

Two findings from the table alone:

1. **vLLM was the only server writing an access-log line per request.** Measured effect:
   none beyond noise (below). Disabled anyway, as a matter of parity.
2. **The three "defaults" are not equally tuned for this workload.** vLLM compiles the
   model by default; SGLang's compilation and its batch tokenizer are opt-in. "Every
   engine at its defaults" (D-028) was a defensible rule but not the strongest baseline, so
   the documented opt-in flags were measured (M-020).

### M-020 The baselines with their documented tuning flags

Script: `bench/fairness_audit.py` → `bench/results/fairness_audit.json`. A fresh server per
configuration, the matrix's warm-up, and the matrix's agent traces at 4 req/s, 16 req/s and
"max" (same seeds, byte-identical prompts). Qwen2.5-1.5B, as in the published matrix.

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

- **vLLM: nothing moved.** Access log off, xxHash prefix hashing, and the `interactivity`
  and `throughput` performance modes all land within the spread of vLLM's own repeated
  default runs (TTFT p50 at 4 req/s: 58–63 ms; at "max": 125–173 ms across six starts;
  saturated throughput 4,899–4,987 tok/s). Its defaults already are its best configuration
  here. The matrix now passes `--disable-uvicorn-access-log` for parity only.
- **SGLang: one flag matters.** `--enable-dynamic-batch-tokenizer` raises saturated
  throughput from 5,071 to 5,796 tok/s (+14%) and lowers saturated TPOT p50 from 8.33 to
  6.83 ms, at the cost of ~5 ms of TTFT p50 at 16 req/s (49.6 → 54.6 ms; its p99 improves).
  Confirmed in a second start (`sglang/tuned`: 5,780 tok/s, 6.83 ms). **The matrix now runs
  SGLang with it**, which makes minfer's saturated-throughput deficit larger than
  published: 4,211 vs 5,780 tok/s is 0.73×, where the README said 0.83× of vLLM and 0.84×
  of SGLang. `--schedule-policy lpm` changed nothing; `--enable-mixed-chunk` lowers
  saturated TPOT (5.69 ms) but triples saturated TTFT p50 (308 ms); `--enable-torch-compile`
  made SGLang several times slower on this workload (TPOT 40–70 ms; cause not
  investigated — its own help text calls it experimental) and is reported, not adopted.
- **Does minfer's TTFT lead survive? Against the tuned baselines, yes, as published:**
  39–40 ms vs 62–64 ms (vLLM) and 48–53 ms (SGLang) at 4–16 req/s, and 84 ms vs 125 and
  113 ms at saturation.
- **But most of that lead over SGLang is the HTTP front end, not the engine.** With the
  prompts sent as token ids — no server-side tokenization for anyone — TTFT p50 is
  **16–17 ms (minfer), 36 ms (vLLM), 19–20 ms (SGLang)** at 4–16 req/s, and at saturation
  **33 / 88 / 31 ms**: SGLang is then level with minfer at the median when saturated
  (minfer keeps the better p99: 216 vs 346 ms). So of minfer's ~10 ms lead over SGLang on
  text prompts, about 3 ms is the engine (admission, cache lookup, the prefill pass) and
  the rest is how each server gets 25 KB of prompt text tokenized and to its engine. Over
  vLLM the lead is the same size with and without tokenization (~20 ms), so there it is in
  the engine path. Text prompts are what agents send, so the published comparison is the
  relevant one, but its explanation ("a shorter path from socket to first token") needed
  this decomposition, and the README now carries it.
- The same experiment shows where minfer's own TTFT goes: 23 of its 39 ms on a mostly
  cached 6k-token request is tokenization and request handling in the front end.
- **TPOT and throughput are unchanged by the audit for vLLM** (2.4–2.5 ms; minfer
  2.6–3.0 ms) **and worse for minfer relative to tuned SGLang at saturation** (above).

### Phase 13 acceptance

Every setting compared is in D-038; every change made is in M-019 (FlashInfer decode
configuration, microbenchmark method) and M-020 (`TUNED` flags in `bench/servers.py`).
TTFT, TPOT and throughput on the agent workload are re-measured in M-020. The README's
Triton claim is replaced, its headline explanation of the TTFT lead is qualified, and its
tables carry the audit; the full matrix with the tuned baselines is re-run at 7B in
Phase 14 and at 1.5B in Phase 17.

---

## Related work (citations verified 2026-11-20)

The revised plan frames minfer as a testbed for measuring how agent-serving techniques
contribute alone and in combination; **no mechanism below is claimed as new.** Each entry
was checked against its arXiv or venue page; "abstract only" means the full text was not
opened, and anything marked *unverified* must not be stated as fact in the README.

| work | id / venue | mechanism | evaluation as reported |
|---|---|---|---|
| **Continuum** — Li et al., "Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live" | arXiv 2511.02230 (preprint; v7 Sep 2026) | After a turn ending in a tool call, pin the KV in GPU memory for a TTL chosen from per-tool historical durations; program-level FCFS | Llama-3.1-8B/70B, Gemma-3-12B, GLM-4.5; SWE-Bench (mini-swe-agent), BFCL V4, OpenHands; vs vLLM 0.10.2 (+LMCache DRAM offload), Autellix, InferCept, SGLang, Dynamo. Body: 1.12–3.66× lower delay, 1.10–3.22× throughput. Code: github.com/Hanchenli/vllm-continuum ("preview"; README says it omits the paper's estimator; base vLLM version *unverified*) |
| **InferCept** — Abhyankar et al. | arXiv 2402.01869, ICML 2024 | Tool calls as interceptions, not request ends; reduce GPU memory wasted while paused | abstract only: 1.6–2× throughput |
| **CacheWise** — Tiwari et al. | arXiv 2606.16824 | Prefix-aware dispatch + predictive eviction by expected time to next reuse (tool name/argument clustering), on vLLM | Qwen2.5-Coder-32B, 2×H200, real Claude Code traces at 10–50 sessions; 2–2.6× fewer evictions; DRAM swap variant ~1.19× over vLLM at N=30 |
| **TraceLab** — Zhu et al. | arXiv 2606.30560 | Characterization + dataset (no serving system): ~4,300 Claude Code / Codex sessions, timings and sizes only | tool latency P50 0.3 s, P90 13.6 s, P99 3.6 min; median prefix 126k / 116k tokens |
| **CacheBlend** — Yao et al. | arXiv 2405.16444, EuroSys 2025 | Reuse per-chunk KV at non-prefix positions; selectively recompute a small subset of tokens | abstract only: 2.2–3.3× lower TTFT vs full recompute (RAG) |
| **EPIC** — Hu et al. | arXiv 2410.15332, ICML 2025 | Position-independent caching; recompute a few tokens at each chunk start (LegoLink, v3) | abstract only: up to 8× TTFT, negligible accuracy loss |
| **MiniPIC** — Ordonez, Parnell | arXiv 2606.13126 | Store unrotated keys; apply RoPE inside attention with per-request logical positions | 2WikiMultihopQA (RAG, not agents): +49% prefill throughput over vLLM |
| **CacheSlide** — Liu et al. | USENIX FAST '26 | Relative-position-dependent caching for agents | abstract only: 3.11–4.3× lower latency |
| **Irminsul** — Ma, Eitzinger, Köstler | arXiv 2605.05696 | Content-hash keying over content-defined chunks in SGLang's radix cache + closed-form key rotation; MLA models only | up to ~83% of prompt tokens recovered beyond exact prefix on synthetic patterns; treatment TTFT not measured |
| **Leyline** — same authors | arXiv 2606.01065 | Policy-issued directives splice edits into cached context with RoPE rotation of downstream keys | +11.2 pp hit rate; GQA support only microbenchmarked |

**Is there an existing composed evaluation?** Five searches (queries in the session log)
found no paper that evaluates prefix caching, TTL retention, a host-memory tier,
position-independent reuse and speculative decoding *together* on real coding-agent traces
with interaction effects. That is a negative search result, not proof of absence. Nearest
neighbours, which bound any claim: Continuum (TTL + DRAM/SSD offload), CacheWise
(scheduling + eviction + DRAM swap on real Claude Code traces), EfficientAgent
(arXiv 2609.33762, host-tier sizing on SWE-bench Verified), "Stateful Inference for
Low-Latency Multi-Agent Tool Calling" (arXiv 2605.26289: persistent KV + radix prefix
cache + prompt-lookup speculation on generated workflows), and the vLLM × AgentX blog post
(2026-09-08: many features on real coding traces, no ablation). **Not yet opened, to check
before the README makes any claim about the speculation × cache or position-independent ×
prefix interactions:** KV-Repurpose (arXiv 2610.13060), AgSpec (2610.01108), Callo
(2611.18459), SAGA (2605.00528).

**Trace sources (Phase 14B).** TraceLab's release (`v0.0.1`, CC BY 4.0, 357,161 LLM rounds)
contains timings, token counts and character counts but **no prompt text or token ids**
(paper §3.3), so real token sequences have to come from elsewhere:
`nvidia/Open-SWE-Traces` (CC BY 4.0; full messages, tool calls and tool definitions for
five agent harnesses including Claude Code; no timestamps) supplies content and turn
structure, and TraceLab supplies tool-call pause durations, sampled per tool name.
`nebius/SWE-rebench-openhands-trajectories` is a single 2.1 GB parquet with no timings.

---

## Phase 14 — Qwen2.5-7B-Instruct

### B-006 CPU-heavy work ran next to a benchmark and corrupted a baseline's numbers (process bug)

- **What happened.** The first 7B matrix run was started, and while it ran the CPU test
  suite (five minutes of multi-threaded PyTorch) and two tokenization passes over the real
  traces were run on the same machine. The container has a quota of 13.6 CPU cores. vLLM's
  agent run at 2 req/s came out at TPOT p50 24.9 ms and TTFT p50 300 ms, against 6.4 ms and
  84 ms at 1 req/s minutes earlier: the servers and the replay client were starved of CPU.
- **Why it matters more than B-004.** The contention was not symmetric: the heavy part
  overlapped vLLM's run, not minfer's, so the comparison would have been biased in
  minfer's favour by my own background work.
- **Fix.** The run was stopped and its partial results discarded (kept outside the
  repository's results directory, not committed), and the whole matrix was restarted on an
  idle machine. Rule adopted, extending B-004: **while a benchmark is running, nothing else
  runs — no tests, no tokenization, no downloads — only editing.** Results now also record
  the commit they ran on (`code` in each results file).

### B-007 The HTTP front end and the engine process deadlocked on their pipes (fixed)

- **Symptom.** The clean rerun of the 7B matrix stopped during minfer's ShareGPT "max" run
  (1,000 requests at once): `vllm bench serve` sat at 0/1000 for twelve minutes, the GPU
  was idle, and the server answered neither `/health` nor `/metrics`.
- **Diagnosis.** `/proc/<pid>/wchan` of both processes read `pipe_write`: the front end was
  blocked writing a command to the engine and the engine was blocked writing results to the
  front end. Neither could reach the read that would have released the other.
- **Root cause.** Present since Phase 5 (D-025). The event loop wrote commands to the
  engine with a blocking `Connection.send`. A pipe holds 64 KB. The engine blocks on its
  results pipe whenever the front end falls behind on reading (ordinary back-pressure, and
  with a thousand streams the front end does fall behind); if the event loop then submits a
  request whose command does not fit in what is left of the *command* pipe, it blocks too —
  and a blocked event loop reads no results. It had survived every earlier run by timing:
  at 1.5B an engine step is a third as long, so the engine drained commands three times as
  often.
- **Fix.** The command pipe is non-blocking on the front end's side
  (`AsyncEngine._send`/`_flush`): whatever a write cannot take is buffered and flushed when
  the pipe becomes writable, framed exactly as `Connection.send` frames it so the engine
  still reads with `recv`. The event loop therefore never blocks on the pipe and always
  gets back to reading results. (A first version used a dedicated writer thread; it was
  replaced before any result was kept, to rule out GIL contention while chasing B-008.)
  Regression test
  `test_commands_sent_while_results_are_backed_up_do_not_deadlock`: it backs the results
  pipe up, then submits 500 KB of commands; it hangs on the old code (verified: killed by a
  90 s time limit) and passes with the fix.
- **Consequence for the numbers.** None of the published 1.5B results came from a
  deadlocked run (a deadlock cannot produce a result), but minfer's ShareGPT "max" rows were
  one unlucky interleaving away from not existing. minfer's 7B rows were measured again on
  the fixed code, so that all of them come from one commit.

### B-008 Score-based eviction rescanned the whole cache on every call (fixed)

- **Symptom.** With session-aware retention on by default, minfer's 7B matrix rows at high
  load got worse from one commit to the next (agent "max" 1,663 → 1,535 tok/s; ShareGPT at
  16 req/s TPOT 10.7 → 13.3 ms) although no request in those workloads names a session.
- **Isolation.** First suspect was the writer thread of B-007's first fix (GIL contention);
  replacing it with non-blocking writes changed nothing (1,529 tok/s), so it was not that.
  A/B on identical traces with a fresh server: `--eviction-policy lru` 1,874 tok/s,
  `session` 1,704 tok/s.
- **Root cause.** `PrefixCache._evict_by_score` walks every node and sorts every leaf per
  call, and it is called with the handful of blocks a decode step is short of — several
  times per engine step once the cache is full, which at 7B (1.0 M tokens of KV, 6–14k per
  session) is after about a hundred sessions. In the full matrix the cache holds a few
  thousand leaves by the time the high rates run.
- **Fix.** One scoring pass frees at least 1/256 of the pool (249 blocks at 7B), so passes
  are rare; and the pause predictor answers `E[pause − elapsed | pause > elapsed]` from
  sorted pauses with suffix sums (a binary search) instead of a scan of 512 samples per
  leaf. Same A/B afterwards: 1,869 tok/s, equal to LRU.
- **What it cost in rigor.** The 7B matrix was measured five times before its minfer rows
  came from code without a known defect (B-006, B-007 twice, B-008). Only the last is
  committed; the earlier result files are kept out of the repository.

### M-022 The three-engine matrix at 7B (Phase 14 acceptance)

`bench/results/matrix_7b/*.json` (`python -m bench.run_matrix --model 7b`), plots
`bench/plots/{agent,sharegpt}_vs_rate.png`. Qwen2.5-7B-Instruct, bf16, one H100; baselines
with the flags of M-020 (vLLM: access log off; SGLang: dynamic batch tokenizer); 0 failed
requests in every run. minfer's rows are from commit 9b0010f, measured after the
baselines' rows in a separate invocation on the same idle machine (B-008). minfer's fp32
logits match transformers at 7B to 2.3e-4 (`bench/results/model_numerics_7b.json`).

| agent workload | minfer | vLLM | SGLang |
|---|---|---|---|
| TTFT p50, 1–16 req/s (ms) | 46–59 | 78–81 | 57–67 |
| TTFT p99, 1–16 req/s (ms) | 87–126 | 121–136 | 97–125 |
| TPOT p50, 1–16 req/s (ms) | 6.67–9.68 | 6.26–7.88 | 6.27–8.20 |
| E2E p50, 1–16 req/s (ms) | 444–637 | 453–555 | 433–561 |
| "max": output tok/s | 1,858 | 2,258 | 2,264 |
| "max": TTFT p50 / p99 (ms) | 105 / 299 | 161 / 835 | 146 / 338 |
| "max": TPOT p50 (ms) | 26.9 | 19.9 | 20.8 |

(These are the rows as first measured, at commit 9b0010f. The committed results file was
re-measured on the final code in Phase 17 and differs by run-to-run noise: TTFT p50
49–57 ms, TPOT p50 6.64–9.78 ms, 1,836 tok/s saturated; M-029.)

- **TTFT: minfer leads at the median at every rate and at saturation**, by 22–32 ms over
  vLLM and 8–10 ms over SGLang; at the 99th percentile it leads everywhere except 16 req/s
  (126.5 ms vs SGLang's 121.0 ms).
- **TPOT: behind at every rate, and the gap grows with load** — 1.06× of vLLM at 1 req/s,
  1.12× at 8, 1.23× at 16, 1.35× at "max". At 1.5B the gap was 1.06–1.24× with no such
  trend at low rates.
- **Saturated throughput: 0.82× of both** (1.5B: 0.83× of vLLM).
- **ShareGPT:** TTFT p50 21–24 ms vs 35–48 (vLLM) and 26–36 (SGLang) at 1–16 req/s; TPOT
  p50 6.31–7.96 vs 6.21–6.82 and 6.19–7.66 ms. With 1,000 requests at once: 12.2k /
  14.7k / 14.0k output tok/s and median TTFT 4.2 / 3.5 / 3.3 s — at 7B minfer no longer has
  the highest burst throughput it had at 1.5B.
- **What scaling the model changed.** The TTFT lead is intact: minfer's median is
  0.59–0.73× of vLLM's and 0.82–0.89× of SGLang's at 1–16 req/s.
  The decode gap, which at 1.5B was a matter of kernel counts, did not close with a model
  whose steps are matmul-bound; it became load-dependent instead. Not isolated by
  measurement: whether the two passes of a mixed step (D-013) account for all of it.

---

## Phase 14B — Real traces

### D-039 Two datasets, because no public one has both halves

- **Content** comes from `nvidia/Open-SWE-Traces` (`claudecode` split, one 54 MB shard,
  382 trajectories; CC BY 4.0): complete Claude-Code-harness sessions on repository issues —
  system prompt, three tool definitions (Bash, Read, Edit), every assistant message with
  its tool calls, every tool result. No timestamps.
- **Timing** comes from TraceLab `v0.0.1` (CC BY 4.0): 141,824 tool calls of real Claude
  Code sessions with emit and result timestamps, and no text at all. A model call's pause is
  sampled from the empirical latency distribution of the tool it calls (the slowest, if it
  calls several), from a committed table of 201 quantiles per tool
  (`bench/data/tracelab_tool_latency.json`, 24 KB; `python -m bench.real_traces latency`
  rebuilds it from the release, whose SHA-256 it records).
- **What is real and what is not.** Token sequences, turn structure, tool names and the
  marginal distribution of pauses per tool are real. The *pairing* of a particular pause
  with a particular call is sampled, so any correlation between what a command does and how
  long it takes (beyond the tool's name) is lost. Human think time does not occur: these
  are autonomous runs on one issue each.
- **Replay** follows D-027: the request for a call is the chat-templated conversation up to
  it, with the *recorded* assistant messages in the history, asking for exactly as many
  tokens as the recorded reply has — byte-identical prompts for every engine. Sessions run
  closed-loop at a fixed concurrency.
- **Context limit.** Qwen2.5 has 32,768 positions; the plan's 16k–48k-token histories are
  therefore 16k–32k. A session keeps the calls that fit 32,000 tokens and ends there:
  163 of 300 sessions are cut this way. Real Claude Code sessions are far longer (TraceLab:
  median prefix 126k tokens); this trace is the part of them a 32k-context model can serve.

### M-021 What the real trace looks like (`bench/results/real_trace_stats_7b.json`)

300 sessions, 8,112 model calls, 141 M prompt tokens and 1.5 M output tokens in total.

| | p50 | p90 | max |
|---|---|---|---|
| model calls per session | 28 | 42 | 57 |
| prompt tokens per call | 17,806 | 28,482 | 31,938 |
| tokens appended per call | 549 | 1,916 | 26,443 |
| output tokens per call | 100 | 451 | 7,179 |

Pauses (after clipping at 120 s): p50 0.26 s, p75 3.5 s, p90 30 s, p99 120 s; 40% of
pauses exceed 1 s, 16% exceed 10 s, 10% exceed 30 s. Per tool in TraceLab: Bash p50 0.41 s /
p90 30 s (66,850 calls), Read 0.04 s / 0.43 s, Edit 0.10 s / 2.2 s.

**How much of the prompt each kind of reuse could serve** (rolling-hash fingerprints, cache
of unlimited size, sessions one after another; a token is *prefix* if a 16-token-aligned
prefix containing it was seen before, *shifted duplicate* if it is not but lies in a
32-token window seen before at another position, *novel* otherwise):

| | prefix | shifted duplicate | novel |
|---|---|---|---|
| all prompt tokens | 94.4% | 0.8% | 4.8% |
| first call of a session | 6.3% | 66.8% | 26.9% |
| later calls | 94.7% | 0.6% | 4.7% |

Three things this says before any engine runs:

1. **Exact-prefix caching already covers 94% of the tokens**, if nothing is ever evicted.
   Everything else in this study is about the remaining 5.6% and about what eviction takes
   away from the 94%.
2. **Position-independent reuse has little to find here: 0.8% of prompt tokens**, 14% of
   the tokens a perfect prefix cache still has to compute. Almost all of it is the system
   prompt of a session's first call: Claude Code's system prompt begins with the working
   directory, the date and a git status, so two sessions share their long system prompt
   only *after* a different first paragraph — which defeats prefix caching (6.3% hit on
   first calls) and is exactly the shifted-duplicate case (66.8%). Within a session,
   re-read files and repeated tool output amount to 0.6% of tokens.
3. **Pauses are short and heavy-tailed, not "2–30 s".** Half of them are under 0.3 s. A
   retention policy has to be right about the 16% of pauses longer than 10 s, which carry
   almost all of the paused time.

---

## Phase 15 — Tool-call-aware KV retention and a host-memory tier

Neither mechanism is new: KV time-to-live for paused agent sessions is Continuum
(arXiv 2511.02230), predictive eviction by expected time to next reuse is CacheWise
(arXiv 2606.16824), and DRAM offload of KV is in LMCache, in vLLM and in SGLang's
hierarchical cache. What this phase adds is both of them in one engine where each can be
switched off, measured on the same real trace next to the baselines' own offloading.

### D-040 Retention is a ranking by expected time to next use, not a pin with a deadline

- **Interface.** A request may carry `session_id`, and optionally `session_tools` (the
  tools the client runs after this response) and `session_final`. Without the last two the
  server reads them off its own output (`<tool_call>` blocks: tool names; none: a final
  answer) and tells the engine after the response (`Engine.note_session`). In the replay
  benchmarks the client sends them, because the history is scripted and the generated text
  is not what the next request contains; this is the same information a server parsing its
  own output has, one message earlier.
- **Policy** (`minfer/cache/retention.py`). Every cached leaf gets a score: the expected
  seconds until it is next needed. Eviction takes the highest score first, trimming a leaf
  from its tail as before.
  - session said it is over → never (evicted first, however recently it was used);
  - session paused in a tool call → `E[pause − elapsed | pause > elapsed]` over the pauses
    observed so far for that tool (falling back to all tools, then to `elapsed` itself);
  - a request of the session is in the engine → 0;
  - KV of no session → its idle time, which orders such entries exactly as LRU does.
- **Why a ranking and not Continuum's TTL.** A TTL answers "may this be evicted yet?"; under
  memory pressure something must go regardless, and the question is which. Ranking by
  expected remaining pause gives the TTL's behaviour where it applies (a session two
  seconds into a pause that usually takes five is kept) and still has an answer when every
  candidate is inside its TTL.
- **Why the conditional expectation.** Tool latencies are heavy-tailed (M-021): a session
  ten seconds into a `Bash` pause whose median is 0.4 s is not about to return, it is in the
  tail. Conditioning on the time already elapsed makes its score *rise* the longer it
  stays away, which is the opposite of what "it has been gone a long time, it must be due"
  intuition (and LRU's opposite, "it has been gone a long time, drop it") would do, and it
  is what the data says.
- **Tradeoffs.** (1) The predictor learns online from this server's own traffic (the last
  512 pauses per tool set): it starts with no history, where it degenerates to LRU order.
  (2) Scoring walks every leaf at each eviction: O(sessions) per call, measured below.
  (3) A node shared by several sessions carries the mark of whichever finished last; it is
  only ever scored after all private tails below it are gone.
- `--eviction-policy lru` restores the previous behaviour and ignores session fields.

### D-041 Host tier: offload at every pause, demote on eviction, restore on return

`minfer/cache/host_tier.py`, `--host-kv-gb N`.

- **Offload eagerly, incrementally.** When a session's request completes and the session is
  not over, the blocks on its cached path that have no host copy yet are copied to pinned
  host memory. KV is immutable once published, so every block is copied at most once and a
  turn costs only its new tokens (median 549 appended + 100 generated tokens ≈ 40 blocks ≈
  37 MB at 7B). Considered and rejected: copying only at eviction time. It makes eviction —
  which happens inside admission, on the request path — wait for a gather of exactly the
  bytes it is trying to free, at the moment memory is tightest.
- **Demotion is free.** Evicting a block that has a host copy drops the GPU copy and keeps
  the tokens in the radix tree as a *host-only* node (the leaf is split at the boundary).
  Blocks without a copy are forgotten as before, together with any host-only KV below them,
  which could never be reached again.
- **Restore on return.** At admission, after a sequence has adopted what the GPU cache
  holds, the tree is asked for host-only nodes that continue its prefix. If there are at
  least `--min-restore-blocks` (8) of them, GPU blocks are allocated (evicting colder KV if
  need be), the copy is started, and the sequence steps out of the queue; other sequences
  keep being admitted and stepped. When the copy's CUDA event has completed the nodes
  become GPU-resident again and the sequence rejoins the front of the queue, where the
  ordinary prefix-cache hit path picks the blocks up. A restored block keeps its host copy,
  so being evicted again is another free demotion.
- **Asynchrony.** Copies run on their own CUDA stream. The GPU store is layer-major (what
  the kernels index) and the host pool block-major, so a run of host blocks is one
  contiguous pinned range; an offload gathers into a staging tensor on the engine's stream
  (after which the GPU blocks may be reused at once) and copies it out on the transfer
  stream; a restore copies into a staging tensor on the transfer stream and the engine
  scatters it into the store once the event is done. The engine polls and never waits.
- **Host pool full.** Room is made by dropping host-only KV whose score is higher than the
  pausing session's own — KV needed later than the newcomer. A host-only node is the only
  copy of its tokens, so it is not given up for a backup of something needed later than it.
- **No predictive prefetch, and why.** The plan asks for bringing KV back *before* the next
  turn arrives. With heavy-tailed pauses there is no moment at which a paused session
  becomes "about to return": its expected remaining pause only grows with the time it has
  been away (D-040). The sessions likely to return soon are the ones that paused a moment
  ago, and those are the last to be demoted in the first place. What is implemented is
  overlap rather than anticipation: a restore runs concurrently with the engine's other
  work and only its own request waits for it.
- **Restore or recompute.** Measured, not assumed: M-023.
- **Correctness.** On the CPU float32 model, a randomized test with eight sessions, a GPU
  pool and a host pool both too small for them asserts every output equal to the cache-free
  oracle and the tree/pool invariants after every step. On the GPU, an evicted and restored
  session continues bit-for-bit as it does in an engine with room for everything
  (`tests/test_host_tier.py`): restored KV *is* the KV that was computed.

### M-023 Restoring KV from host memory vs recomputing it

Script: `bench/host_tier_cost.py` → `bench/results/host_tier_cost_7b.json`. Qwen2.5-7B,
bf16, one H100 on PCIe 5.0 x16, pinned host memory; median of 5. *Restore* is timed from
the request to the blocks being usable (side-stream copy, completion event, scatter into
the store); *recompute* is the engine prefilling the same number of uncached tokens to its
first token (one request, chunked prefill off).

  | tokens | KV (MB) | offload (ms) | restore (ms) | restore GB/s | recompute (ms) | recompute tok/s | recompute / restore |
  |---|---|---|---|---|---|---|---|
  | 1,024 | 56 | 1.3 | 1.2 | 44 | 23 | 44.6k | 18× |
  | 4,096 | 224 | 5.1 | 4.8 | 45 | 93 | 44.0k | 19× |
  | 8,192 | 448 | 9.3 | 9.1 | 48 | 193 | 42.4k | 21× |
  | 16,384 | 896 | 17.7 | 17.8 | 49 | 423 | 38.8k | 24× |
  | 30,000 | 1,641 | 31.9 | 31.8 | 50 | 928 | 32.3k | 29× |

- **A copy is 18–29× cheaper than a prefill**, and the ratio grows with the history: the
  copy scales with bytes at 44–50 GB/s, the prefill's attention is quadratic in context
  (44k tokens/s at 1k, 32k tokens/s at 30k). A 30,000-token history comes back in 32 ms
  instead of 928 ms.
- **So "choose per session" has a simple answer on this hardware:** restore whenever a
  copy exists. The crossover is not in the measured range; the only reason to recompute is
  latency quantization — a restore is noticed at the next engine step, so for a handful of
  blocks a prefill in that same step is as fast. That is `--min-restore-blocks` (8 blocks,
  128 tokens, about 3 ms of prefill).
- **Offload costs the same as restore** (it is the same bytes), 1.3 ms per 1,024 tokens,
  and happens during the pause, off the request path.
- At 57 KB per token a 96 GB host tier holds 1.8 M tokens, 1.9× what the GPU pool holds at
  7B (963k tokens).

### M-024 KV memory pressure on the real trace: three engines, with and without a host tier (Phase 15 acceptance)

> **Re-measured (M-030).** The minfer rows below were measured with the Triton decode kernel. The committed results file and the README now hold the rows re-measured on the default kernel; M-030 has them. The baselines' rows are these.

Script: `bench/pressure.py` → `bench/results/pressure_7b.json`, plot
`bench/plots/pressure_7b.png`. Qwen2.5-7B, the 300-session real trace (D-039) replayed
closed-loop at 32, 64, 96 and 128 concurrent sessions; a fresh server per run; 360 s of
load of which the first 120 s (30 s of staggered starts, then warm-up) are not counted; the
whole GPU for every engine. 0 failed requests in all 24 runs. During the four hours of the
sweep only documentation and result-rendering code changed in the repository
(`git diff 558a87f 5848785 -- minfer bench/pressure.py bench/real_traces.py
bench/servers.py` is empty), which is why the recorded commits differ between rows.

**Capacity.** One session holds 17k tokens of KV at the median call and 26k at its end
(M-021). GPU KV pools at 7B: minfer 963k tokens, vLLM 1,106k, SGLang 952k — about 55–65
median sessions. Host tiers, 96 GB each (the most vLLM's native offloading can place in
this machine's `/dev/shm`): minfer 1.80 M tokens, SGLang 1.67 M, vLLM a 103 GB region.
vLLM is the baselines' `--kv-offloading-size 96 --kv-offloading-backend native`, SGLang
`--enable-hierarchical-cache --hicache-size 96`; LMCache is not installed in the pinned
vLLM environment and was not tried.

| sessions | configuration | calls | hit rate | prefill tokens / call | TTFT p50 / p99 (ms) | TPOT p50 (ms) | out tok/s |
|---|---|---|---|---|---|---|---|
| 32 | minfer, session-aware | 627 | 94.2% | 959 | 101 / 287 | 10.7 | 376 |
| | vLLM | 654 | 93.9% | 1,005 | 131 / 277 | 8.6 | 403 |
| | SGLang | 653 | 94.1% | 972 | 109 / 317 | 8.9 | 400 |
| 64 | minfer, LRU | 1,030 | 88.5% | 1,906 | 134 / 1,350 | 27.7 | 564 |
| | minfer, session-aware | 1,074 | 91.0% | 1,496 | 128 / 1,245 | 25.5 | 593 |
| | minfer, session-aware + host tier | 1,101 | 94.2% | 962 | 128 / 478 | 25.7 | 604 |
| | vLLM | 1,238 | 89.9% | 1,695 | 165 / 875 | 17.6 | 701 |
| | vLLM + KV offloading | 1,272 | 94.1% | 1,002 | 174 / 392 | 16.5 | 717 |
| | SGLang | 1,202 | 88.7% | 1,891 | 134 / 1,025 | 18.5 | 670 |
| | SGLang + hierarchical cache | 1,264 | 93.8% | 1,053 | 126 / 528 | 16.9 | 729 |
| 96 | minfer, LRU | 788 | 55.3% | 6,913 | 1,142 / 37,006 | 62.9 | 388 |
| | minfer, session-aware | 788 | 59.7% | 6,227 | 1,204 / 28,856 | 69.0 | 398 |
| | minfer, session-aware + host tier | 1,194 | 94.0% | 985 | 266 / 3,913 | 46.8 | 641 |
| | vLLM | 1,302 | 82.2% | 3,012 | 221 / 1,703 | 38.3 | 696 |
| | vLLM + KV offloading | 1,494 | 91.0% | 1,545 | 206 / 1,091 | 27.2 | 809 |
| | SGLang | 824 | 50.2% | 8,024 | 410 / 19,150 | 42.9 | 406 |
| | SGLang + hierarchical cache | 1,446 | 90.1% | 1,702 | 153 / 1,409 | 29.7 | 772 |
| 128 | minfer, LRU | 722 | 34.0% | 9,506 | 8,607 / 55,017 | 77.1 | 319 |
| | minfer, session-aware | 736 | 42.5% | 8,287 | 8,973 / 45,232 | 84.8 | 338 |
| | minfer, session-aware + host tier | 1,183 | 89.5% | 1,655 | 3,501 / 14,640 | 55.0 | 605 |
| | vLLM | 791 | 35.2% | 9,877 | 2,736 / 26,787 | 138.8 | 362 |
| | vLLM + KV offloading | 1,484 | 85.0% | 2,472 | 704 / 7,278 | 45.4 | 765 |
| | SGLang | 743 | 29.4% | 10,572 | 10,280 / 30,487 | 135.5 | 333 |
| | SGLang + hierarchical cache | 1,356 | 84.7% | 2,486 | 3,071 / 12,147 | 45.8 | 694 |

"Hit rate" is the share of prompt tokens the server reported as served from cache; with
an unlimited cache this trace gives 94.4% (M-021).

**1. Below capacity nothing differs** (32 sessions): 93.9–94.2% for all three, i.e. the
ceiling.

**2. Without a host tier every engine collapses once live KV exceeds GPU memory**, and it
is a cliff: minfer's LRU goes from 88.5% at 64 sessions to 55.3% at 96 and 34.0% at 128,
SGLang 88.7 → 50.2 → 29.4%, vLLM 89.9 → 82.2 → 35.2% (its pool is 15% larger and it drains
requests faster, which moves its cliff from before 96 to before 128, not away). Past the
cliff an engine prefills 7–10k tokens per call instead of 1k, median TTFT is measured in
seconds, and throughput halves.

**3. Session-aware retention alone is a small, consistent gain and does not prevent the
collapse.** Against LRU in the same engine: hit rate +2.5 / +4.5 / +8.5 points at
64 / 96 / 128 sessions, prefill tokens per call −22% / −10% / −13%, TTFT p99 −8% / −22% /
−18%, throughput +5% / +2% / +6%. When the live sessions do not fit, *which* one is evicted
matters little: on this trace every session comes back within seconds (median pause
0.26 s), so every victim is the wrong one. Continuum reports much larger gains; its
design also includes program-level scheduling, which this engine does not have, and its
models and workloads differ, so the two results are not a like-for-like comparison. On
this trace, with this policy alone, the honest size of the effect is the numbers above.

**4. A host-memory tier is what decides this workload — for every engine.** minfer with
its tier is back at the ceiling at 64 and 96 sessions (94.2%, 94.0%) and at 89.5% at 128,
where the host pool is full as well; throughput is 1.6× and 1.8× that of the same engine
without it at 96 and 128, TTFT p99 7× and 3× lower. vLLM's and SGLang's own offloading do
the same for them (throughput 2.1× at 128 for both). At 96 sessions minfer served 748
requests from host memory in the six-minute run, copying 10 M tokens of KV back.

**5. With a host tier, the baselines match or beat minfer on latency and throughput;
minfer avoids the most prefill work.**

| with host tier, 64 / 96 / 128 sessions | minfer | vLLM | SGLang |
|---|---|---|---|
| hit rate | **94.2 / 94.0 / 89.5%** | 94.1 / 91.0 / 85.0% | 93.8 / 90.1 / 84.7% |
| prefill tokens per call | **962 / 985 / 1,655** | 1,002 / 1,545 / 2,472 | 1,053 / 1,702 / 2,486 |
| output tok/s | 604 / 641 / 605 | 717 / **809 / 765** | **729** / 772 / 694 |
| TTFT p50 (ms) | 128 / 266 / 3,501 | 174 / 206 / **704** | **126 / 153** / 3,071 |
| TTFT p99 (ms) | 478 / 3,913 / 14,640 | **392 / 1,091 / 7,278** | 528 / 1,409 / 12,147 |
| TPOT p50 (ms) | 25.7 / 46.8 / 55.0 | **16.5 / 27.2 / 45.4** | 16.9 / 29.7 / 45.8 |

minfer computes 36% fewer prefill tokens per call than vLLM at 96 sessions and 33% fewer
at 128 (offload at every pause plus session-aware demotion keep more of the history
reachable), and it still delivers 0.79× of vLLM's throughput at both, with a higher TTFT
at the median and the tail. The reason is not KV management: it is the decode path of
M-022. On this trace *every* step has prefill work (a call appends ~950 tokens), so minfer
pays its two passes per step all the time: TPOT is 1.56× vLLM's at 64 sessions, where
both engines are at the hit-rate ceiling and KV management plays no part.

**Phase 15 acceptance, honestly.** Plots and results are committed for all three engines
and the three minfer variants; policy and tradeoffs are D-040 and D-041. The headline the
plan hoped for — tool-call-aware retention beating recency under pressure — is true and
small (finding 3). The technique that matters is the host tier, the baselines have it
too, and with it they are faster than minfer end to end (finding 5).

---

## Phase 16B — Close the decode gap (optional; one item attempted)

### M-025 Unified prefill+decode passes: better at low load, worse at saturation

> **Re-measured (M-030).** The numbers below are with the Triton decode kernel; on the default kernel the result is the same (M-030).

`--unified-batches`: in a step with both kinds of work, decoding sequences join the ragged
prefill pass as one-token spans instead of getting a second pass over the weights; prefill
CUDA graphs are captured for up to 256 spans. On the CPU model the outputs equal the
oracle under both settings and a mixed step is one pass (`tests/test_engine.py`).

`bench/unified_batches.py` → `bench/results/unified_batches_7b.json`, agent workload at 7B,
same traces, fresh server per setting:

| | 4 req/s | 16 req/s | "max" |
|---|---|---|---|
| split passes: TPOT p50 (ms) | 7.29 | 9.58 | 26.69 |
| unified: TPOT p50 (ms) | 7.18 | 9.31 | 36.66 |
| split passes: TTFT p50 (ms) | 53.3 | 57.9 | 106.2 |
| unified: TTFT p50 (ms) | 47.4 | 55.9 | 141.9 |
| split passes: output tok/s | 256 | 743 | 1,858 |
| unified: output tok/s | 257 | 745 | 1,429 |

- **It helps where steps are small** (TPOT −1.5% and −2.8%, TTFT p50 −6 and −2 ms at 4 and
  16 req/s) **and costs 23% of saturated throughput.** A unified pass serves every decoding
  sequence through FlashInfer's batch *prefill* kernel as a one-token query over a context
  of 6–14k tokens; the split design serves them through the Triton decode kernel inside a
  decode-shaped CUDA graph. With 64 sequences decoding, the second pass over the weights
  that unification saves is cheaper than what the prefill kernel loses on those rows.
  (Inferred from the load-dependence; the two kernels were not timed side by side on
  one-token spans.)
- **Off by default, kept as an option**, as the plan's ground rules require for a feature
  that underperforms. Closing the gap this way needs a paged attention kernel that is good
  at mixed batches — what vLLM's FlashAttention path is — not a rerouting of rows.
- Items (b) fusing the KV write into the RoPE kernel and (c) further profiling were not
  attempted: the phase is optional in the revised plan and its first item gave a negative
  result. The decode gap of M-022 stands.

---

## Phase 15C — Position-independent KV reuse

The mechanism is CacheBlend's and EPIC's (reuse KV of identical content at another
position, recompute a small part), with MiniPIC's observation that RoPE makes the
positional part of a key a rotation; nothing here is new.

### D-042 Rotate cached keys instead of storing unrotated ones; recompute each run's head

- **Finding runs.** With `--position-independent-reuse` the prefix cache also indexes every
  cached block by the hash of its 16 tokens alone. At admission the uncached part of a
  prompt is scanned for windows that hit the index, each hit is checked token by token and
  extended along the source edge, and runs of at least `--reuse-min-tokens` (64) are
  planned; their source blocks are pinned until copied.
- **Copying.** The store holds *rotated* keys (D-013). A key cached at position p and
  needed at p + Δ is the same vector rotated by Δ more steps, so one gather, one rotation
  by Δ·θ in float32 and one scatter per run move the keys of all layers; values are copied
  unchanged (`PagedKVStore.copy_rotated`). MiniPIC stores unrotated keys and rotates inside
  attention; rotating at copy time needs no kernel change and costs one pass over the
  copied tokens.
- **Selective recomputation, EPIC-style.** The first `--reuse-recompute-ratio` (15%) of
  each run is computed in its new context before the rest is copied. CacheBlend picks the
  tokens with the largest KV deviation anywhere in a chunk; that needs a pass with queries
  at arbitrary positions under a causal mask, which the prefill kernel used here does not
  offer, so the recomputed part is always a run's head and the engine stays strictly
  left to right: compute up to a head's end, copy, continue.
- **What is exact and what is not.** Layer 0's keys and values are a function of the token
  and its position only, so their reuse is exact up to the rotation's rounding. Every
  deeper layer's were computed after different preceding tokens. A one-layer model
  therefore reproduces the cache-free oracle exactly under reuse
  (`test_reuse_is_exact_for_a_one_layer_model`, at 0% and 25% recomputation), which pins
  down the rotation, the slots and the positions; for real models the error is measured.
- **Reused KV is published** to the prefix cache like any other, so later turns of the
  session hit on it — and inherit its approximation.

### M-026 Quality and savings on real trajectories (Phase 15C acceptance)

Script: `bench/pic_quality.py` → `bench/results/pic_quality_7b.json`. Qwen2.5-7B, the real
trace. Sixteen sessions' first calls fill the cache; then the first call of each of 48
other sessions — the shifted-duplicate case of M-021 — is prefilled without reuse and at
five recompute ratios. After each prefill the first 128 tokens of the recorded reply are
teacher-forced on top of the KV the engine left behind, and the next-token distributions
are compared with the exact prefill's at 4,679 positions.

| prefill | prompt tokens computed | top-1 same as exact | top-1 = recorded token | mean KL from exact (nats) | max KL | mean log-prob of recorded token |
|---|---|---|---|---|---|---|
| exact (reference) | 93.7% | — | 71.04% | — | — | -1.712 |
| exact, prefill in 1,024-token chunks (noise floor) | 93.7% | 99.70% | 70.93% | 0.0003 | 0.02 | -1.713 |
| reuse, recompute 0% of each run | 27.7% | 96.92% | 70.96% | 0.0143 | 1.01 | -1.748 |
| reuse, recompute 5% of each run | 31.0% | 96.90% | 70.96% | 0.0122 | 0.73 | -1.740 |
| reuse, recompute 15% of each run | 37.6% | 97.11% | 70.96% | 0.0098 | 0.81 | -1.729 |
| reuse, recompute 30% of each run | 47.5% | 97.44% | 70.96% | 0.0083 | 1.10 | -1.729 |
| reuse, recompute 50% of each run | 60.7% | 98.10% | 70.78% | 0.0052 | 0.47 | -1.726 |

- **Savings: the prompt tokens computed on a session's first call fall from 93.7% to
  27.7%** with no recomputation, and to 37.6% at the default 15%.
- **Quality: measurably worse than exact, by a small amount.** About 3% of next-token
  argmaxes change (96.9–97.4% agreement at 0–30% recomputation) against a noise floor of
  0.3% between two exact bf16 prefills that differ only in chunking; mean KL is
  0.008–0.014 nats against 0.0003. Agreement with the token the original agent actually
  produced does not move (70.8–71.0% in every row), and its mean log-probability drops by
  0.01–0.03 nats. 38–45 of the 48 calls have at least one changed argmax within 128
  tokens, against 12 for the noise floor: under greedy decoding most replies would
  eventually diverge from the exact run's, as they do between any two bf16 runs (M-002),
  only sooner.
- **Recomputing more buys little until it is half the run**: agreement is 96.9% at 0%,
  97.1% at 15%, 97.4% at 30%, 98.1% at 50%. The head of a run is not where the error
  lives; this is the cost of recomputing heads rather than CacheBlend's highest-deviation
  tokens.
- **How much it is worth on this workload: little.** First calls are 1,526 tokens
  on average here and are 1 of every 27 calls; over the whole trace shifted duplicates are
  0.8% of prompt tokens (M-021). The saving is about a thousand tokens of prefill — some
  25 ms at 7B (M-023) — once per session, paid for with the approximation above for the
  rest of the session. The composition study measures it end to end.

---

## Phase 16 — Making speculative decoding pay off

### D-043 Verification in a CUDA graph; one pass per speculative step; per-request pausing

- **Why Phase 7's speculation lost** (M-010): a verification span needs the prediction of
  every row, which the captured prefill graphs (one row per span) did not give, so it ran
  in the eager prefill pass — and the sequences without a draft took a second, decode pass.
- **`VerifyGraphs`** (`minfer/attention/prefill_graphs.py`) captures the pass at shapes
  `(S, S·m)` for S ∈ {2 … 256} spans and m = 1 … 1 + `num_draft_tokens`, applies the
  unembedding to *every* row and takes the argmax inside the graph, so only token ids
  leave the device. Greedy sequences that have no draft this step ride along as one-token
  spans: a speculative step is one graph replay instead of an eager pass plus a decode
  replay. Sampling requests still need full distributions and fall back to the eager pass.
- **Adaptive draft length, with an off switch per request.** The draft length still doubles
  after a fully accepted draft and falls to `accepted + 1` after a rejection (D-029). New:
  a running acceptance average per request (`0.7·old + 0.3·accepted/drafted`); below
  `--draft-min-acceptance` (0.3) nothing is drafted for that request for
  `--draft-cooldown-steps` (32) decode steps, then it starts again at one token.
- **Greedy outputs are unchanged by construction**: the accept rule is the same, fed with
  the same argmaxes. On the CPU float32 model outputs with speculation equal the oracle
  for scripted drafts of any quality and with pausing (`tests/test_speculation.py`, 35
  tests); on the GPU the bf16 near-tie caveat of M-010 applies as before.

### M-027 Speculation through the server at 7B (Phase 16 acceptance)

> **Re-measured (M-030).** The table below is with the Triton decode kernel; the committed results file now holds the re-measurement on the default kernel, which M-030 tabulates. The conclusions are unchanged.

Script: `bench/speculation_serve.py` → `bench/results/speculation_serve_7b.json`. A fresh
`minfer serve` per configuration, the matrix's agent traces at 1, 4 and 8 req/s, greedy.

| configuration | TPOT p50 (ms) at 1 / 4 / 8 req/s | vs off | TPOT p99 (ms) | E2E p50 (ms) | drafts accepted | output from drafts |
|---|---|---|---|---|---|---|
| no speculation | 6.63 / 7.26 / 7.88 | — | 7.8 / 9.0 / 12.3 | 441 / 493 / 545 | — | — |
| n=3, ≤ 4 drafted | 5.76 / 6.84 / 8.21 | 0.87× / 0.94× / 1.04× | 8.8 / 13.1 / 16.8 | 391 / 472 / 596 | 62–64% | 24–28% |
| n=2, ≤ 4 drafted | 5.91 / 6.99 / 9.01 | 0.89× / 0.96× / 1.14× | 8.7 / 13.5 / 17.0 | 401 / 498 / 635 | 54–55% | 23–26% |
| n=4, ≤ 4 drafted | 6.03 / 6.81 / 8.52 | 0.91× / 0.94× / 1.08× | 8.6 / 13.0 / 17.0 | 387 / 488 / 609 | 64–67% | 21–26% |
| **n=3, ≤ 8 drafted** | **5.71 / 6.71 / 7.93** | **0.86× / 0.92× / 1.01×** | 8.6 / 12.7 / 16.2 | 384 / 463 / 570 | 52–54% | 26–29% |
| n=5, ≤ 8 drafted | 5.96 / 6.77 / 8.09 | 0.90× / 0.93× / 1.03× | 8.6 / 13.0 / 15.7 | 386 / 457 / 587 | 55–58% | 22–25% |
| n=3, ≤ 4, never paused | 5.59 / 6.72 / 8.27 | 0.84× / 0.93× / 1.05× | 8.8 / 12.9 / 16.9 | 389 / 474 / 594 | 61–63% | 25–29% |
| n=3, ≤ 4, eager verification (Phase 7) | 6.34 / 8.36 / 11.17 | 0.96× / 1.15× / 1.42× | 11.6 / 17.4 / 20.7 | 422 / 619 / 774 | 62–63% | 24–28% |

- **The target is met at low and medium rates, not beyond.** With the best setting (n=3,
  up to 8 drafted tokens) median TPOT is 14% lower than without speculation at 1 req/s and
  8% lower at 4 req/s; at 8 req/s it is level (+1%). End-to-end latency of a turn follows:
  −13%, −6%, +5%.
- **The graph is what made the difference.** The same drafts verified eagerly (last row;
  Phase 7's design) are 4% faster at 1 req/s and 15% and 42% *slower* at 4 and 8.
- **The tail gets worse in every speculative configuration**: TPOT p99 8.6–8.8 ms against
  7.8 at 1 req/s, 13 against 9 at 4, 16–17 against 12 at 8. A request whose drafts are
  rejected pays for verification rows that produced nothing, and a verification pass with
  many rows delays every stream in that step.
- **Why the gain fades with load.** A quarter of the output tokens come from drafts
  (24–29%), so a stream needs about 25% fewer steps; but a step with drafts carries up to
  nine rows per sequence instead of one, and at 7B a step's cost follows its rows. At
  1 req/s steps are nearly empty and the extra rows are free; at 8 req/s they are not.
- **Tuning.** n=3 is the best n-gram size at draft length 4 (n=2 accepts less: 54% vs 62%;
  n=4 and 5 accept more of fewer drafts). Longer drafts (8) help a little at every rate.
  Pausing made no measurable difference on this workload (never-paused: 0.84× / 0.93× /
  1.05×): prompt lookup only drafts when it finds a match, so requests with persistently
  bad drafts are rare here.
- **Still off by default.** It is a latency feature for lightly loaded servers that costs
  tail latency, and it changes nothing for sampling requests' graphs. Recommended setting
  when on: `--speculation ngram --ngram-size 3 --num-draft-tokens 8`.

---

## Phase 17 — Composition study

### M-028 Five techniques, every valid combination, on the real trace under memory pressure

> **Re-measured (M-030).** The table below is with the Triton decode kernel; the committed results file now holds the re-measurement on the default kernel, which M-030 tabulates. The conclusions are unchanged; the absolute numbers and the unexplained failures are not.

Script: `bench/composition.py` → `bench/results/composition_7b.json`, chart
`bench/plots/composition_7b.png`. Qwen2.5-7B, the real trace at 96 concurrent sessions
(1.5–1.7× what the GPU's KV pool holds, M-024), a fresh `minfer serve` per combination,
360 s of load with the first 120 s not counted. Techniques: `prefix` (radix-tree prefix
cache), `session` (session-aware retention; needs prefix), `host` (96 GB host tier; needs
session), `reuse` (position-independent reuse at 15% recomputation; needs prefix), `spec`
(prompt lookup, n=3, ≤ 8 drafted, graph verification). 14 valid combinations.

| techniques | calls | prefill tokens / call | hit rate | TTFT p50 / p99 (ms) | TPOT p50 (ms) | out tok/s |
|---|---|---|---|---|---|---|
| none | 513 | 13,281 | 0% | 23,088 / 30,311 | 74.8 | 228 |
| spec | 428 | 12,614 | 0% | 32,370 / 43,931 | 84.6 | 166 |
| prefix | 787 | 6,850 | 55.7% | 866 / 34,093 | 63.0 | 391 |
| prefix + session | 795 | 6,260 | 59.5% | 1,258 / 28,920 | 71.0 | 388 |
| prefix + reuse | 789 | 6,414 | 58.5% | 811 / 32,312 | 63.8 | 387 |
| prefix + spec | 572 | 5,723 | 59.9% | 2,047 / 46,053 | 132.0 | 272 |
| prefix + session + reuse | 792 | 5,967 | 61.4% | 978 / 28,246 | 69.1 | 397 |
| prefix + session + spec | 595 | 5,120 | 64.4% | 2,018 / 32,444 | 134.3 | 278 |
| prefix + reuse + spec | 575 | 5,262 | 63.3% | 1,521 / 41,748 | 134.9 | 264 |
| prefix + session + reuse + spec | 589 | 4,877 | 66.2% | 2,569 / 34,261 | 133.9 | 280 |
| **prefix + session + host** | **1,228** | 1,002 | 93.9% | **257 / 2,906** | **45.5** | **661** |
| prefix + session + host + reuse | 1,169 | **942** | **94.3%** | 289 / 4,563 | 48.1 | 630 |
| prefix + session + host + spec | 699 | 1,027 | 93.0% | 538 / 11,194 | 131.3 | 350 |
| prefix + session + host + reuse + spec | 694 | 997 | 93.2% | 635 / 11,104 | 133.7 | 352 |

- **Noise.** Three of these configurations were also measured in the pressure sweep
  (M-024): throughput differs by 1–3% and hit rate by under half a point between the two
  runs, TTFT p50 by up to 24% (prefix-only: 866 vs 1,142 ms) and TTFT p99 by up to 26%
  (with the host tier: 2,906 vs 3,913 ms). Latency differences of that size between rows
  mean nothing.
- **Decomposition** (the chart's path): no cache 13,281 prefill tokens per call → prefix
  cache 6,850 → + retention 6,260 → + host tier 1,002 → + reuse 942. The prefix cache
  halves the work and still thrashes; the host tier removes 84% of what is left;
  retention and reuse are 9% and 6% effects.
- **What the best valid combination is.** prefix + session + host: highest throughput,
  lowest TTFT at the median and the tail, lowest TPOT. Adding reuse saves another 6% of
  prefill tokens and gains nothing measurable in latency or throughput (630 vs 661 tok/s
  is at the edge of the noise, in the wrong direction), for an approximation that changes
  ~3% of next-token predictions (M-026). Adding speculation halves the throughput.
- **Interactions** (`bench/readme_tables.py`, "interactions"; relative to what both
  techniques need):
  - *retention × reuse*: −590 and −435 prefill tokens per call alone, −883 together
    (143 short of additive, 2% of the base: inside the noise).
  - *host tier × reuse*: −5,258 and −292 alone, −5,318 together: reuse adds almost nothing
    once the host tier has restored the hit rate, because what reuse can find (a system
    prompt behind a different preamble) is a few hundred tokens once per session.
  - *host tier × speculation*: the tier lowers TPOT by 25 ms, speculation raises it by 63;
    together +60, i.e. speculation erases the tier's latency gain and most of its
    throughput gain (661 → 350 tok/s). They compete for the same thing: the tier frees
    compute by avoiding prefill, speculation spends compute on verification rows.
  - *anything × speculation at this load*: TPOT 132–135 ms against 45–71 without, in every
    combination; throughput −27% to −47%.
  - The lower "prefill tokens per call" in speculative rows of the collapsing
    configurations (e.g. 5,723 vs 6,850) is not a saving: those servers complete fewer and
    different calls, and the figure depends on which.
- **Failures.** One request in each of two overloaded speculative runs (`spec`,
  `prefix + spec`) ended with a client-side `ServerDisconnectedError`, with nothing in the
  server's log; a third speculative run of the same kind had none. Not explained. Two
  combinations with host tier and reuse first crashed the engine (B-009) and were
  re-measured after the fix; the table has the re-measured rows.
- **Against the baselines** the comparison is M-024's: at 96 sessions vLLM with its native
  offloading delivers 809 tok/s and a TTFT of 206 / 1,091 ms, SGLang with its hierarchical
  cache 772 tok/s and 153 / 1,409 ms, full-stack minfer 641–661 tok/s and 257–266 /
  2,906–3,913 ms while computing 985–1,002 prefill tokens per call against their 1,545 and
  1,702. `vllm-continuum` was not installed: its repository describes itself as preview
  code that omits the paper's TTL estimation, and does not state which vLLM release it is
  based on (the paper's experiments used vLLM 0.10.2, against the 0.34.0 pinned here), so
  it would have been neither Continuum as published nor a like-for-like second vLLM.

### B-009 A reuse copy ran for a sequence that had just been preempted (fixed)

- **Symptom.** `prefix+session+host+reuse` at 96 sessions: the engine process died with
  `IndexError: index 640 is out of bounds for axis 0 with size 0` in `_run_copies`, and
  251 requests failed.
- **Root cause.** The scheduler queues a sequence's KV copy and then allocates blocks for
  its next prefill chunk in the same pass. If that allocation fails with nothing left to
  evict, the victim of preemption is the youngest running sequence — possibly this one.
  Its block table is then gone while its copy is still in the step's plan. The existing
  invariant "a victim is never a sequence already selected for this step" (D-015) did not
  cover work queued *before* the failing allocation of the same sequence.
- **Fix.** After planning, copies of sequences that are no longer running are dropped and
  their pinned sources released; and reuse is not planned again for a sequence that has
  been preempted, since the pinned sources may be exactly the memory it lacked.
  Regression test `test_a_sequence_preempted_after_its_copy_was_queued` (fails before,
  passes after). Only configurations with `--position-independent-reuse` were exposed.

### M-029 Final re-measurement, and the Triton decode kernel loses its place as the default

**Final reruns on the final code** (Phase 17): the three-engine matrix at 1.5B with the
baselines' audit-selected flags; minfer's rows at 7B; the 1.5B ablations; the 1.5B
micro-benchmarks (`make bench-micro`). The first pass of these reruns exposed the next
finding, after which everything labelled "default" was measured once more.

**In-engine A/B of the decode kernels.** The ablation "FlashInfer decode instead of Triton"
had always been measured against FlashInfer's plain decode path. With the tensor-core path
(M-019) it came out *faster* than the default: 4,691 vs 4,199 tok/s saturated at 1.5B.
Checked at 7B on identical traces (`bench/unified_batches.py --experiment decode_kernel`
→ `bench/results/decode_kernel_7b.json`):

| 7B, agent workload | 4 req/s | 16 req/s | "max" |
|---|---|---|---|
| Triton decode: TPOT p50 (ms) | 7.29 | 9.67 | 26.81 |
| FlashInfer decode: TPOT p50 (ms) | 7.18 | 8.84 | 23.97 |
| Triton decode: output tok/s | 256 | 743 | 1,852 |
| FlashInfer decode: output tok/s | 256 | 746 | 2,046 |

- **The custom kernel underperforms the library kernel in the engine at both model sizes**
  (−10% saturated throughput), although in isolation it matches or beats it at most shapes
  (M-019). Not determined: whether the difference is ragged context lengths (the
  microbenchmark uses equal ones), the per-step upload of a padded `[batch, max_pages]`
  block table, or the kernels' behaviour at group size 7 (the microbenchmark is group
  size 6).
- **Decision.** `decode_kernel="auto"` now selects FlashInfer's kernel. The Triton kernel
  stays: tested, documented, selectable, and reported as slower. This reverses D-036.
- **What was re-measured on the new default:** minfer's matrix rows at 7B and 1.5B, the
  1.5B ablations (their "Triton decode kernel instead" row is now the option, not the
  default), the micro-benchmarks, and the full stack under memory pressure
  (`minfer/session_host_fi` in `pressure_7b.json`).
- **What was not, and is labelled so:** the pressure sweep's three minfer rows, the
  composition study, the serve-level speculation benchmark and the unified-pass A/B were
  measured with the Triton kernel, which their scripts now request explicitly. Their
  conclusions are comparisons within one engine configuration and do not depend on it; the
  absolute numbers do.

**Final numbers** (all from the committed results files):

| | minfer | vLLM | SGLang |
|---|---|---|---|
| 7B agent, 1–16 req/s: TTFT p50 (ms) | 47–58 | 78–81 | 57–67 |
| 7B agent, 1–16 req/s: TPOT p50 (ms) | 6.62–8.95 | 6.26–7.88 | 6.27–8.20 |
| 7B agent, saturated tok/s | 2,049 | 2,258 | 2,264 |
| 1.5B agent, 1–16 req/s: TTFT p50 (ms) | 37–40 | 62–65 | 47–55 |
| 1.5B agent, 1–16 req/s: TPOT p50 (ms) | 2.49–2.89 | 2.41–2.50 | 2.25–2.59 |
| 1.5B agent, saturated tok/s | 4,744 | 4,890 | 5,741 |
| real trace, 96 sessions, host tiers: tok/s | 759 | 809 | 772 |
| real trace, 96 sessions, host tiers: TTFT p50 / p99 (ms) | 169 / 1,270 | 206 / 1,091 | 153 / 1,409 |
| real trace, 96 sessions, host tiers: prefill tokens / call | 1,041 | 1,545 | 1,702 |
| real trace, 128 sessions, host tiers: tok/s | 684 | 765 | 694 |

The decode gap of M-022 is smaller with the default kernel (TPOT 1.06–1.14× of vLLM at
1–16 req/s at 7B instead of 1.06–1.24×; saturated 0.91× instead of 0.81×) and is still
there. With host tiers on the real trace, minfer is at 0.94× / 0.94× / 0.89× of vLLM's
throughput at 64 / 96 / 128 sessions and level with SGLang at 96 and 128, while computing
30–39% fewer prefill tokens per call than either past capacity.

**Phase 17 acceptance.** Every number in the README is generated from a committed results
file by `bench/readme_tables.py` (`--check` runs in CI), or quoted from one in prose that
was re-checked against the final files. Not re-measured at the end, and so older than the
final commit: the fairness audit (Phase 13 code, 1.5B), the Phase 12 reproduction check,
and the Triton-kernel-era benchmarks listed above; each results file since Phase 14
records its commit.

### M-030 The Triton-era benchmarks, re-measured on the default decode kernel

M-029 left four benchmarks measured with the Triton decode kernel after FlashInfer's had
become the default: the pressure sweep's minfer rows, the composition study, the
serve-level speculation benchmark and the unified-pass A/B. Their scripts pinned
`--decode-kernel triton` to keep the files reproducible. That left the README's headline
table (default kernel) and its decomposition (Triton kernel) on different engine
configurations. The pins are removed and all four were run again on commit `560098f`,
in one unattended chain with nothing else on the machine: 10 + 14 + 8 + 2 server launches,
0 failed requests. The results files are replaced (the earlier ones are in git history at
`4df53e0`); `pressure_7b.json` keeps the baselines' rows from M-024 (`pressure.py
--resume --remeasure`), and its separate `minfer/session_host_fi` configuration is gone
because `minfer/session_host` is now that configuration.

**Pressure sweep, minfer rows** (`bench/results/pressure_7b.json`; baselines from M-024):

| sessions | configuration | hit rate | prefill tokens / call | TTFT p50 / p99 (ms) | TPOT p50 (ms) | out tok/s |
|---|---|---|---|---|---|---|
| 32 | session-aware | 94.0% | 986 | 100 / 410 | 9.7 | 382 |
| 64 | LRU | 88.5% | 1,899 | 126 / 1,240 | 21.4 | 625 |
| 64 | session-aware | 91.0% | 1,491 | 121 / 1,207 | 20.2 | 655 |
| 64 | + host tier | 94.2% | 966 | 117 / 449 | 19.5 | 687 |
| 96 | LRU | 56.6% | 6,958 | 624 / 33,852 | 50.5 | 406 |
| 96 | session-aware | 60.4% | 6,332 | 985 / 24,202 | 57.1 | 424 |
| 96 | + host tier | 94.1% | 1,013 | 159 / 570 | 33.9 | 772 |
| 128 | LRU | 34.7% | 9,638 | 13,163 / 55,075 | 73.6 | 347 |
| 128 | session-aware | 44.8% | 8,253 | 12,516 / 41,633 | 77.2 | 393 |
| 128 | + host tier | 89.5% | 1,696 | 2,811 / 10,983 | 47.5 | 693 |

- Session-aware retention against LRU: +2.5 / +3.9 / +10.1 points of hit rate at
  64 / 96 / 128 sessions, 21% / 9% / 14% fewer prefill tokens, TTFT p99 3% / 29% / 24%
  lower, throughput +4–13%. Still a small gain that does not prevent the collapse.
- The host tier: 1.82× and 1.76× the throughput of the same engine without it at 96 and
  128 sessions.
- Full stack against the baselines with their offloading (vLLM 717 / 809 / 765 tok/s,
  SGLang 729 / 772 / 694): **0.96× / 0.95× / 0.91× of vLLM, 0.94× / 1.00× / 1.00× of
  SGLang**; 34% / 31% fewer prefill tokens per call than vLLM at 96 / 128 and 41% / 32%
  fewer than SGLang; TPOT 1.18× / 1.25× / 1.05× of vLLM's. vLLM is still the fastest
  system. M-029's run of the same configuration gave 672 / 759 / 684 tok/s: the
  throughput repeats to 2%.
- **TTFT p99 does not repeat.** The full stack at 96 sessions: 570 ms in this sweep,
  1,252 ms in the composition run below, 1,270 ms in M-029's run — the same configuration
  three times. The README table shows 570 ms, in bold as the lowest of the three engines,
  because the table is generated from the file; the prose under it says that this is not
  a finding. The baselines were each measured once, so their p99s carry the same
  uncertainty.

**Composition study** (`bench/results/composition_7b.json`, 96 sessions):

| techniques | calls | prefill tokens / call | hit rate | TTFT p50 / p99 (ms) | TPOT p50 (ms) | out tok/s |
|---|---|---|---|---|---|---|
| none | 537 | 13,428 | 0% | 20,820 / 28,444 | 70.5 | 242 |
| spec | 423 | 12,634 | 0% | 32,670 / 46,279 | 85.2 | 171 |
| prefix | 835 | 7,083 | 55.8% | 634 / 34,705 | 52.5 | 404 |
| prefix + session | 849 | 6,217 | 61.2% | 878 / 22,946 | 53.4 | 427 |
| prefix + reuse | 829 | 6,674 | 58.3% | 778 / 31,019 | 53.7 | 405 |
| prefix + spec | 573 | 5,500 | 61.4% | 1,732 / 42,780 | 132.6 | 260 |
| prefix + session + reuse | 838 | 6,160 | 61.5% | 875 / 23,373 | 63.1 | 414 |
| prefix + session + spec | 599 | 4,982 | 65.2% | 2,536 / 33,321 | 133.4 | 279 |
| prefix + reuse + spec | 567 | 5,401 | 61.9% | 1,495 / 43,232 | 134.5 | 260 |
| prefix + session + reuse + spec | 587 | 4,877 | 66.0% | 2,066 / 33,427 | 135.8 | 284 |
| **prefix + session + host** | **1,366** | 1,038 | 93.9% | **174 / 1,252** | **38.3** | **758** |
| prefix + session + host + reuse | 1,343 | **975** | **94.2%** | 189 / 1,726 | 38.4 | 740 |
| prefix + session + host + spec | 702 | 1,030 | 93.0% | 546 / 10,185 | 131.2 | 351 |
| prefix + session + host + reuse + spec | 700 | 976 | 93.4% | 634 / 11,105 | 131.0 | 353 |

- **Every conclusion of M-028 holds**: decomposition 13,428 → 7,083 (prefix) → 6,217
  (+ retention) → 1,038 (+ host tier) → 975 (+ reuse); best valid combination the three
  exact KV techniques; reuse saves 1–6% of prefill tokens and nothing measurable in
  latency or throughput (740 vs 758 tok/s); speculation costs 29–54% of throughput in
  every combination and takes the host tier's gain back (758 → 351 tok/s); no pair
  compounds.
- **What changed:** absolute throughput (the best combination 661 → 758 tok/s, +15%; the
  collapsing ones +3–10%), and speculation's relative cost with the host tier (−47% →
  −54%): speculative rows have a TPOT of 131–136 ms whatever the decode kernel, so the
  faster non-speculative rows widen the gap.
- **Noise**, from the three configurations also in the sweep above: throughput 404 / 427 /
  758 here against 406 / 424 / 772 (within 2%); hit rate within a point; TTFT p50 within
  12%; TTFT p99 as above.
- **The two `ServerDisconnectedError` failures of M-028 did not recur**: 0 failed requests
  in all 14 runs, including the two configurations that had one each. They remain
  unexplained; the only evidence is that they are not deterministic.

**Speculation through the server** (`bench/results/speculation_serve_7b.json`):

| configuration | TPOT p50 (ms) at 1 / 4 / 8 req/s | vs off | TPOT p99 (ms) | E2E p50 (ms) | drafts accepted |
|---|---|---|---|---|---|
| no speculation | 6.60 / 7.21 / 7.68 | — | 7.7 / 8.7 / 10.8 | 444 / 489 / 532 | — |
| n=3, ≤ 4 drafted | 5.68 / 6.88 / 8.01 | 0.86× / 0.95× / 1.04× | 8.7 / 13.4 / 17.2 | 391 / 475 / 587 | 62–64% |
| n=2, ≤ 4 drafted | 5.85 / 6.96 / 8.73 | 0.89× / 0.96× / 1.14× | 9.0 / 13.7 / 16.9 | 403 / 496 / 631 | 54–56% |
| n=4, ≤ 4 drafted | 6.03 / 6.89 / 8.27 | 0.91× / 0.95× / 1.08× | 8.6 / 12.3 / 16.4 | 386 / 478 / 602 | 64–66% |
| **n=3, ≤ 8 drafted** | **5.60 / 6.65 / 8.02** | **0.85× / 0.92× / 1.04×** | 8.3 / 12.9 / 15.5 | 371 / 463 / 550 | 52–54% |
| n=5, ≤ 8 drafted | 5.96 / 6.78 / 7.96 | 0.90× / 0.94× / 1.04× | 8.6 / 12.7 / 15.2 | 386 / 478 / 568 | 56–57% |
| n=3, ≤ 4, never paused | 5.66 / 6.76 / 8.11 | 0.86× / 0.94× / 1.06× | 8.8 / 13.6 / 16.8 | 390 / 471 / 590 | 61–64% |
| n=3, ≤ 4, eager verification | 6.34 / 8.02 / 11.14 | 0.96× / 1.11× / 1.45× | 11.9 / 16.2 / 19.9 | 420 / 608 / 755 | 61–64% |

Same result as M-027: TPOT p50 15% and 8% lower at 1 and 4 req/s with n=3, ≤ 8 drafted.
At 8 req/s it is now 4% *higher* in every graph-verified configuration (M-027: +1% for
the best one), because the no-speculation row got faster at that rate (7.88 → 7.68 ms)
and the speculative ones did not. The tail is worse in every speculative configuration,
graph verification is what makes it pay, pausing makes no difference.

**Unified prefill+decode passes** (`bench/results/unified_batches_7b.json`):

| | 4 req/s | 16 req/s | "max" |
|---|---|---|---|
| split passes: TPOT p50 (ms) | 7.28 | 8.91 | 24.52 |
| unified: TPOT p50 (ms) | 7.08 | 8.71 | 37.99 |
| split passes: TTFT p50 (ms) | 55.0 | 56.9 | 108.5 |
| unified: TTFT p50 (ms) | 44.7 | 49.9 | 119.7 |
| split passes: output tok/s | 256 | 745 | 2,026 |
| unified: output tok/s | 257 | 747 | 1,455 |

Same shape as M-025: TPOT −2 to −3% and TTFT p50 −7 to −10 ms below saturation, saturated
throughput −28% (was −23% against the slower Triton split path). Off by default.

**What is still older than the final code:** the fairness audit (Phase 13 code, 1.5B),
the Phase 12 reproduction check, the baselines' rows of the pressure sweep (measured
2026-11-21, before this rerun; the baselines' software is unchanged), and the
position-independent-reuse quality study and the host-tier cost measurement, which were
not run again.
