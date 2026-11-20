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
