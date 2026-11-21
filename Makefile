# FlashInfer's JIT shells out to `ninja`, which lives in the venv's bin directory.
export PATH := $(CURDIR)/.venv/bin:$(PATH)

PY := .venv/bin/python

.PHONY: setup env lint format test test-cpu bench bench-agent bench-micro bench-ablations bench-audit bench-real plots

setup:  ## build .venv, .venv-vllm and .venv-sglang from the pinned lock files
	scripts/setup_envs.sh all

env:  ## regenerate bench/ENV.md
	$(PY) bench/collect_env.py

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

test:  ## full suite (needs the GPU and downloaded weights)
	$(PY) -m pytest

test-cpu:  ## the subset CI runs
	$(PY) -m pytest -m "not gpu"

bench: bench-micro bench-agent bench-ablations bench-audit bench-real plots  ## every benchmark behind the README (about a day of GPU time)
	$(PY) -m bench.readme_tables

bench-micro:  ## engine-level benchmarks and equivalence checks (minfer; vLLM only for the fixture)
	.venv-vllm/bin/python bench/vllm_greedy_fixture.py
	$(PY) -m bench.vllm_equivalence
	$(PY) -m bench.decode_latency
	$(PY) -m bench.prefill_latency
	$(PY) -m bench.chunked_prefill
	$(PY) -m bench.prefix_cache_equivalence
	$(PY) -m bench.triton_kernel
	$(PY) -m bench.speculation
	$(PY) -m bench.gsm8k
	$(PY) -m bench.profile_engine --tag after
	$(PY) -m bench.tensor_parallel_check
	bench/serve_compat.sh

bench-agent:  ## minfer vs vLLM vs SGLang on the agent workload and ShareGPT, all rates, both models
	$(PY) -m bench.run_matrix --model 7b
	$(PY) -m bench.run_matrix --model 1.5b

bench-ablations:  ## the agent workload with one feature changed at a time
	$(PY) -m bench.ablations --model 1.5b

bench-audit:  ## the baselines under their documented tuning flags; token-id prompts
	$(PY) -m bench.fairness_audit --model 1.5b

PRESSURE := --duration 360 --warmup 120 --ramp 30

bench-real:  ## real coding-agent traces at 7B: pressure sweep, host tier, reuse quality, speculation, composition
	$(PY) -m bench.real_traces compile --model 7b --out bench/traces/real_7b.jsonl
	$(PY) -m bench.real_traces stats bench/traces/real_7b.jsonl --out bench/results/real_trace_stats_7b.json
	$(PY) -m bench.host_tier_cost
	$(PY) -m bench.pressure $(PRESSURE) --concurrency 64 96 128
	$(PY) -m bench.pressure $(PRESSURE) --resume --concurrency 32 --configs minfer/session vllm/default sglang/default
	$(PY) -m bench.pic_quality
	$(PY) -m bench.speculation_serve
	$(PY) -m bench.unified_batches
	$(PY) -m bench.composition $(PRESSURE) --concurrency 96 --spec-flags="--ngram-size 3 --num-draft-tokens 8"

plots:  ## regenerate bench/plots/ from bench/results/
	$(PY) -m bench.plot
