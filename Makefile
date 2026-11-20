# FlashInfer's JIT shells out to `ninja`, which lives in the venv's bin directory.
export PATH := $(CURDIR)/.venv/bin:$(PATH)

PY := .venv/bin/python

.PHONY: setup env lint format test test-cpu bench bench-agent bench-micro bench-ablations bench-audit plots

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

bench: bench-micro bench-agent bench-ablations bench-audit plots  ## every benchmark behind the README
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

bench-agent:  ## minfer vs vLLM vs SGLang on the agent workload and ShareGPT, all rates
	$(PY) -m bench.run_matrix --model 1.5b

bench-ablations:  ## the agent workload with one feature changed at a time
	$(PY) -m bench.ablations --model 1.5b

bench-audit:  ## the baselines under their documented tuning flags; token-id prompts
	$(PY) -m bench.fairness_audit --model 1.5b

plots:  ## regenerate bench/plots/ from bench/results/
	$(PY) -m bench.plot
