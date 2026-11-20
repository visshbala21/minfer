# FlashInfer's JIT shells out to `ninja`, which lives in the venv's bin directory.
export PATH := $(CURDIR)/.venv/bin:$(PATH)

PY := .venv/bin/python

.PHONY: setup env lint format test test-cpu bench bench-agent bench-micro plots

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

bench: bench-micro bench-agent plots  ## every benchmark behind the README, then the charts

bench-micro:  ## engine-level benchmarks and equivalence checks (minfer only)
	$(PY) -m bench.decode_latency
	$(PY) -m bench.chunked_prefill
	$(PY) -m bench.prefix_cache_equivalence

bench-agent:  ## minfer vs vLLM vs SGLang on the agent workload and ShareGPT, all rates
	$(PY) -m bench.run_matrix

plots:  ## regenerate bench/plots/ from bench/results/
	$(PY) -m bench.plot
