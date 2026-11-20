# FlashInfer's JIT shells out to `ninja`, which lives in the venv's bin directory.
export PATH := $(CURDIR)/.venv/bin:$(PATH)

PY := .venv/bin/python

.PHONY: setup env lint format test test-cpu

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
