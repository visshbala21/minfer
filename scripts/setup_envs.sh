#!/usr/bin/env bash
# Build the isolated virtualenvs used by this repo.
#
#   .venv          minfer itself            (pyproject.toml + uv.lock)
#   .venv-vllm     vLLM baseline            (bench/envs/vllm.txt, fully pinned)
#   .venv-sglang   SGLang baseline          (bench/envs/sglang.txt, fully pinned)
#
# The baselines live in their own environments so their dependency pins can never
# conflict with (or silently change) minfer's.
#
# Usage: scripts/setup_envs.sh [minfer|vllm|sglang|all]      (default: all)
#
# Set MINFER_VENV_ROOT=/fast/local/disk to build the environments there and leave
# symlinks in the repo. Useful when the checkout is on a network filesystem, where
# importing torch from the venv is several times slower.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="3.12"
WHAT="${1:-all}"

# Resolve where environment <name> lives; echo the real path.
env_path() {
    local link="$1" name="$2"
    if [[ -n "${MINFER_VENV_ROOT:-}" ]]; then
        mkdir -p "${MINFER_VENV_ROOT}"
        ln -sfn "${MINFER_VENV_ROOT}/${name}" "${ROOT}/${link}"
        echo "${MINFER_VENV_ROOT}/${name}"
    else
        echo "${ROOT}/${link}"
    fi
}

setup_minfer() {
    local path
    path="$(env_path .venv minfer)"
    echo ">>> minfer env at ${path}"
    (cd "${ROOT}" && UV_PROJECT_ENVIRONMENT="${path}" uv sync --frozen --python "${PYTHON_VERSION}")
}

setup_baseline() {
    local name="$1" path
    path="$(env_path ".venv-${name}" "${name}")"
    echo ">>> ${name} env at ${path}"
    uv venv --allow-existing --python "${PYTHON_VERSION}" "${path}"
    uv pip install --python "${path}/bin/python" -r "${ROOT}/bench/envs/${name}.txt"
}

case "${WHAT}" in
    minfer) setup_minfer ;;
    vllm | sglang) setup_baseline "${WHAT}" ;;
    all)
        setup_minfer
        setup_baseline vllm
        setup_baseline sglang
        ;;
    *)
        echo "unknown target: ${WHAT} (expected minfer|vllm|sglang|all)" >&2
        exit 2
        ;;
esac
