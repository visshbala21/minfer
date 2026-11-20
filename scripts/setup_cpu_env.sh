#!/usr/bin/env bash
# Build a CPU-only environment for linting and the CPU-runnable unit tests (what CI uses).
#
# It installs exactly the versions in uv.lock, with two differences: the CUDA-only stack
# (flashinfer-python, triton, and the nvidia-* runtime wheels the CUDA build of torch pulls
# in) is left out, and torch is the CPU build of the same version. The full lock is passed
# as constraints so torch's own dependencies stay pinned.
#
# Usage: scripts/setup_cpu_env.sh [venv-path]      (default: .venv)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-${ROOT}/.venv}"
TORCH_VERSION="2.14.0"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

cd "${ROOT}"
uv venv --allow-existing --python 3.12 "${VENV}"
uv export --frozen --no-hashes --no-emit-project -o "${TMP}/constraints.txt" >/dev/null
uv export --frozen --no-hashes --no-emit-project \
    --prune torch --prune triton --prune flashinfer-python \
    -o "${TMP}/requirements.txt" >/dev/null
uv pip install --python "${VENV}/bin/python" \
    -r "${TMP}/requirements.txt" "torch==${TORCH_VERSION}+cpu" \
    -c "${TMP}/constraints.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match
uv pip install --python "${VENV}/bin/python" --no-deps -e "${ROOT}"
