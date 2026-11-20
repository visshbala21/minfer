#!/usr/bin/env bash
# Phase 5 acceptance: run vLLM's own benchmark client (`vllm bench serve`) against a minfer
# server, through both the completions and the chat-completions endpoints.
#
#   bench/serve_compat.sh
#
# Results (vLLM's JSON result files) go to bench/results/serve_compat/. This checks protocol
# compatibility under load; it is not the performance comparison (see bench/agent_workload.py).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PATH="${ROOT}/.venv/bin:${PATH}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8011}"
OUT="bench/results/serve_compat"
mkdir -p "${OUT}" bench/logs

.venv/bin/python -m minfer.cli serve --model "${MODEL}" --port "${PORT}" --log-level warning \
    > bench/logs/serve_compat_server.log 2>&1 &
SERVER=$!
trap 'kill ${SERVER} 2>/dev/null || true; wait ${SERVER} 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
    curl -sf "localhost:${PORT}/health" > /dev/null && break
    kill -0 "${SERVER}" 2>/dev/null || { echo "server exited during start-up" >&2; exit 1; }
    sleep 1
done

run() {  # run <name> <backend> <endpoint>
    .venv-vllm/bin/vllm bench serve \
        --backend "$2" --endpoint "$3" --base-url "http://localhost:${PORT}" --model "${MODEL}" \
        --dataset-name random --random-input-len 512 --random-output-len 128 \
        --num-prompts 500 --request-rate 30 --ignore-eos --temperature 0 --seed 0 \
        --save-result --result-dir "${OUT}" --result-filename "$1.json" \
        > "bench/logs/serve_compat_$1.log" 2>&1
    grep -E "Successful requests|Failed requests|Output token throughput|Median TTFT|Median TPOT" \
        "bench/logs/serve_compat_$1.log" | sed "s/^/[$1] /"
}

run completions openai /v1/completions
run chat openai-chat /v1/chat/completions
curl -s "localhost:${PORT}/metrics" | grep -v '^#' > "${OUT}/metrics_after.txt"
