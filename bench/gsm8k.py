"""GSM8K accuracy and capacity/throughput under FP8 (Phase 9).

    .venv/bin/python -m bench.gsm8k

Runs the first 200 problems of the GSM8K test split through the engine with greedy
decoding, once per precision setting (bf16 baseline, FP8 KV cache, FP8 weights, both), and
records exact-match accuracy, how often each setting's final answer agrees with the
baseline's, KV-cache capacity, and generation throughput. The questions, prompts and
decoding parameters are identical across settings.
Writes ``bench/results/gsm8k_fp8.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_PROBLEMS = 200
MAX_TOKENS = 512
INSTRUCTION = (
    "Solve the following math problem. Reason step by step, then give the final numeric "
    "answer on the last line in the form '#### <number>'."
)
SETTINGS: dict[str, dict[str, str]] = {
    "bf16": {},
    "fp8_kv": {"kv_cache_dtype": "fp8"},
    "fp8_weights": {"weight_dtype": "fp8"},
    "fp8_weights_and_kv": {"kv_cache_dtype": "fp8", "weight_dtype": "fp8"},
}

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_number(text: str) -> float | None:
    """A number written like ``1,234.50`` as a float; ``None`` if ``text`` holds none."""
    match = _NUMBER.search(text)
    if match is None:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def extract_answer(completion: str) -> float | None:
    """The model's final answer: the number after the last ``####``, else the last number."""
    if "####" in completion:
        answer = parse_number(completion.rsplit("####", 1)[1])
        if answer is not None:
            return answer
    numbers = _NUMBER.findall(completion)
    return parse_number(numbers[-1]) if numbers else None


def gold_answer(solution: str) -> float:
    """The reference answer of a GSM8K record (the number after ``####``)."""
    answer = parse_number(solution.rsplit("####", 1)[1])
    if answer is None:
        raise ValueError(f"no reference answer in {solution[-80:]!r}")
    return answer


def is_correct(predicted: float | None, gold: float) -> bool:
    """Exact match up to float formatting."""
    return predicted is not None and abs(predicted - gold) < 1e-6


def load_problems(count: int) -> list[dict[str, Any]]:
    """The first ``count`` problems of the GSM8K test split."""
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split="test").select(range(count))
    return [{"question": row["question"], "gold": gold_answer(row["answer"])} for row in rows]


def evaluate(
    name: str, overrides: dict[str, str], prompts: list[list[int]], tokenizer: Any
) -> dict:
    """Run every prompt through an engine built with ``overrides``."""
    import torch

    from minfer.engine import Engine, EngineConfig
    from minfer.sampling import SamplingParams

    torch.cuda.empty_cache()
    free_before, _ = torch.cuda.mem_get_info()
    # Chunked prefill off, as in bench/decode_latency.py (steady-state decode is timed).
    engine = Engine.from_pretrained(MODEL, EngineConfig(prefill_chunk_tokens=None, **overrides))
    tensors = [*engine.model.parameters(), *engine.model.buffers()]
    weights_bytes = sum(t.numel() * t.element_size() for t in tensors)
    params = SamplingParams(max_tokens=MAX_TOKENS)
    engine.generate(prompts[:8], params)  # warm-up
    start = time.perf_counter()
    outputs = engine.generate(prompts, params)
    seconds = time.perf_counter() - start
    texts = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    # Steady-state decode-step latency at fixed batch sizes (512-token contexts).
    from bench.decode_latency import measure

    decode = [measure(engine, batch, 512, steps=100, warmup=10) for batch in (1, 64, 256)]
    generated = sum(map(len, outputs))
    result = {
        "setting": name,
        "engine_overrides": overrides,
        "texts": texts,
        "generated_tokens": generated,
        "wall_seconds": seconds,
        "output_tokens_per_second": generated / seconds,
        "truncated_at_max_tokens": sum(len(o) == MAX_TOKENS for o in outputs),
        "decode_step": decode,
        "weight_bytes": weights_bytes,
        "kv_blocks": engine.blocks.num_usable,
        "kv_capacity_tokens": engine.blocks.num_usable * engine.config.block_size,
        "kv_store_bytes": engine.store.nbytes,
        "kv_bytes_per_token": engine.store.nbytes
        / (engine.blocks.num_blocks * engine.config.block_size),
        "gpu_free_bytes_before_load": free_before,
    }
    del engine
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("bench/results/gsm8k_fp8.json"))
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    problems = load_problems(NUM_PROBLEMS)
    prompts = []
    for problem in problems:
        messages = [{"role": "user", "content": f"{INSTRUCTION}\n\n{problem['question']}"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(tokenizer(text, add_special_tokens=False)["input_ids"])

    results: dict[str, Any] = {
        "model": MODEL,
        "dataset": "openai/gsm8k (main), test split, first 200 problems",
        "decoding": f"greedy, max_tokens={MAX_TOKENS}, zero-shot chat prompt",
        "instruction": INSTRUCTION,
        "settings": {},
    }
    baseline: list[float | None] | None = None
    for name in args.settings:
        run = evaluate(name, SETTINGS[name], prompts, tokenizer)
        answers = [extract_answer(text) for text in run.pop("texts")]
        correct = [is_correct(a, p["gold"]) for a, p in zip(answers, problems, strict=True)]
        if baseline is None:
            baseline = answers
        run["correct"] = sum(correct)
        run["accuracy"] = sum(correct) / len(problems)
        run["no_answer_extracted"] = sum(a is None for a in answers)
        run["same_final_answer_as_first_setting"] = sum(
            a == b for a, b in zip(answers, baseline, strict=True)
        )
        results["settings"][name] = run
        print(json.dumps({k: v for k, v in run.items() if k != "engine_overrides"}))
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
