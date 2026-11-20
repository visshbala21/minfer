"""Generate the vLLM greedy-decoding fixture used by the Phase 2 equivalence test.

Run inside the vLLM virtualenv:

    .venv-vllm/bin/python bench/vllm_greedy_fixture.py

Prompts are tokenized here (chat template applied) and passed to vLLM as token ids, so the
fixture pins both sides of the comparison to the same input ids. Sampling parameters are
given explicitly: the checkpoint's ``generation_config.json`` asks for sampling with a
repetition penalty, which would otherwise become vLLM's default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_TOKENS = 128

PROMPTS = [
    "Explain how a hash map handles collisions.",
    "Write a Python function that checks whether a string is a palindrome.",
    "What were the main causes of the First World War?",
    "Translate into French: 'The train to Lyon leaves at half past seven tomorrow morning.'",
    "用中文简要解释什么是量子纠缠。",
    "Give me a recipe for a simple tomato soup.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "What is the derivative of x^3 * sin(x)? Show the steps.",
    "Write a SQL query that returns the top five customers by total order value.",
    "Describe the difference between TCP and UDP.",
    'Return a JSON object with the keys "name", "age" and "city" for a fictional person.',
    "Why is the sky blue?",
    "Write a limerick about a programmer who loves coffee.",
    "List the planets of the solar system in order from the sun, one per line.",
    "Explain the rules of chess castling.",
    "A store sells apples at 3 for $2. How much do 18 apples cost? Explain.",
    "Write a bash one-liner that counts the lines in every .py file under the current directory.",
    "What is the capital of Australia, and why was it chosen?",
    "Explain what a mutex is to a five-year-old.",
    "Draft a short, polite email asking a colleague to review a pull request.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/vllm_greedy.json"))
    args = parser.parse_args()

    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = []
    for text in PROMPTS:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        prompt_ids.append(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.5, seed=0)
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        max_tokens=MAX_TOKENS,
        skip_special_tokens=False,
    )
    # One request at a time: the reference outputs then do not depend on vLLM's batching.
    serial = [llm.generate([{"prompt_token_ids": ids}], params)[0].outputs[0] for ids in prompt_ids]
    # And all at once, to measure how far vLLM agrees with itself across batch compositions.
    batched = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], params)
    cases = [
        {
            "prompt": text,
            "prompt_token_ids": ids,
            "output_token_ids": list(one.token_ids),
            "finish_reason": one.finish_reason,
            "output_token_ids_batched": list(many.outputs[0].token_ids),
        }
        for text, ids, one, many in zip(PROMPTS, prompt_ids, serial, batched, strict=True)
    ]
    fixture = {
        "model": MODEL,
        "vllm_version": vllm.__version__,
        "dtype": "bfloat16",
        "max_tokens": MAX_TOKENS,
        "sampling": "greedy (temperature=0, repetition_penalty=1)",
        "batching": "output_token_ids: one request at a time; output_token_ids_batched: all 20",
        "cases": cases,
    }
    args.out.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
