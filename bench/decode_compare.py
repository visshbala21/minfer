"""Decode step time of minfer and vLLM at 7B, measured the same way in both (Phase 18).

    .venv/bin/python -m bench.decode_compare --engine minfer
    .venv-vllm/bin/python bench/decode_compare.py --engine vllm

For each (batch size, context length), ``batch`` random-token prompts of ``context``
tokens are generated greedily with ``ignore_eos`` twice through the engine's offline API:
once for ``short`` and once for ``short + EXTRA`` output tokens. The prompts differ between
the two calls and between points (no prefix-cache hits). The time per decode step is the
difference of the two wall-clock times divided by ``EXTRA``: prefill, start-up and the
final read-back cancel, and what remains is the steady-state cost of one step of
``batch`` sequences, including each engine's own scheduling and sampling. This holds only
if every sequence is still decoding when the last prompt has been prefilled — both
engines interleave prefill chunks with the decoding of sequences admitted earlier — so
``short`` is ``32`` plus the number of 2,048-token chunks the prompts take (an upper
bound on the steps prefill spans in either engine). Each point is measured ``REPEATS``
times; the median is reported.

Run once per engine, in that engine's environment (this file imports nothing from minfer
at module level). Results are merged into ``bench/results/decode_compare_7b.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen2.5-7B-Instruct"
POINTS = ((1, 1024), (8, 1024), (32, 1024), (64, 1024), (128, 1024))
POINTS += ((1, 8192), (16, 8192), (32, 8192), (64, 8192))
EXTRA = 256
REPEATS = 3
OUT = Path("bench/results/decode_compare_7b.json")


def _prompts(rng: random.Random, batch: int, context: int) -> list[list[int]]:
    return [[rng.randrange(1000, 100_000) for _ in range(context)] for _ in range(batch)]


class _Minfer:
    def __init__(self, overrides: dict[str, Any]) -> None:
        from minfer.engine import Engine, EngineConfig

        self.engine = Engine.from_pretrained(MODEL, EngineConfig(**overrides))

    def generate(self, prompts: list[list[int]], tokens: int) -> None:
        from minfer.sampling import SamplingParams

        params = SamplingParams(max_tokens=tokens, ignore_eos=True)
        outputs = self.engine.generate(prompts, params)
        assert all(len(output) == tokens for output in outputs)

    def version(self) -> str:
        root = Path(__file__).resolve().parent.parent

        def git(*args: str) -> str:
            return subprocess.run(["git", *args], cwd=root, capture_output=True).stdout.decode()

        dirty = " (dirty tree)" if git("status", "--porcelain", "-uno").strip() else ""
        return git("rev-parse", "HEAD").strip() + dirty


class _Vllm:
    def __init__(self, overrides: dict[str, Any]) -> None:
        from vllm import LLM

        self.llm = LLM(MODEL, dtype="bfloat16", max_model_len=32768, **overrides)

    def generate(self, prompts: list[list[int]], tokens: int) -> None:
        from vllm import SamplingParams, TokensPrompt

        params = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False
        )
        assert all(len(o.outputs[0].token_ids) == tokens for o in outputs)

    def version(self) -> str:
        import vllm

        return vllm.__version__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", choices=("minfer", "vllm"), required=True)
    parser.add_argument("--name", default=None, help="result key (default: the engine)")
    parser.add_argument(
        "--set", nargs="*", default=[], help="engine config overrides, key=json-value"
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    overrides = {key: json.loads(value) for key, value in (s.split("=", 1) for s in args.set)}
    runner = _Minfer(overrides) if args.engine == "minfer" else _Vllm(overrides)
    rng = random.Random(0)
    runner.generate(_prompts(rng, 8, 512), EXTRA)  # warm-up
    rows = []
    for batch, context in POINTS:
        short = 32 + -(-batch * context // 2048)
        steps = []
        for _ in range(REPEATS):
            times = {}
            for tokens in (short, short + EXTRA):
                prompts = _prompts(rng, batch, context)
                start = time.perf_counter()
                runner.generate(prompts, tokens)
                times[tokens] = time.perf_counter() - start
            steps.append((times[short + EXTRA] - times[short]) / EXTRA * 1e3)
        row = {
            "batch_size": batch,
            "context_tokens": context,
            "short_tokens": short,
            "step_ms": statistics.median(steps),
            "step_ms_runs": steps,
        }
        rows.append(row)
        print(args.engine, json.dumps(row), flush=True)
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    results.setdefault("model", MODEL)
    results.setdefault("method", __doc__.split("\n\n")[2].replace("\n", " "))
    results.setdefault("engines", {})[args.name or args.engine] = {
        "engine": args.engine,
        "version": runner.version(),
        "overrides": overrides,
        "rows": rows,
    }
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
