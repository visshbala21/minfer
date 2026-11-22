"""Command-line entry point: ``minfer generate ...`` (and ``minfer serve ...``)."""

from __future__ import annotations

import argparse
import functools
import sys
import time
from collections.abc import Sequence

from minfer.engine import EngineConfig
from minfer.sampling import SamplingParams

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags that configure the engine; shared by every subcommand."""
    defaults = EngineConfig()
    group = parser.add_argument_group("engine")
    group.add_argument("--model", default=DEFAULT_MODEL, help="local directory or Hub repo id")
    group.add_argument("--block-size", type=int, default=defaults.block_size)
    group.add_argument(
        "--num-blocks", type=int, default=None, help="KV pool size (default: from free memory)"
    )
    group.add_argument("--kv-memory-fraction", type=float, default=defaults.kv_memory_fraction)
    group.add_argument("--kv-gb", type=float, default=None, help="KV cache size in GiB")
    group.add_argument("--max-num-seqs", type=int, default=defaults.max_num_seqs)
    group.add_argument("--max-batched-tokens", type=int, default=defaults.max_batched_tokens)
    group.add_argument("--max-model-len", type=int, default=None)
    group.add_argument(
        "--kv-cache-dtype",
        choices=["auto", "fp8"],
        default=defaults.kv_cache_dtype,
        help="store the KV cache in FP8 (E4M3) with calibrated per-layer scales",
    )
    group.add_argument(
        "--weight-dtype",
        choices=["auto", "fp8"],
        default=defaults.weight_dtype,
        help="run the transformer's linear layers with FP8 weights and activations",
    )
    group.add_argument(
        "--fused-kernels",
        action=argparse.BooleanOptionalAction,
        default=defaults.fused_kernels,
        help="merged projections and fused norm/RoPE/activation kernels",
    )
    group.add_argument(
        "--fused-kv-write",
        action=argparse.BooleanOptionalAction,
        default=defaults.fused_kv_write,
        help="write keys and values into the KV cache from the fused RoPE kernel",
    )
    group.add_argument(
        "--decode-kernel",
        choices=["auto", "flashinfer", "triton"],
        default=defaults.decode_kernel,
        help="paged-attention kernel used for decode steps",
    )
    group.add_argument(
        "--cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=defaults.cuda_graphs,
        help="replay captured CUDA graphs for decode steps",
    )
    group.add_argument(
        "--prefill-graphs",
        action=argparse.BooleanOptionalAction,
        default=defaults.prefill_graphs,
        help="replay captured CUDA graphs for prefill passes that fit a bucket",
    )
    group.add_argument(
        "--async-decode",
        action=argparse.BooleanOptionalAction,
        default=defaults.async_decode,
        help="launch a decode step before the previous step's tokens are read back",
    )
    group.add_argument(
        "--unified-batches",
        action=argparse.BooleanOptionalAction,
        default=defaults.unified_batches,
        help="run a step's prefill and decode work as one joint pass over the weights",
    )
    group.add_argument(
        "--prefill-chunk-tokens",
        type=int,
        default=defaults.prefill_chunk_tokens,
        help="chunked prefill: prefill tokens per step",
    )
    group.add_argument(
        "--no-chunked-prefill",
        dest="prefill_chunk_tokens",
        action="store_const",
        const=None,
        help="prefill whole prompts in one pass",
    )
    group.add_argument(
        "--prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=defaults.prefix_cache,
        help="share KV between requests with a common prefix (radix-tree cache)",
    )
    group.add_argument(
        "--cache-aware-scheduling",
        action=argparse.BooleanOptionalAction,
        default=defaults.cache_aware_scheduling,
        help="admit the waiting request with the longest cached prefix first",
    )
    group.add_argument("--starvation-seconds", type=float, default=defaults.starvation_seconds)
    group.add_argument(
        "--eviction-policy",
        choices=["lru", "session"],
        default=defaults.eviction_policy,
        help="cached KV to give up first: least recently used, or expected to be needed last",
    )
    group.add_argument(
        "--host-kv-gb",
        type=float,
        default=defaults.host_kv_gb,
        help="host-memory KV tier in GiB: paused sessions' KV is copied there and restored",
    )
    group.add_argument("--min-restore-blocks", type=int, default=defaults.min_restore_blocks)
    group.add_argument(
        "--position-independent-reuse",
        action=argparse.BooleanOptionalAction,
        default=defaults.position_independent_reuse,
        help="reuse KV of tokens cached at another position (approximate)",
    )
    group.add_argument("--reuse-min-tokens", type=int, default=defaults.reuse_min_tokens)
    group.add_argument(
        "--reuse-recompute-ratio", type=float, default=defaults.reuse_recompute_ratio
    )
    group.add_argument(
        "--speculation",
        choices=["none", "ngram", "draft"],
        default=defaults.speculation or "none",
        help="speculative decoding: prompt-lookup n-grams or a draft model",
    )
    group.add_argument("--num-draft-tokens", type=int, default=defaults.num_draft_tokens)
    group.add_argument("--ngram-size", type=int, default=defaults.ngram_size)
    group.add_argument("--draft-min-acceptance", type=float, default=defaults.draft_min_acceptance)
    group.add_argument("--draft-cooldown-steps", type=int, default=defaults.draft_cooldown_steps)
    group.add_argument(
        "--verify-graphs",
        action=argparse.BooleanOptionalAction,
        default=defaults.verify_graphs,
        help="replay speculative verification from captured CUDA graphs",
    )
    group.add_argument("--draft-model", default=defaults.draft_model)
    group.add_argument("--seed", type=int, default=defaults.seed)


def engine_config_from_args(args: argparse.Namespace) -> EngineConfig:
    """Build an :class:`EngineConfig` from parsed engine flags."""
    return EngineConfig(
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        kv_memory_fraction=args.kv_memory_fraction,
        kv_gb=args.kv_gb,
        max_num_seqs=args.max_num_seqs,
        max_batched_tokens=args.max_batched_tokens,
        max_model_len=args.max_model_len,
        kv_cache_dtype=args.kv_cache_dtype,
        weight_dtype=args.weight_dtype,
        fused_kernels=args.fused_kernels,
        fused_kv_write=args.fused_kv_write,
        decode_kernel=args.decode_kernel,
        cuda_graphs=args.cuda_graphs,
        prefill_graphs=args.prefill_graphs,
        async_decode=args.async_decode,
        unified_batches=args.unified_batches,
        prefill_chunk_tokens=args.prefill_chunk_tokens,
        prefix_cache=args.prefix_cache,
        cache_aware_scheduling=args.cache_aware_scheduling,
        starvation_seconds=args.starvation_seconds,
        eviction_policy=args.eviction_policy,
        host_kv_gb=args.host_kv_gb,
        min_restore_blocks=args.min_restore_blocks,
        position_independent_reuse=args.position_independent_reuse,
        reuse_min_tokens=args.reuse_min_tokens,
        reuse_recompute_ratio=args.reuse_recompute_ratio,
        speculation=None if args.speculation == "none" else args.speculation,
        num_draft_tokens=args.num_draft_tokens,
        ngram_size=args.ngram_size,
        draft_min_acceptance=args.draft_min_acceptance,
        draft_cooldown_steps=args.draft_cooldown_steps,
        verify_graphs=args.verify_graphs,
        draft_model=args.draft_model,
        seed=args.seed,
    )


def _generate(args: argparse.Namespace) -> int:
    """Generate a completion for each prompt given on the command line."""
    from transformers import AutoTokenizer

    from minfer.engine import Engine

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engine = Engine.from_pretrained(args.model, engine_config_from_args(args))
    prompts = []
    for text in args.prompt:
        if not args.raw:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
        prompts.append(tokenizer(text, add_special_tokens=False)["input_ids"])
    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.sampling_seed,
    )
    start = time.perf_counter()
    outputs = engine.generate(prompts, params)
    elapsed = time.perf_counter() - start
    for text, output in zip(args.prompt, outputs, strict=True):
        print(f"### {text}\n{tokenizer.decode(output, skip_special_tokens=True)}\n")
    generated = sum(map(len, outputs))
    print(
        f"[{generated} tokens in {elapsed:.2f}s, {generated / elapsed:.0f} tok/s]", file=sys.stderr
    )
    return 0


def _serve(args: argparse.Namespace) -> int:
    """Serve the OpenAI-compatible API."""
    import uvicorn
    from transformers import AutoTokenizer

    from minfer.engine import Engine
    from minfer.server import create_app

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # The engine (and the model) live in a separate process; see minfer/server/async_engine.py.
    factory = functools.partial(Engine.from_pretrained, args.model, engine_config_from_args(args))
    app = create_app(factory, tokenizer, args.served_model_name or args.model)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        loop="uvloop",
        http="httptools",
        access_log=False,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The ``minfer`` argument parser."""
    parser = argparse.ArgumentParser(prog="minfer", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate completions for prompts")
    generate.add_argument("prompt", nargs="+", help="one or more prompts (batched together)")
    generate.add_argument("--raw", action="store_true", help="do not apply the chat template")
    generate.add_argument("--max-tokens", type=int, default=256)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument("--sampling-seed", type=int, default=None)
    add_engine_arguments(generate)
    generate.set_defaults(handler=_generate)

    serve = commands.add_parser("serve", help="serve the OpenAI-compatible HTTP API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--served-model-name", default=None, help="name clients use (default: --model)"
    )
    serve.add_argument("--log-level", default="info")
    add_engine_arguments(serve)
    serve.set_defaults(handler=_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its exit status."""
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
