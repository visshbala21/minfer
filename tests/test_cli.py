"""CLI flag parsing (no model is loaded)."""

from __future__ import annotations

from minfer.cli import build_parser, engine_config_from_args
from minfer.engine import EngineConfig


def parse(*argv: str) -> EngineConfig:
    return engine_config_from_args(build_parser().parse_args(["generate", "hi", *argv]))


def test_defaults_match_engine_config():
    assert parse() == EngineConfig()


def test_feature_toggles():
    assert parse("--no-cuda-graphs").cuda_graphs is False
    assert parse("--cuda-graphs").cuda_graphs is True
    assert parse("--prefill-chunk-tokens", "512").prefill_chunk_tokens == 512
    assert (
        parse("--prefill-chunk-tokens", "512", "--no-chunked-prefill").prefill_chunk_tokens is None
    )


def test_prefix_cache_toggles():
    assert parse("--no-prefix-cache").prefix_cache is False
    assert parse("--no-cache-aware-scheduling").cache_aware_scheduling is False
    assert parse("--starvation-seconds", "0.25").starvation_seconds == 0.25


def test_limits():
    config = parse("--max-num-seqs", "8", "--num-blocks", "100", "--block-size", "32")
    assert (config.max_num_seqs, config.num_blocks, config.block_size) == (8, 100, 32)
