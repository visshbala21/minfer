"""bench/medians.py: combining repeated benchmark runs (CPU only)."""

from __future__ import annotations

from bench.medians import combine


def test_median_min_max_at_every_number():
    runs = [
        {"model": "m", "runs": {"64": {"tok_s": 3.0, "ttft": {"p99": 10}}}, "errors": ["a"]},
        {"model": "m", "runs": {"64": {"tok_s": 1.0, "ttft": {"p99": 30}}}, "errors": []},
        {"model": "m", "runs": {"64": {"tok_s": 2.0, "ttft": {"p99": 20}}, "96": {"tok_s": 5.0}}},
    ]
    median, low, high, counts = combine(runs)
    assert median["model"] == "m" and "errors" not in median
    assert median["runs"]["64"] == {"tok_s": 2.0, "ttft": {"p99": 20.0}}
    assert low["runs"]["64"]["tok_s"] == 1.0 and high["runs"]["64"]["ttft"]["p99"] == 30.0
    # A number only one run has is reported, with how many runs it came from.
    assert median["runs"]["96"]["tok_s"] == 5.0 and counts == {"/runs/96/tok_s": 1}


def test_lists_are_kept_only_when_identical():
    runs = [{"flags": ["--a"], "errors": ["x"]}, {"flags": ["--a"], "errors": []}]
    median, _, _, _ = combine(runs)
    assert median == {"flags": ["--a"]}
