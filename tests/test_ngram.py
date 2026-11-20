"""Prompt-lookup drafting."""

from __future__ import annotations

import random

from minfer.spec import NgramIndex


def brute_force(tokens: list[int], n: int, k: int) -> list[int]:
    """Continuation of the most recent earlier occurrence of the last n-gram."""
    if len(tokens) < n + 1:
        return []
    key = tokens[-n:]
    for end in range(len(tokens) - 1, n - 1, -1):
        if tokens[end - n : end] == key:
            return tokens[end : end + k]
    return []


def test_no_match_and_short_sequences():
    index = NgramIndex(3)
    assert index.propose([1, 2], 4) == []
    assert index.propose([1, 2, 3], 4) == []
    assert index.propose([1, 2, 3, 4, 5, 6], 4) == []
    assert index.propose([1, 2, 3, 4, 1, 2, 3], 0) == []


def test_proposes_the_earlier_continuation():
    index = NgramIndex(3)
    tokens = [9, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3]
    assert index.propose(tokens, 4) == [4, 5, 6, 7]
    assert index.propose(tokens, 2) == [4, 5]


def test_most_recent_occurrence_wins():
    index = NgramIndex(2)
    tokens = [1, 2, 10, 0, 1, 2, 20, 0, 1, 2]
    assert index.propose(tokens, 2) == [20, 0]


def test_continuation_may_run_into_the_current_context():
    # "a b a b a b": the earlier "a b" is followed by "a b" again, i.e. a repetition loop.
    index = NgramIndex(2)
    assert index.propose([5, 6, 5, 6], 6) == [5, 6]
    assert index.propose([5, 6, 5, 6, 5, 6], 6) == [5, 6]


def test_incremental_matches_brute_force():
    rng = random.Random(0)
    for n in (1, 2, 3):
        index = NgramIndex(n)
        tokens: list[int] = []
        for _ in range(600):
            tokens.append(rng.randrange(6))  # small alphabet: plenty of repeats
            assert index.propose(tokens, 5) == brute_force(tokens, n, 5)


def test_truncate_forgets():
    index = NgramIndex(2)
    assert index.propose([1, 2, 3, 1, 2], 2) == [3, 1]
    index.truncate()
    assert index.propose([7, 8, 9], 2) == []
    assert index.propose([7, 8, 9, 7, 8], 2) == [9, 7]
