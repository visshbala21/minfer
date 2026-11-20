"""Answer extraction for the GSM8K evaluation (no model, no dataset)."""

from __future__ import annotations

import pytest

from bench.gsm8k import extract_answer, gold_answer, is_correct, parse_number


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("... so she has 18 dollars left.\n#### 18", 18.0),
        ("Step 1: 3 * 4 = 12\nStep 2: 12 + 5 = 17\n#### 17.", 17.0),
        ("#### $1,234.50", 1234.5),
        ("#### -7", -7.0),
        ("The answer is 42 apples, since 6 * 7 = 42.", 42.0),  # no marker: last number
        ("First #### 3 was wrong.\nCorrected:\n#### 5", 5.0),  # last marker wins
        ("#### \nThe total is 9", 9.0),  # empty marker: fall back to the last number
        ("I cannot solve this.", None),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_parse_number_and_gold():
    assert parse_number("about 2,000 km") == 2000.0
    assert parse_number("none") is None
    assert gold_answer("He pays 5*3=<<5*3=15>>15.\n#### 15") == 15.0
    assert gold_answer("...\n#### 1,080") == 1080.0
    with pytest.raises(ValueError):
        gold_answer("no marker #### here")


def test_is_correct():
    assert is_correct(18.0, 18.0) and is_correct(0.1 + 0.2, 0.3)
    assert not is_correct(18.5, 18.0) and not is_correct(None, 18.0)
