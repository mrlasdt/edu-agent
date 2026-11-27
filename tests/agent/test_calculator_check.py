"""
The calculator check: a deterministic comparison of the final answer with the math tool results.

Behaviors under test:
  1. an answer that is a tool value agrees, and the matched pair is reported
  2. a percent answer matches the factor and the fraction ("-8%" = 0.92)
  3. an answer that is none of the tool values differs
  4. nothing to compare (no calls, failed calls, no answer, a mix) is `none`, never a guess
  5. quantity comparisons compare the letter with `math_compare`'s verdict
"""
from dataclasses import dataclass
from typing import Any

import pytest

from services.agent.src.agent.skills.calculator_check import (
    AGREES,
    DIFFERS,
    NONE,
    calculator_check,
)


@dataclass
class Call:
    name: str
    data: dict[str, Any] | None
    ok: bool = True


def evaluate(value: str, **extra: str) -> Call:
    return Call("math_evaluate", {"value": value, **extra})


def solve(*roots: str) -> Call:
    return Call("math_solve", {"variable": "x", "roots": list(roots)})


@pytest.mark.parametrize(
    ("answer", "call", "pair"),
    [
        ("0.92", evaluate("0.92"), ("0.92", "0.92")),
        ("-8%", evaluate("0.92", exact="23/25"), ("-8%", "0.92")),
        ("8% decrease", evaluate("0.92"), ("-8%", "0.92")),
        ("-0.08", evaluate("-0.08"), ("-0.08", "-0.08")),
        ("3/4", evaluate("0.75", exact="3/4"), ("3/4", "0.75")),
        ("(3)/(4)", evaluate("0.75"), ("(3)/(4)", "0.75")),
        ("1,200", evaluate("1200"), ("1,200", "1200")),
        ("2.33", evaluate("2.333333333", exact="7/3"), ("2.33", "2.333333333")),
        ("18", evaluate("18"), ("18", "18")),
    ],
)
def test_an_answer_that_is_a_tool_value_agrees(answer, call, pair) -> None:
    check = calculator_check(answer, [call])
    assert check.verdict == AGREES
    assert (check.answer, check.tool_value) == pair
    assert check.tool == "math_evaluate"


def test_every_root_in_the_answer_must_be_a_tool_root() -> None:
    assert calculator_check("-1, 4", [solve("-1", "4")]).verdict == AGREES
    assert calculator_check("4", [solve("-1", "4")]).verdict == AGREES  # a part, all of it right
    assert calculator_check("-1, 5", [solve("-1", "4")]).verdict == NONE  # a mix is not judged


def test_an_answer_that_is_none_of_the_tool_values_differs() -> None:
    check = calculator_check("-12%", [evaluate("0.92")])
    assert check.verdict == DIFFERS
    assert (check.answer, check.tool_value) == ("-12%", "0.92")
    assert calculator_check("5", [solve("-1", "4")]).verdict == DIFFERS


def test_a_rounded_answer_agrees_but_a_coarser_miss_does_not() -> None:
    assert calculator_check("2.3", [evaluate("2.333333")]).verdict == AGREES
    assert calculator_check("2.4", [evaluate("2.333333")]).verdict == DIFFERS


def test_a_positive_percent_is_not_the_factor_of_a_fall() -> None:
    assert calculator_check("8%", [evaluate("0.92")]).verdict == DIFFERS


@pytest.mark.parametrize(
    ("answer", "calls"),
    [
        ("0.92", []),
        ("0.92", [Call("math_evaluate", {"error": "bad"}, ok=False)]),
        (None, [evaluate("0.92")]),
        ("", [evaluate("0.92")]),
        ("Quantity A", [evaluate("0.92")]),  # no compare call to judge a letter by
        ("no number here", [evaluate("0.92")]),
        ("0.92", [Call("notes_search", {"value": "0.92"})]),  # only math tools count
        ("0.92", [Call("math_evaluate", None)]),
        ("1", [evaluate("x + 1")]),  # symbolic results have no number to compare
    ],
)
def test_nothing_to_compare_is_none(answer, calls) -> None:
    assert calculator_check(answer, calls).verdict == NONE


def test_a_failed_call_is_ignored_when_another_succeeded() -> None:
    calls = [Call("math_evaluate", {"error": "bad"}, ok=False), evaluate("0.92")]
    assert calculator_check("0.92", calls).verdict == AGREES


@pytest.mark.parametrize(
    ("answer", "verdict", "expected"),
    [
        ("A", "A", AGREES),
        ("Quantity A is greater", "A", AGREES),
        ("Quantity B is greater", "A", DIFFERS),
        ("The two quantities are equal", "C", AGREES),
        ("The relationship cannot be determined", "D", AGREES),
        ("B", "undetermined", NONE),
    ],
)
def test_a_quantity_answer_is_compared_with_the_compare_verdict(answer, verdict, expected) -> None:
    call = Call("math_compare", {"verdict": verdict, "why": "..."})
    assert calculator_check(answer, [call]).verdict == expected


def test_the_pair_is_public_as_plain_strings() -> None:
    check = calculator_check("-8%", [evaluate("0.92")])
    assert check.public() == {
        "verdict": "agrees", "answer": "-8%", "tool_value": "0.92", "tool": "math_evaluate",
    }
