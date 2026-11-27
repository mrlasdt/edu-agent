"""
The calculator check: does the reply's final answer match what the model's math tools returned?

Deterministic and free: no model call, only a comparison of numbers. It is a different check from
the independent check (`independent_check.py`), which asks a model to formalize the problem and
solves that with sympy. This one reads the tool calls the model itself made in the turn and asks
whether the answer it then wrote down is one of their results.

  agrees    every value in the final answer is a value a successful tool call returned
            (a percent `p%` also matches the factor `1 + p/100` and the fraction `p/100`)
  differs   tool results exist, the answer has values, and none of them is a tool result
  none      nothing to compare: no successful tool call, no extractable answer, or only some of
            the answer's values match (an answer that mixes checked and unchecked values is not
            called either way)

A `differs` verdict is information for the learner and the log; it does not trigger a retry.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Protocol

AGREES = "agrees"
DIFFERS = "differs"
NONE = "none"

_NUMBER = re.compile(
    r"(?P<frac>\(?(?P<n>[-+]?\d+(?:\.\d+)?)\)?\s*/\s*\(?(?P<d>\d+(?:\.\d+)?)\)?)"
    r"|(?P<num>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+)"
)
_PERCENT_AFTER = re.compile(r"\s*(?:%|percent\b)", re.IGNORECASE)
_TOOLS_WITH_VALUES = ("math_evaluate", "math_solve", "math_compare")
_VERDICT_WORDS = (
    (re.compile(r"\bquantity\s+a\s+is\s+greater\b", re.IGNORECASE), "A"),
    (re.compile(r"\bquantity\s+b\s+is\s+greater\b", re.IGNORECASE), "B"),
    (re.compile(r"\b(?:two\s+quantities|quantities)\s+are\s+equal\b", re.IGNORECASE), "C"),
    (re.compile(r"\bcannot\s+be\s+determined\b", re.IGNORECASE), "D"),
)
_DOWN = re.compile(
    r"\b(?:decreas\w*|drops?|falls?|loss|lower\w*|reduc\w*|declin\w*)\b", re.IGNORECASE
)
_VERDICT_LETTER = re.compile(r"^\W*(?:quantity\s+)?([A-D])\W*$", re.IGNORECASE)


class _Call(Protocol):
    name: str
    ok: bool
    data: dict[str, Any] | None


@dataclass(frozen=True)
class CalcCheck:
    """The verdict and, for `agrees`, the pair that matched (printed as "answer -8% = 0.92")."""

    verdict: str
    answer: str = ""  # the answer's value as written, "-8%"
    tool_value: str = ""  # the tool's value it matched, "0.92"
    tool: str = ""  # which tool returned it

    def public(self) -> dict[str, str]:
        return {
            "verdict": self.verdict,
            "answer": self.answer,
            "tool_value": self.tool_value,
            "tool": self.tool,
        }


@dataclass(frozen=True)
class _Value:
    shown: str
    number: float
    percent: bool
    decimals: int  # digits after the point as written; sets how coarse a rounded answer may be


def _to_number(text: str) -> float | None:
    """A tool value as a float: a decimal or a plain fraction; symbolic text is skipped."""
    cleaned = text.strip().replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        pass
    match = re.fullmatch(r"\(?\s*(-?\d+)\s*\)?\s*/\s*\(?\s*(\d+)\s*\)?", cleaned)
    if match and int(match.group(2)) != 0:
        return float(Fraction(int(match.group(1)), int(match.group(2))))
    return None


def _answer_values(answer: str) -> list[_Value]:
    values: list[_Value] = []
    for match in _NUMBER.finditer(answer):
        shown = match.group(0).strip()
        if match.group("frac"):
            denominator = float(match.group("d"))
            if denominator == 0:
                continue
            number = float(match.group("n")) / denominator
            decimals = 12
        else:
            text = match.group("num").replace(",", "")
            number = float(text)
            decimals = len(text.split(".")[1]) if "." in text else 0
        percent = bool(_PERCENT_AFTER.match(answer, match.end()))
        if percent:
            shown += "%"
            if number > 0 and _DOWN.search(answer):  # "8% decrease" is -8%
                number, shown = -number, "-" + shown
        values.append(_Value(shown, number, percent, decimals))
    return values


def _verdict_letter(answer: str) -> str | None:
    for pattern, letter in _VERDICT_WORDS:
        if pattern.search(answer):
            return letter
    match = _VERDICT_LETTER.match(answer)
    return match.group(1).upper() if match else None


def _tool_values(call: _Call) -> list[tuple[str, float]]:
    """The numbers a successful call returned, as (text, number)."""
    data = call.data or {}
    texts: list[str] = []
    for key in ("value", "exact", "a", "b"):
        if isinstance(data.get(key), str):
            texts.append(data[key])
    for key in ("roots", "decimals"):
        if isinstance(data.get(key), list):
            texts += [t for t in data[key] if isinstance(t, str)]
    out: list[tuple[str, float]] = []
    for text in texts:
        number = _to_number(text)
        if number is not None and math.isfinite(number):
            out.append((text, number))
    return out


def _close(answer: float, tool: float, decimals: int) -> bool:
    """Equal, or equal once the tool value is rounded to the digits the answer shows."""
    if math.isclose(answer, tool, rel_tol=1e-9, abs_tol=1e-9):
        return True
    return decimals < 12 and abs(answer - tool) <= 0.5 * 10 ** (-decimals) + 1e-12


def _matches(value: _Value, tool: float) -> bool:
    if _close(value.number, tool, value.decimals):
        return True
    if value.percent:
        # "-8%" is the factor 0.92 and the fraction -0.08
        step = value.decimals + 2
        return _close(1 + value.number / 100, tool, step) or _close(value.number / 100, tool, step)
    return False


def calculator_check(final_answer: str | None, tool_calls: Iterable[_Call]) -> CalcCheck:
    """Compare `final_answer` (as `answers.normalize_answer` returns it) with the successful
    math tool calls of the turn."""
    ok_calls = [c for c in tool_calls if c.ok and c.name in _TOOLS_WITH_VALUES and c.data]
    if not final_answer or not ok_calls:
        return CalcCheck(NONE)

    letter = _verdict_letter(final_answer)
    if letter is not None:
        compared = [c for c in ok_calls if c.name == "math_compare"]
        verdicts = [str(c.data.get("verdict")) for c in compared if c.data]
        if not verdicts:
            return CalcCheck(NONE)
        if letter in verdicts:
            return CalcCheck(AGREES, letter, letter, "math_compare")
        if any(v in ("A", "B", "C", "D") for v in verdicts):
            return CalcCheck(DIFFERS, letter, verdicts[-1], "math_compare")
        return CalcCheck(NONE)  # "undetermined": the tool did not decide

    values = _answer_values(final_answer)
    available = [
        (call.name, text, number) for call in ok_calls for text, number in _tool_values(call)
    ]
    if not values or not available:
        return CalcCheck(NONE)

    pairs: list[tuple[_Value, str, str]] = []
    for value in values:
        hit = next((a for a in available if _matches(value, a[2])), None)
        if hit is not None:
            pairs.append((value, hit[1], hit[0]))
    if len(pairs) == len(values):
        first = pairs[0]
        if len(pairs) == 1:
            return CalcCheck(AGREES, first[0].shown, first[1], first[2])
        return CalcCheck(
            AGREES,
            ", ".join(p[0].shown for p in pairs),
            ", ".join(p[1] for p in pairs),
            first[2],
        )
    if not pairs:
        last = available[-1]
        return CalcCheck(DIFFERS, ", ".join(v.shown for v in values), last[1], last[0])
    return CalcCheck(NONE)
