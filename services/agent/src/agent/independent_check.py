"""
The independent answer check for a Quant solve turn.

The well-formedness check, `QuantAgent.verify_well_formed`, compares the model's answer with
itself, so a wrong answer that parses passes. This check does not: a second, short model call
writes the problem as an expression or equation WITHOUT seeing the first answer
(`parse_expression` reads its `<expr>` tag), and the math verifier's sympy sandbox computes that
expression. The computed value is then compared with the answer the solver claimed (`compare`).

  agrees         the computed value matches the claimed answer
  disagrees      it does not; the computed value (`solver`) is shown next to it
  not_checkable  no number to compare: the problem has no numeric answer (a quantitative
                 comparison, a statement to pick), the model wrote no usable expression, the
                 solver could not compute one, or the verifier was unreachable

The limit is ADR-0012's: only questions whose answer is a number the solver can recompute are
checked. "agrees" means two routes (the working, and the model's equation run by a solver) reached
the same number. It does not mean the problem was read correctly: a model that misreads the
problem the same way twice agrees with itself. Every function here is pure or goes through the
verify-math skill; the model call lives in `QuantAgent._formalize`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, get_args

from services.agent.src.agent.capabilities.verify_math import VerifyMath
from shared.src.shared import events
from shared.src.shared.events import Verdict

AGREES: Final = "agrees"
DISAGREES: Final = "disagrees"
NOT_CHECKABLE: Final = "not_checkable"
VERDICTS: Final = get_args(Verdict)

# The model answers `<expr>NONE</expr>` when the problem has no numeric answer to compute.
_NONE_WORDS = {"none", "n/a", "not checkable", "not_checkable"}
_EXPR_TAG = re.compile(r"<expr>(.*?)</expr>", re.IGNORECASE | re.DOTALL)
# Names the sandbox allow-list knows (services/math_verifier safe_parser); a value that holds
# any other letter still has an unknown in it, so there is no number to compare.
_NAMES = re.compile(
    r"\b(?:sqrt|cbrt|pi|abs|exp|log|ln|sin|cos|tan|asin|acos|atan|factorial|binomial|gcd|lcm|"
    r"floor|ceiling|min|max)\b",
    re.IGNORECASE,
)
_NUMBER_CHARS = re.compile(r"^[\d\s.,+\-*/^()!]*$")
_DECIMAL = re.compile(r"^-?\d+\.(\d+)$")
_VARIABLE_PREFIX = re.compile(r"^\s*\w+\s*=\s*")
_THOUSANDS = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?!\d)")


@dataclass(frozen=True)
class IndependentCheck:
    verdict: Verdict
    solver: str = ""  # what the solver computed; set only for "disagrees"
    reason: str = ""  # why a check was "not_checkable" (for the log, not the learner)

    def event(self) -> events.IndependentCheck:
        """What the turn reports: the verdict, and the solver's value when they disagree."""
        return events.IndependentCheck(verdict=self.verdict, solver=self.solver)


def parse_expression(text: str) -> str | None:
    """The expression inside the last `<expr>...</expr>` tag; None for no tag or NONE."""
    tags = _EXPR_TAG.findall(text)
    if not tags:
        return None
    expression = " ".join(str(tags[-1]).split())
    return None if not expression or expression.lower() in _NONE_WORDS else expression


def _values(text: str) -> list[str]:
    """The bare values in "2, -2" / "[1/3, sqrt(2)]" / "x = 4"."""
    body = text.strip().strip("[]{}")
    values = [_VARIABLE_PREFIX.sub("", v).strip() for v in re.split(r"[,;]|\bor\b|\band\b", body)]
    return [v for v in values if v]


def is_number_list(text: str) -> bool:
    """True when every value in "2, -2" / "[1/3, sqrt(2)]" / "x = 4" is a number, not a symbol."""
    values = _values(text)
    if not values:
        return False
    for value in values:
        bare = _NAMES.sub("", value)
        if not _NUMBER_CHARS.match(bare) or not (re.search(r"\d", bare) or "pi" in value.lower()):
            return False
    return True


def _display(computed: str) -> str:
    """The solver's value for the learner: "[2, 3]" -> "2, 3"."""
    return computed.strip().strip("[]").strip()


async def compare(
    skill: VerifyMath, expression: str | None, claimed: str
) -> IndependentCheck:
    """
    Compute `expression` with the solver and compare it with the `claimed` answer. Never raises.
    A claimed answer that is not a number (a comparison letter, a word) is not checkable, and
    so is anything the solver could not turn into a number: a wrong verdict here would send a
    right answer through the retry ladder.
    """
    if expression is None:
        return IndependentCheck(NOT_CHECKABLE, reason="no expression for a numeric answer")
    claimed = _THOUSANDS.sub(lambda m: m.group().replace(",", ""), claimed)
    if not is_number_list(claimed):
        return IndependentCheck(NOT_CHECKABLE, reason="the answer is not a number")
    result = await skill.verify(expression=expression, expected=claimed)
    if result.error:
        return IndependentCheck(NOT_CHECKABLE, reason=f"solver: {result.error}")
    if result.verified:
        return IndependentCheck(AGREES)
    if not is_number_list(result.computed):
        return IndependentCheck(NOT_CHECKABLE, reason="the solver did not compute a number")
    if await _is_rounding_of(skill, expression, claimed):
        return IndependentCheck(AGREES)
    if "=" in expression and await _is_some_of_the_roots(skill, result.computed, claimed):
        return IndependentCheck(
            NOT_CHECKABLE, reason="the answer is some of the roots; the question may restrict them"
        )
    return IndependentCheck(DISAGREES, solver=_display(result.computed))


async def _is_rounding_of(skill: VerifyMath, expression: str, claimed: str) -> bool:
    """
    A decimal answer that is the right value rounded to its own number of places ("3.3" for
    10/3) agrees. Asked of the same solver as whole numbers, which compare exactly: is
    floor(value * 10^d + 1/2) equal to the claimed answer with its point removed (33)?
    Only for a single decimal against a plain expression (an equation has no value to round).
    """
    text = claimed.strip()
    match = _DECIMAL.match(text)
    if match is None or "=" in expression:
        return False
    scale = 10 ** len(match.group(1))
    rounded = await skill.verify(
        expression=f"floor(({expression}) * {scale} + 1/2)",
        expected=str(int(text.replace(".", ""))),
    )
    return rounded.verified and not rounded.error


async def _is_some_of_the_roots(skill: VerifyMath, computed: str, claimed: str) -> bool:
    """
    True when every claimed value is one of the solver's roots and some root is left over
    ("2" against "-2, 2"): the question may ask for the positive root, so this is no disagreement.
    Each value is matched with the same solver, so "sqrt(4)" matches the root "2".
    """
    roots = _values(computed)
    wanted = _values(claimed)
    matched: set[int] = set()
    for value in wanted:
        for index, root in enumerate(roots):
            check = await skill.verify(expression=value, expected=root)
            if check.verified and not check.error:
                matched.add(index)
                break
        else:
            return False
    return bool(wanted) and len(matched) < len(roots)
