"""
Deterministic verifier tools for generated Quant items.

Every tool is a pure check returning a CheckResult; the pipeline runs them as a tool-use
step and feeds failures back to the generator. The solver check recomputes the key through
the same verify-math skill the Quant agent uses (HTTP to the math-verifier in production,
the in-process sandbox in evals), so the generator never grades its own answer.
"""
from __future__ import annotations

import itertools
import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from services.agent.src.agent.pipeline.schema import CheckResult, QuestionItem, QuestionType
from services.agent.src.agent.skills.verify_math import VerifyMathSkill

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_QC_LETTERS = {"A", "B", "C", "D"}
_MIN_STEM_CHARS = 15
_LEAK_PHRASES = re.compile(
    r"(correct answer|the answer is|answer\s*[:=]|\(correct\)|\bcorrect\b\s*[:*])", re.IGNORECASE
)
_QC_GRID = range(-3, 6)  # integer sample points used to decide whether a QC is determinable


def split_values(text: str) -> list[str]:
    """'2, -2' / 'x=2 or x=3' / '[2, 3]' -> ['2', '-2'] (variable prefixes stripped)."""
    body = text.strip().strip("[]{}")
    parts = re.split(r"\bor\b|[,;]", body, flags=re.IGNORECASE)
    return [re.sub(r"^\s*\w+\s*=\s*", "", p).strip() for p in parts if p.strip()]


def _parse(text: str) -> sp.Expr | None:
    try:
        return sp.simplify(parse_expr(text.strip(), transformations=_TRANSFORMS))
    except Exception:  # noqa: BLE001 - any parse failure just means "not a usable value"
        return None


def _parse_values(text: str) -> list[sp.Expr] | None:
    values = [_parse(v) for v in split_values(text)]
    if not values or any(v is None for v in values):
        return None
    return [v for v in values if v is not None]


def _same(a: sp.Expr, b: sp.Expr) -> bool:
    try:
        return bool(sp.simplify(a - b) == 0)
    except Exception:  # noqa: BLE001 - e.g. a boolean "value" from a degenerate equation
        return False


def _contains(values: list[sp.Expr], target: sp.Expr) -> bool:
    return any(_same(v, target) for v in values)


# ── structure ────────────────────────────────────────────────────────────────


def structure_check(item: QuestionItem) -> CheckResult:
    problems: list[str] = []
    qtype = item.question_type
    if len(item.stem.strip()) < _MIN_STEM_CHARS:
        problems.append("stem is too short to be a complete question")
    if qtype in (QuestionType.multiple_choice_single, QuestionType.multiple_choice_multiple):
        n = len(item.choices)
        if not 4 <= n <= 6:
            problems.append(f"multiple choice needs 4-6 choices, got {n}")
        if len({c.strip().lower() for c in item.choices}) != n:
            problems.append("two choices are textually identical")
    elif item.choices:
        problems.append(f"{qtype.value} must not have answer choices")
    if qtype == QuestionType.multiple_choice_multiple and "all" not in item.stem.lower():
        problems.append("a select-all question must tell the candidate to select all that apply")
    if qtype == QuestionType.quantitative_comparison:
        if not (item.quantity_a.strip() and item.quantity_b.strip()):
            problems.append("quantitative comparison needs quantity_a and quantity_b")
        if item.answer.strip().upper() not in _QC_LETTERS:
            problems.append("quantitative comparison answer must be one of A, B, C, D")
    elif not item.expression.strip():
        problems.append("expression is required so the key can be recomputed")
    return CheckResult(name="structure", passed=not problems, detail="; ".join(problems))


# ── solver ───────────────────────────────────────────────────────────────────


async def solve_check(item: QuestionItem, skill: VerifyMathSkill) -> CheckResult:
    """Recompute the key independently and compare it with the item's claimed answer."""
    if item.question_type == QuestionType.quantitative_comparison:
        return await _solve_qc(item, skill)
    result = await skill.verify(expression=item.expression, expected=item.answer)
    if result.error:
        return CheckResult(name="solver", passed=False, detail=f"cannot solve: {result.error}")
    if not result.verified:
        return CheckResult(
            name="solver",
            passed=False,
            detail=f"keyed answer {item.answer!r} does not match computed {result.computed!r}",
        )
    return CheckResult(name="solver", passed=True, detail=f"computed {result.computed}")


def _qc_signs(diff: sp.Expr) -> set[int] | None:
    """Signs the difference A - B takes (over an integer grid when it has free symbols)."""
    symbols = sorted(diff.free_symbols, key=str)
    if not symbols:
        return {int(sp.sign(diff))} if diff.is_real else None
    if len(symbols) > 2:
        return None
    signs: set[int] = set()
    for point in itertools.product(_QC_GRID, repeat=len(symbols)):
        value = diff.subs(dict(zip(symbols, point, strict=True)))
        if value.is_real:
            signs.add(int(sp.sign(value)))
    return signs or None


async def _solve_qc(item: QuestionItem, skill: VerifyMathSkill) -> CheckResult:
    diff_expr = f"({item.quantity_a}) - ({item.quantity_b})"
    # The skill returns what it computed whether or not `expected` matches.
    computed = await skill.verify(expression=diff_expr, expected="0")
    diff = _parse(computed.computed) if computed.computed else _parse(diff_expr)
    if diff is None:
        return CheckResult(name="solver", passed=False, detail="cannot parse the quantities")
    signs = _qc_signs(diff)
    if signs is None:
        return CheckResult(name="solver", passed=False, detail="cannot decide the comparison")
    if len(signs) > 1:
        correct = "D"
    else:
        correct = {1: "A", -1: "B", 0: "C"}[next(iter(signs))]
    claimed = item.answer.strip().upper()
    if claimed != correct:
        return CheckResult(
            name="solver",
            passed=False,
            detail=f"keyed {claimed} but the comparison gives {correct}",
        )
    return CheckResult(name="solver", passed=True, detail=f"comparison gives {correct}")


async def distractor_check(item: QuestionItem, skill: VerifyMathSkill) -> CheckResult:
    """Multiple choice only: the keyed options are exactly the solution values."""
    if item.question_type not in (
        QuestionType.multiple_choice_single,
        QuestionType.multiple_choice_multiple,
    ):
        return CheckResult(name="distractors", passed=True, detail="not applicable")
    choices = [_parse(c) for c in item.choices]
    if any(c is None for c in choices):
        return CheckResult(
            name="distractors", passed=False, detail="every choice must be a parseable value"
        )
    values = [c for c in choices if c is not None]
    answers = _parse_values(item.answer)
    if answers is None:
        return CheckResult(name="distractors", passed=False, detail="answer is not parseable")
    problems: list[str] = []
    for i, a in enumerate(values):
        if any(_same(a, b) for b in values[:i]):
            problems.append(f"choices {i} and an earlier choice are numerically equal")
    for ans in answers:
        hits = [i for i, v in enumerate(values) if _same(v, ans)]
        if len(hits) != 1:
            problems.append(f"keyed value {ans} appears in {len(hits)} choices, expected 1")
    if item.question_type == QuestionType.multiple_choice_single and len(answers) != 1:
        problems.append("a single-answer question must key exactly one value")
    if item.question_type == QuestionType.multiple_choice_multiple and len(answers) < 2:
        problems.append("a select-all question must key at least two values")
    solved = await skill.verify(expression=item.expression, expected=item.answer)
    solution_set = _parse_values(solved.computed.strip("[]")) if solved.computed else None
    if solution_set:
        for i, v in enumerate(values):
            if _contains(solution_set, v) and not _contains(answers, v):
                problems.append(f"distractor {item.choices[i]!r} is also a correct value")
    return CheckResult(name="distractors", passed=not problems, detail="; ".join(problems))


# ── leak / ambiguity (rule based) ────────────────────────────────────────────


def leak_check(item: QuestionItem) -> CheckResult:
    """The stem and choices must not give the key away or mark the right option."""
    problems: list[str] = []
    if _LEAK_PHRASES.search(item.stem):
        problems.append("stem states or labels the answer")
    if any(re.search(r"[*✓✔]|\(correct\)", c, re.IGNORECASE) for c in item.choices):
        problems.append("a choice is marked as correct")
    return CheckResult(name="leak", passed=not problems, detail="; ".join(problems))


async def run_checks(item: QuestionItem, skill: VerifyMathSkill) -> list[CheckResult]:
    """All deterministic tools, in order. The solver only runs on a structurally sound item."""
    results = [structure_check(item), leak_check(item)]
    if results[0].passed:
        results.append(await solve_check(item, skill))
        results.append(await distractor_check(item, skill))
    return results
