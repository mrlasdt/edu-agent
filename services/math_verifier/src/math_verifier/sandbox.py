from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


@dataclass
class VerificationResult:
    verified: bool
    computed: str
    error: str | None = None


# ── internal helpers ──────────────────────────────────────────────────────────


def _parse_safe(s: str) -> sp.Expr | None:
    try:
        return parse_expr(s.strip(), transformations=TRANSFORMS)
    except Exception:
        return None


def _value_list(s: str) -> list[str]:
    """Split a value list ("2, 3", "x=2 or x=3", "[2, 3]") into bare value strings."""
    s = s.strip().lstrip("[{").rstrip("]}]")
    parts = re.split(r"\bor\b|\band\b|[,;]", s, flags=re.IGNORECASE)
    return [re.sub(r"^\s*\w+\s*=\s*", "", p.strip()) for p in parts if p.strip()]


def _normalize_expected(expected: str) -> set[sp.Expr]:
    """
    Parse an expected-answer string into a set of sympy values.
    Handles: "2", "2, 3", "x=2 or x=3", "[2, 3]", "{2, 3}"
    """
    result: set[sp.Expr] = set()
    for cleaned in _value_list(expected):
        expr = _parse_safe(cleaned)
        if expr is not None:
            result.add(sp.simplify(expr))
    return result


_PROSE = re.compile(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}")


def _parse_value(value: str) -> sp.Expr:
    """Strict parse of one value; words are rejected rather than read as implicit products."""
    if _PROSE.search(value):
        raise ValueError(f"not a math value: {value!r}")
    return parse_expr(value, transformations=TRANSFORMS)


def _is_value_list(expression: str) -> bool:
    """A list of two or more values ("2, -2"), not an equation to solve."""
    values = _value_list(expression)
    return len(values) > 1 and not any("=" in v for v in values)


def _solve_equation(expression: str) -> tuple[str, set[sp.Expr]]:
    """
    Solve an equation 'lhs = rhs'. Returns (computed_str, solution_set).
    Raises ValueError on parse failure.
    """
    parts = expression.split("=", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse as equation: {expression!r}")

    lhs = parse_expr(parts[0].strip(), transformations=TRANSFORMS)
    rhs = parse_expr(parts[1].strip(), transformations=TRANSFORMS)
    diff = lhs - rhs

    free_syms = sorted(diff.free_symbols, key=str)
    if not free_syms:
        # Pure numeric equality
        simplified = sp.simplify(diff)
        val = sp.sympify(simplified == 0)
        return str(val), {val}

    solutions = sp.solve(diff, free_syms[0] if len(free_syms) == 1 else free_syms)
    solution_list = solutions if isinstance(solutions, list) else [solutions]
    solution_set = {sp.simplify(s) for s in solution_list}
    return str(sorted(solution_set, key=str)), solution_set


def _sync_verify(expression: str, expected: str) -> VerificationResult:
    """Run sympy verification synchronously. Executed inside a thread executor."""
    if not expression or not expression.strip():
        return VerificationResult(
            verified=False, computed="", error="parse error: empty expression"
        )
    try:
        if _is_value_list(expression):
            computed_set = {sp.simplify(_parse_value(v)) for v in _value_list(expression)}
            computed_str = str(sorted(computed_set, key=str))
        elif "=" in expression:
            computed_str, computed_set = _solve_equation(expression)
        else:
            expr = parse_expr(expression.strip(), transformations=TRANSFORMS)
            val = sp.simplify(expr)
            computed_str = str(val)
            computed_set = {val}

        expected_set = _normalize_expected(expected)
        verified = bool(computed_set and expected_set and computed_set == expected_set)
        return VerificationResult(verified=verified, computed=computed_str)
    except Exception as exc:
        return VerificationResult(verified=False, computed="", error=f"parse error: {exc}")


# ── public API ────────────────────────────────────────────────────────────────


async def verify_expression(
    expression: str, expected: str, timeout: float = 2.0
) -> VerificationResult:
    """
    Async entry point: run sympy in a thread executor with a hard timeout.
    Never raises — returns a VerificationResult with error set on failure.
    """
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _sync_verify, expression, expected),
            timeout=timeout,
        )
        return result
    except TimeoutError:
        return VerificationResult(verified=False, computed="", error="timeout")
