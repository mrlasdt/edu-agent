"""
The computations behind the three math tools a model can call: `math_evaluate`, `math_solve`,
`math_compare`.

The model-facing descriptions live with the agent (`agent.skills.math_tools`); this module is
the sympy side. Every function here:

- takes untrusted text a language model wrote, and parses it only through `parse_math` (the
  allow-list in `safe_parser`);
- runs inside the throwaway child process `sandbox.run_math_tool` starts, so a runaway
  computation is killed, not waited for;
- never raises: it returns a small JSON-ready dict, either the result or `{"error": "..."}`.

Results are deliberately short. The model pays for every character it reads, so a result
carries the answer and only what is needed to use it (`value`, `exact` when different, a
decimal when the exact form is not already one). Errors say what was wrong and what to write
instead, because the model's next move is to retry with a corrected call.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from typing import Any

import sympy as sp

from services.math_verifier.src.math_verifier.safe_parser import (
    CONSTANTS,
    FUNCTIONS,
    MAX_EXPRESSION_LENGTH,
    MathSyntaxError,
    parse_math,
)

# Tool name -> the argument names it takes (the first are required).
REQUIRED: dict[str, tuple[str, ...]] = {
    "math_evaluate": ("expression",),
    "math_solve": ("equation",),
    "math_compare": ("quantity_a", "quantity_b"),
}
OPTIONAL: dict[str, tuple[str, ...]] = {"math_solve": ("variable",)}

DECIMAL_DIGITS = 10
_MAX_DIGITS_SHOWN = 30
_MAX_STEPS = 3
# Values a free variable is tried at by math_compare: both signs, zero, fractions, a big one.
_SAMPLE_VALUES = (-3, -1, sp.Rational(-1, 2), 0, sp.Rational(1, 2), 1, 2, 5)
_MAX_SAMPLE_POINTS = 64

_ALLOWED = (
    "numbers, + - * / ^ !, parentheses, single-letter variables, the constants "
    + ", ".join(sorted(CONSTANTS))
    + " and the functions "
    + ", ".join(sorted({"sqrt", "abs", "log", "ln", "exp", "factorial", "binomial", "gcd", "lcm"}))
)
_UNICODE_FIXES = {
    "×": "write `*`",
    "÷": "write `/`",
    "−": "write a plain hyphen `-`",
    "–": "write a plain hyphen `-`",
    "√": "write `sqrt(...)`",
    "π": "write `pi`",
    "²": "write `^2`",
    "³": "write `^3`",
    "≤": "inequalities are not supported",
    "≥": "inequalities are not supported",
}


# ── errors the model can act on ─────────────────────────────────────────────


def _shown(text: str) -> str:
    """The offending text, shortened so an error never repeats a whole paragraph."""
    text = text.strip()
    return text if len(text) <= 60 else text[:57] + "..."


def explain_parse_failure(text: str, field: str) -> str:
    """
    Say why `text` is not an accepted expression and what to write instead. The checks run
    from the most specific cause to the generic one, so the message names the real problem.
    """
    body = text.strip()
    if not body:
        return f"`{field}` is empty; pass a math expression such as `(1 + 15/100) * (1 - 20/100)`."
    if len(text) > MAX_EXPRESSION_LENGTH:
        return (
            f"`{field}` is {len(text)} characters (limit {MAX_EXPRESSION_LENGTH}); "
            "split the problem into smaller steps and call the tool once per step."
        )
    if "%" in body:
        return (
            f"could not parse `{_shown(body)}`: `%` is not supported; write 15% as `15/100`, "
            "so '15% of 80' is `15/100 * 80`."
        )
    for char, fix in _UNICODE_FIXES.items():
        if char in body:
            return f"could not parse `{_shown(body)}`: the symbol `{char}` is not supported; {fix}."
    if any(c in body for c in "[]{}"):
        return f"could not parse `{_shown(body)}`: group with ( ) only, not [ ] or {{ }}."
    if body.count("(") != body.count(")"):
        return (
            f"could not parse `{_shown(body)}`: {body.count('(')} opening and "
            f"{body.count(')')} closing parentheses; every ( needs a matching )."
        )
    if re.search(r"[-+*/^]\s*$", body):
        return (
            f"could not parse `{_shown(body)}`: it ends with the operator `{body[-1]}`; "
            "add the missing operand (write powers as `x^2`)."
        )
    if re.search(r"[<>]", body):
        return (
            f"could not parse `{_shown(body)}`: inequalities are not supported; "
            "solve the boundary equation instead, or compare two values with math_compare."
        )
    unknown = [
        w for w in re.findall(r"[A-Za-z]{2,}", body)
        if w not in FUNCTIONS and w not in CONSTANTS
    ]
    if unknown:
        return (
            f"could not parse `{_shown(body)}`: `{unknown[0]}` is not a known function or "
            f"constant. Allowed: {_ALLOWED}. Write each variable as one letter."
        )
    return f"could not parse `{_shown(body)}`. Allowed: {_ALLOWED}."


def _parse(text: Any, field: str) -> tuple[sp.Expr | None, str | None]:
    if not isinstance(text, str):
        return None, f"`{field}` must be a string, for example `2 + 3 * 4`."
    try:
        return parse_math(text), None
    except MathSyntaxError:
        return None, explain_parse_failure(text, field)


# ── formatting ──────────────────────────────────────────────────────────────


def _text(expr: sp.Basic) -> str:
    """Plain-text math the tutor prompt asks for: `x^2`, not `x**2`."""
    return str(expr).replace("**", "^")


def _decimal(value: sp.Expr) -> str:
    """A rounded decimal, trailing zeros removed (`1.15`, not `1.150000000`)."""
    shown = str(sp.N(value, DECIMAL_DIGITS))
    if "." in shown and "e" not in shown:
        shown = shown.rstrip("0").rstrip(".")
    return shown


def _is_plain_number(value: sp.Expr) -> bool:
    return bool(value.is_number and value.is_real)


def _describe(value: sp.Expr) -> dict[str, str]:
    """`value` is what to quote; `exact` is added only when it says more than `value` does."""
    if value.free_symbols or not _is_plain_number(value):
        return {"value": _text(value)}  # symbolic or complex: nothing to round
    if value.is_Integer:
        digits = len(str(abs(int(value))))
        if digits > _MAX_DIGITS_SHOWN:
            return {"value": _decimal(value), "exact": f"an integer with {digits} digits"}
        return {"value": str(value)}
    decimal = _decimal(value)
    exact = _text(value)
    return {"value": decimal} if decimal == exact else {"value": decimal, "exact": exact}


def _undefined(value: sp.Basic) -> bool:
    return bool(value.has(sp.zoo, sp.nan, sp.oo, -sp.oo))


# ── the tools ───────────────────────────────────────────────────────────────


def math_evaluate(expression: Any) -> dict[str, Any]:
    expr, error = _parse(expression, "expression")
    if error or expr is None:
        if isinstance(expression, str) and "=" in expression:
            return {
                "error": f"`{_shown(expression)}` contains `=`, so it is an equation; "
                "use math_solve to find the unknown, or remove the `=` to evaluate one side."
            }
        return {"error": error}
    value = sp.simplify(expr)
    if _undefined(value):
        return {"error": "the expression is undefined (division by zero or an infinite value)."}
    result = _describe(value)
    if value.free_symbols:
        names = ", ".join(sorted(str(s) for s in value.free_symbols))
        result["note"] = f"symbolic result: it still contains {names}; no single number exists."
    return result


def _steps(equation_lhs_minus_rhs: sp.Expr, symbol: sp.Symbol) -> list[str]:
    """At most a few short lines, only for polynomials where factoring teaches something."""
    try:
        poly = sp.Poly(equation_lhs_minus_rhs, symbol)
    except sp.PolynomialError:
        return []
    if poly.degree() < 2:
        return []
    lines = [f"rearranged: {_text(equation_lhs_minus_rhs)} = 0"]
    factored = sp.factor(equation_lhs_minus_rhs)
    if factored != equation_lhs_minus_rhs and not factored.is_Atom:
        lines.append(f"factored: {_text(factored)} = 0")
    if poly.degree() == 2:
        a, b, c = poly.all_coeffs()
        lines.append(f"discriminant b^2 - 4ac = {sp.simplify(b**2 - 4 * a * c)}")
    return lines[:_MAX_STEPS]


def _split_equation(equation: str) -> tuple[str, str] | str:
    """(left, right) of exactly one `=`; a bare expression means `expression = 0`."""
    if re.search(r"[<>]|==|!=", equation):
        return (
            "inequalities and `==` are not supported; write one equation with a single `=`, "
            "or solve the boundary equation."
        )
    parts = equation.split("=")
    if len(parts) == 1:
        return equation, "0"
    if len(parts) > 2:
        return (
            "an equation has exactly one `=`; solve a system by substituting one equation "
            "into the other first."
        )
    if not parts[0].strip() or not parts[1].strip():
        return "one side of the `=` is empty; write both sides, for example `x^2 = 4`."
    return parts[0], parts[1]


def _root_order(root: sp.Expr) -> tuple[bool, float, str]:
    """Numbers first, smallest to largest; anything symbolic after, by name."""
    if _is_plain_number(root):
        return (False, float(root), "")
    return (True, 0.0, str(root))


def math_solve(equation: Any, variable: Any = None) -> dict[str, Any]:
    if not isinstance(equation, str):
        return {"error": "`equation` must be a string, for example `x^2 - 5x + 6 = 0`."}
    sides = _split_equation(equation)
    if isinstance(sides, str):
        return {"error": sides}
    left, error = _parse(sides[0], "equation")
    right, error_right = _parse(sides[1], "equation")
    if left is None or right is None:
        return {"error": error or error_right}
    difference = sp.expand(left - right)
    free = sorted(difference.free_symbols, key=str)
    if variable is not None and variable != "":
        if not isinstance(variable, str) or not re.fullmatch(r"[A-Za-z]", variable):
            return {"error": "`variable` must be a single letter such as `x`."}
        symbol = sp.Symbol(variable)
        if symbol not in free:
            have = ", ".join(str(s) for s in free) or "none"
            return {
                "error": f"`{variable}` does not appear in the equation; "
                f"its variables are: {have}."
            }
    elif len(free) == 1:
        symbol = free[0]
    elif not free:
        return {
            "error": "there is no variable to solve for; to check whether two numbers are "
            "equal use math_compare, to get a number use math_evaluate."
        }
    else:
        names = ", ".join(str(s) for s in free)
        return {
            "error": f"the equation has several variables ({names}); pass `variable` to say "
            "which one to solve for. The others are treated as constants."
        }
    if len(free) > 1:  # the other letters are constants: an expression in them, not a set of reals
        solutions: sp.Basic = sp.FiniteSet(*sp.solve(difference, symbol))
    else:
        solutions = sp.solveset(difference, symbol, domain=sp.S.Reals)
    if solutions == sp.S.Reals:
        return {
            "variable": str(symbol),
            "roots": [],
            "note": "every real value satisfies it (an identity).",
        }
    if solutions == sp.S.EmptySet:
        out: dict[str, Any] = {"variable": str(symbol), "roots": [], "note": "no real solutions."}
        complex_roots = sp.solveset(difference, symbol, domain=sp.S.Complexes)
        if isinstance(complex_roots, sp.FiniteSet) and complex_roots:
            out["note"] = "no real solutions (it has complex roots, which the GRE does not use)."
        return out
    if not isinstance(solutions, sp.FiniteSet):
        return {
            "error": "could not list the solutions as a finite set; check the equation, or "
            "solve a simpler form of it."
        }
    roots = sorted(solutions, key=_root_order)
    result: dict[str, Any] = {"variable": str(symbol), "roots": [_text(r) for r in roots]}
    if any(_is_plain_number(r) and not r.is_Integer for r in roots):
        result["decimals"] = [_decimal(r) if _is_plain_number(r) else _text(r) for r in roots]
    steps = _steps(difference, symbol)
    if steps:
        result["steps"] = steps
    return result


def _verdict(sign: int) -> dict[str, str]:
    if sign > 0:
        return {"verdict": "A", "meaning": "Quantity A is greater"}
    if sign < 0:
        return {"verdict": "B", "meaning": "Quantity B is greater"}
    return {"verdict": "C", "meaning": "the two quantities are equal"}


def _sign_at(difference: sp.Expr, point: dict[sp.Symbol, sp.Expr]) -> int | None:
    value = sp.simplify(difference.subs(point))
    if _undefined(value) or not value.is_real:
        return None
    return int(sp.sign(value))


def math_compare(quantity_a: Any, quantity_b: Any) -> dict[str, Any]:
    a, error = _parse(quantity_a, "quantity_a")
    if a is None:
        return {"error": error}
    b, error = _parse(quantity_b, "quantity_b")
    if b is None:
        return {"error": error}
    difference = sp.simplify(a - b)
    if _undefined(difference):
        return {"error": "one of the quantities is undefined (division by zero or infinite)."}
    free = sorted(difference.free_symbols, key=str)
    if not free:
        if not difference.is_real:
            return {"error": "the quantities are not real numbers, so they cannot be ordered."}
        sign = int(sp.sign(difference))
        return {
            **_verdict(sign),
            "a": _describe(sp.simplify(a))["value"],
            "b": _describe(sp.simplify(b))["value"],
            "why": f"A - B = {_describe(difference)['value']}",
        }
    if difference == 0:
        return {**_verdict(0), "why": "A - B simplifies to 0 for every value of the variables."}
    names = ", ".join(str(s) for s in free)
    seen: dict[int, str] = {}
    grid = itertools.islice(itertools.product(_SAMPLE_VALUES, repeat=len(free)), _MAX_SAMPLE_POINTS)
    for values in grid:
        point = dict(zip(free, values, strict=True))
        at_point = _sign_at(difference, point)
        if at_point is not None and at_point not in seen:
            seen[at_point] = ", ".join(f"{s}={v}" for s, v in point.items())
    if len(seen) > 1:
        examples = "; ".join(
            f"at {where}: {_verdict(sign)['meaning']}"
            for sign, where in sorted(seen.items(), reverse=True)
        )
        return {
            "verdict": "D",
            "meaning": "cannot be determined from the quantities alone",
            "why": f"the comparison changes with {names}: {examples}.",
            "note": "if the problem restricts the variable (for example x > 0), check that these "
            "values satisfy the restriction before answering D.",
        }
    return {
        "verdict": "undetermined",
        "why": f"A - B = {_text(difference)}; it kept the same sign at every value tried for "
        f"{names}, which is evidence, not proof. Reason about it algebraically, or try more "
        "values with math_evaluate.",
    }


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "math_evaluate": math_evaluate,
    "math_solve": math_solve,
    "math_compare": math_compare,
}


def check_arguments(name: str, arguments: Any) -> str | None:
    """A message for the model when the call itself is malformed, else None."""
    if name not in TOOLS:
        return f"unknown tool `{name}`; the tools are: {', '.join(TOOLS)}."
    if not isinstance(arguments, dict):
        return f"arguments for `{name}` must be a JSON object."
    allowed = REQUIRED[name] + OPTIONAL.get(name, ())
    missing = [k for k in REQUIRED[name] if k not in arguments]
    if missing:
        return f"`{name}` is missing required argument `{missing[0]}`."
    extra = [k for k in arguments if k not in allowed]
    if extra:
        return f"`{name}` has no argument `{extra[0]}`; its arguments are: {', '.join(allowed)}."
    return None


def run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one tool. Never raises: any failure becomes an error the model can read."""
    problem = check_arguments(name, arguments)
    if problem:
        return {"error": problem}
    try:
        return TOOLS[name](**arguments)
    except Exception as exc:
        return {
            "error": f"`{name}` could not compute this ({type(exc).__name__}); "
            "try a simpler form."
        }
