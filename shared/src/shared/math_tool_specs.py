"""
The contract of the three math tools the Quant agent's model can call: `math_evaluate`,
`math_solve`, `math_compare`.

This is the one place a tool's name, description and argument schema are written, in the
OpenAI function-calling format litellm passes to every provider. Both sides import it: the
agent hands `MATH_TOOL_SPECS` to the model, and the math verifier builds the argument rules it
checks a call against (`required_arguments`, `optional_arguments`) from the same dicts, so a
changed parameter is one edit and the two sides cannot disagree. It lives in `shared` because
the agent and the math verifier are separate images that both ship `shared/` and nothing else
of each other; a module here costs no network call at agent startup.

The descriptions are written the way you would brief a new colleague who has never seen the
tool: what it is for, which of the three to reach for, what to type, one worked example, and
what comes back. A model picks tools from these words alone, so they are the part of the code
worth rewriting when the transcripts show the wrong tool being chosen.
"""
from __future__ import annotations

from typing import Any

_NOTATION = (
    "Notation: + - * / ^ and parentheses; write 15% as 15/100; sqrt(x), abs(x), log(x), "
    "exp(x), factorial(n) or n!, binomial(n, k), gcd(a, b), lcm(a, b); constants pi and E; "
    "variables are single letters."
)

MATH_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "math_evaluate",
            "description": (
                "Compute the exact value of an arithmetic or algebraic expression. Use it for "
                "every calculation with more than one step or any awkward number: percent "
                "chains, fractions, powers, roots, factorials, combinations, or simplifying an "
                "expression that has a variable. Do not do that arithmetic yourself. "
                "It does not take equations (use math_solve) and it does not compare two "
                "quantities (use math_compare). "
                f"{_NOTATION} "
                "Returns `value` (a number, rounded to 10 digits, or a simplified expression) "
                "and `exact` (the fraction or root form, only when it differs from `value`). "
                "Example: a price rises 15% then falls 20%. The new price over the old is "
                "expression `(1 + 15/100) * (1 - 20/100)`, which returns value 0.92, so the "
                "net change is -8%. On failure it returns `error` saying what to write instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "One expression, no `=` sign, at most 200 characters. "
                            "Example: `3/4 + 5/6`."
                        ),
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "math_solve",
            "description": (
                "Find the real solutions of one equation in one unknown, such as a linear or "
                "quadratic equation. Use it whenever the problem asks 'solve for x', 'which "
                "values satisfy', 'what is the root'. It gives every real root, so you never "
                "miss the second one or get a sign wrong. Pass the equation exactly as "
                "written, with a single `=`. If you give an expression with no `=` it is "
                "solved for 0. It does not take inequalities or systems: substitute one "
                "equation into the other first. For a plain numeric answer use math_evaluate. "
                f"{_NOTATION} "
                "Returns `roots` (exact, e.g. `5*sqrt(2)`), `decimals` when a root is not a "
                "whole number, and `steps` (rearranged form, factored form, discriminant) for "
                "a quadratic. `roots: []` with a `note` means there is no real solution. "
                "Example: equation `3x^2 - 7x + 2 = 0` returns roots 1/3 and 2."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {
                        "type": "string",
                        "description": (
                            "One equation with a single `=`. Example: `x^2 - 5x + 6 = 0`."
                        ),
                    },
                    "variable": {
                        "type": "string",
                        "description": (
                            "The single letter to solve for. Only needed when the equation has "
                            "more than one letter; the others are treated as constants."
                        ),
                    },
                },
                "required": ["equation"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "math_compare",
            "description": (
                "Decide a GRE Quantitative Comparison: which of Quantity A and Quantity B is "
                "greater, or whether they are equal. Use it instead of estimating, because two "
                "values that look close (square roots, fractions, percents of different bases) "
                "are exactly where mental comparison fails. Give each quantity as an "
                "expression. "
                f"{_NOTATION} "
                "Returns `verdict` (A, B, C for equal, or D for cannot be determined), "
                "`meaning`, the two values `a` and `b`, and `why`. When a quantity contains a "
                "variable, it tries values on both sides of zero: if the order changes it "
                "returns D with the values that show it, and you must check those values "
                "against any condition the problem gives (for example x > 0). If the order "
                "never changes it returns `undetermined`, which is evidence, not proof. "
                "Example: quantity_a `3^2`, quantity_b `2^3` returns verdict A (9 vs 8)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity_a": {
                        "type": "string",
                        "description": "Quantity A as an expression. Example: `15/100 * 80`.",
                    },
                    "quantity_b": {
                        "type": "string",
                        "description": "Quantity B as an expression. Example: `25/100 * 40`.",
                    },
                },
                "required": ["quantity_a", "quantity_b"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_NAMES = tuple(spec["function"]["name"] for spec in MATH_TOOL_SPECS)


def required_arguments(name: str) -> tuple[str, ...]:
    """The argument names tool `name` must be called with, in schema order."""
    function = _function(name)
    return tuple(function["parameters"]["required"])


def optional_arguments(name: str) -> tuple[str, ...]:
    """The argument names tool `name` may also take."""
    function = _function(name)
    required = set(function["parameters"]["required"])
    return tuple(arg for arg in function["parameters"]["properties"] if arg not in required)


def _function(name: str) -> dict[str, Any]:
    for spec in MATH_TOOL_SPECS:
        if spec["function"]["name"] == name:
            function: dict[str, Any] = spec["function"]
            return function
    raise KeyError(name)
