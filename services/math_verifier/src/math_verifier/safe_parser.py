"""
Restricted parser for model-supplied math.

The math verifier's input is text a language model wrote, and a user can steer that text
through the question. `sympy.parse_expr` hands its input to Python's `eval`, so passing it
raw lets `__import__("os").getpid()` run inside the verifier. This module is the only door
between that text and sympy:

1. Cap the length.
2. Tokenize against an allow-list: numbers, a few operators, parentheses, a short list of
   named functions and constants, and variables. A word made only of letters that is not a
   listed function or constant is split into single-letter variables (`xy` is `x*y`), but two
   such words in a row are rejected: that is prose ("import os"), not algebra. Anything
   else (underscores, digits inside a word, attribute dots, brackets, quotes) is rejected
   before parsing.
3. Rebuild the string from the validated tokens, so the checker and the parser see the same
   stream, and parse it with an explicit namespace that has no builtins.

To accept new notation, add the function to `FUNCTIONS` (or the constant to `CONSTANTS`).
Nothing outside those tables can be named in an expression.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

MAX_EXPRESSION_LENGTH = 200
NOT_MATH = "not a math expression"

TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

# Named functions a learner's answer may call. Extend here.
FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": sp.sqrt,
    "cbrt": sp.cbrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "log": sp.log,
    "ln": sp.log,
    "exp": sp.exp,
    "factorial": sp.factorial,
    "binomial": sp.binomial,
    "gcd": sp.gcd,
    "lcm": sp.lcm,
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "min": sp.Min,
    "max": sp.Max,
}

# Named constants. A bare single letter is otherwise a plain variable.
CONSTANTS: dict[str, Any] = {"pi": sp.pi, "E": sp.E, "I": sp.I}

# What the generated code may reference. `__builtins__` is set to an empty dict on purpose:
# `eval` would otherwise inject the real builtins into a namespace that lacks the key.
# Integer/Float/Symbol are what sympy's own number and symbol transformations emit.
_NAMESPACE: dict[str, Any] = {
    **FUNCTIONS,
    **CONSTANTS,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Symbol": sp.Symbol,
    "Function": sp.Function,
    "__builtins__": {},
}

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<number>\d+(?:\.\d*)?|\.\d+)
      | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
      | (?P<op>\*\*|[-+*/^()!,])
    )""",
    re.VERBOSE,
)


class MathSyntaxError(ValueError):
    """The text is not an expression in the allowed math notation."""

    def __init__(self) -> None:
        super().__init__(NOT_MATH)


def _tokenize(text: str) -> list[str]:
    if not text.strip() or len(text) > MAX_EXPRESSION_LENGTH:
        raise MathSyntaxError
    tokens: list[str] = []
    depth = 0
    pos = 0
    after_word = False
    while pos < len(text):
        if not text[pos:].strip():
            break
        match = _TOKEN.match(text, pos)
        if match is None:
            raise MathSyntaxError
        pos = match.end()
        name, op = match.group("name"), match.group("op")
        if name is not None and name not in FUNCTIONS and name not in CONSTANTS:
            if not name.isalpha():
                raise MathSyntaxError
            is_word = len(name) > 1
            if is_word and after_word:
                raise MathSyntaxError
            after_word = is_word
            tokens.extend(name)
            continue
        after_word = False
        if op == "(":
            depth += 1
        elif op == ")":
            depth -= 1
            if depth < 0:
                raise MathSyntaxError
        elif op == "," and depth == 0:
            # A comma separates function arguments only; value lists are split earlier.
            raise MathSyntaxError
        tokens.append("**" if op == "^" else match.group(0).strip())
    if depth != 0:
        raise MathSyntaxError
    return tokens


def parse_math(text: str) -> sp.Expr:
    """
    Parse one math expression, or raise `MathSyntaxError`.

    Never evaluates Python from `text`: the allow-list rejects every name that is not a
    listed function, constant or run of letters (split into single-letter variables), and the
    namespace has no builtins.
    """
    source = " ".join(_tokenize(text))
    try:
        parsed = parse_expr(
            source,
            local_dict={},
            global_dict=dict(_NAMESPACE),
            transformations=TRANSFORMS,
        )
    except Exception as exc:
        raise MathSyntaxError from exc
    if not isinstance(parsed, sp.Basic):
        raise MathSyntaxError
    return parsed
