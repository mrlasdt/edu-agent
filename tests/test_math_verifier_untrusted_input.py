"""
The math verifier's input is text a language model wrote, so it is untrusted.

Before the restricted parser, `verify_expression('__import__("os").getpid()', '1')` returned
the verifier's own process id in `computed`: the text ran as Python. These tests pin the fix:
hostile text is rejected as "not a math expression", never evaluated, and the valid notation
tutors depend on keeps working.
"""

import os

import pytest
import sympy as sp

from services.math_verifier.src.math_verifier.safe_parser import (
    MAX_EXPRESSION_LENGTH,
    MathSyntaxError,
    parse_math,
)
from services.math_verifier.src.math_verifier.sandbox import verify_expression
from shared.src.shared.events import Verified
from tests.agent_events import reply_of

HOSTILE = [
    '__import__("os").getpid()',
    '__import__("os").system("id")',
    "().__class__.__bases__[0].__subclasses__()",
    "open('/etc/passwd').read()",
    "eval('1+1')",
    "exec('import os')",
    "[c for c in ().__class__.__mro__]",
    "(lambda: 1)()",
    "x.__class__",
    "sympy.os.getpid()",
    "getattr(1, 'real')",
    "globals()",
    "x; import os",
    "1 if True else 2",
    "'a' * 3",
    "2 ** 3\nimport os",
    "x = __import__('os').getpid()",
    "9" * (MAX_EXPRESSION_LENGTH + 1),
]


@pytest.mark.parametrize("text", HOSTILE)
def test_parser_rejects_hostile_text(text: str) -> None:
    with pytest.raises(MathSyntaxError, match="not a math expression"):
        parse_math(text)


async def test_pid_probe_no_longer_returns_the_process_id() -> None:
    result = await verify_expression('__import__("os").getpid()', "1")

    assert result.verified is False
    assert result.error == "parse error: not a math expression"
    assert result.computed == ""
    assert str(os.getpid()) not in result.computed


@pytest.mark.parametrize("text", HOSTILE)
async def test_hostile_text_is_an_error_not_a_computation(text: str) -> None:
    for expression in (text, f"x = {text}", f"2, {text}"):
        result = await verify_expression(expression, "1")
        assert result.verified is False
        assert result.computed == ""
        assert result.error is not None
    # As the expected answer it is never run either: it parses to nothing, so nothing matches.
    assert (await verify_expression("1", text)).verified is False


async def test_hostile_text_cannot_leave_a_side_effect(tmp_path) -> None:
    marker = tmp_path / "pwned"
    await verify_expression(f"open({str(marker)!r}, 'w').write('x')", "1")
    await verify_expression(f"__import__('pathlib').Path({str(marker)!r}).touch()", "1")
    assert not marker.exists()


def test_juxtaposed_variables_are_a_product() -> None:
    x, y = sp.symbols("x y")
    assert parse_math("x y") == x * y


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2 + 3 * 4", sp.Integer(14)),
        ("x**2 - 5*x + 6", sp.Symbol("x") ** 2 - 5 * sp.Symbol("x") + 6),
        ("x^2", sp.Symbol("x") ** 2),
        ("3.14 * 2**2", sp.Float("12.56")),
        ("1/2 + 1/3", sp.Rational(5, 6)),
        ("sqrt(16)", sp.Integer(4)),
        ("sqrt 16", sp.Integer(4)),
        ("2pi", 2 * sp.pi),
        ("5!", sp.Integer(120)),
        ("binomial(5, 2)", sp.Integer(10)),
        ("gcd(12, 18)", sp.Integer(6)),
        ("2(x + 1)", 2 * (sp.Symbol("x") + 1)),
        ("-(3 - 5)", sp.Integer(2)),
        ("3xy", 3 * sp.Symbol("x") * sp.Symbol("y")),
        ("x^2y", sp.Symbol("x") ** 2 * sp.Symbol("y")),
        ("12 cm", 12 * sp.Symbol("c") * sp.Symbol("m")),
        ("45 degrees", 45 * sp.Mul(*sp.symbols("d e g r e e s"))),
    ],
)
def test_valid_notation_still_parses(text: str, expected: sp.Expr) -> None:
    assert sp.simplify(parse_math(text) - expected) == 0


@pytest.mark.parametrize("text", ["2, 3", "(2, 3)", "2 +", "((2)", "2)", "", "   "])
def test_malformed_text_is_rejected_cleanly(text: str) -> None:
    with pytest.raises(MathSyntaxError):
        parse_math(text)


async def test_value_lists_and_prose_handoffs_still_work() -> None:
    assert (await verify_expression("2, -2", "-2, 2")).verified is True
    assert (await verify_expression("x = 2 or x = 3", "3, 2")).verified is True
    assert (await verify_expression("The values are 2 and -2", "2, -2")).verified is False


async def test_concurrent_checks_each_get_their_own_answer() -> None:
    import asyncio

    results = await asyncio.gather(
        *[verify_expression(f"{n} + 1", str(n + 1)) for n in range(10)]
        + [verify_expression('__import__("os").getpid()', "1")]
    )
    assert [r.verified for r in results[:10]] == [True] * 10
    assert results[10].error == "parse error: not a math expression"


async def test_a_solve_turn_whose_final_answer_is_hostile_degrades_and_still_answers() -> None:
    """The model's reply carries the PID probe as its answer; the learner still gets a turn."""
    from types import SimpleNamespace
    from typing import Any
    from unittest.mock import AsyncMock, patch

    from evals.agent_eval.runner import in_process_verify_math
    from services.agent.src.agent.agents.quant import QuantAgent
    from shared.src.shared.models import Mode, Session

    reply = 'Working it out.\n**Answer: __import__("os").getpid()**'

    async def stream() -> Any:
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=reply))])

    async def respond(*args: Any, **kwargs: Any) -> Any:
        return stream()

    agent = QuantAgent(
        model_config_path="config/model_config.dev.yaml",
        verify_math=in_process_verify_math(),
    )
    session = Session(candidate_id="c1", test_type="gre", section="quant", mode=Mode.solve)
    with patch("litellm.acompletion", new=AsyncMock(side_effect=respond)):
        items = [i async for i in agent.run(session, "Solve it")]

    text = reply_of(items)
    assert "Working it out." in text  # the turn is still answered
    assert Verified(passed=False) in items
    assert str(os.getpid()) not in text
