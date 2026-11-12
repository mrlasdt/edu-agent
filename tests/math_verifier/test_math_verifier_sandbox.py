"""
Tests for the math verifier sandbox (sympy-backed).

Behaviors under test (from issue 04 acceptance criteria):
  1. Correct quadratic equation verified
  2. Incorrect answer not verified
  3. Arithmetic expression evaluated and verified
  4. Malformed expression returns parse error (no crash)
  5. Timeout returns timeout error (no crash)
"""

import multiprocessing

import pytest

from services.math_verifier.src.math_verifier.sandbox import verify_expression

# ── 1. Correct quadratic ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_correct_quadratic_equation():
    result = await verify_expression("x**2 - 5*x + 6 = 0", "x=2 or x=3")
    assert result.verified is True
    assert result.error is None


@pytest.mark.asyncio
async def test_correct_quadratic_comma_notation():
    result = await verify_expression("x**2 - 5*x + 6 = 0", "2, 3")
    assert result.verified is True


# ── 2. Incorrect answer ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_incorrect_quadratic_answer():
    result = await verify_expression("x**2 + 1 = 0", "x=1")
    assert result.verified is False
    assert result.error is None  # not an error — just wrong


@pytest.mark.asyncio
async def test_wrong_root_for_quadratic():
    result = await verify_expression("x**2 - 5*x + 6 = 0", "x=1 or x=2")
    assert result.verified is False


# ── 3. Arithmetic expression ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_arithmetic_correct():
    result = await verify_expression("2 + 3 * 4", "14")
    assert result.verified is True


@pytest.mark.asyncio
async def test_arithmetic_incorrect():
    result = await verify_expression("2 + 3 * 4", "20")
    assert result.verified is False


# ── 4. Malformed expression → parse error, no crash ──────────────────────────

@pytest.mark.asyncio
async def test_malformed_expression_returns_parse_error():
    result = await verify_expression("not valid math $$%", "5")
    assert result.verified is False
    assert result.error is not None
    assert "parse error" in result.error.lower()


@pytest.mark.asyncio
async def test_empty_expression_returns_parse_error():
    result = await verify_expression("", "5")
    assert result.verified is False
    assert result.error is not None


# ── 5. Timeout → timeout error, no crash ─────────────────────────────────────

@pytest.mark.asyncio
async def test_timeout_returns_timeout_error_and_kills_the_worker():
    # A real runaway computation: the worker process must be killed, not left running.
    result = await verify_expression("9**9**9**9", "1", timeout=0.5)

    assert result.verified is False
    assert result.error == "timeout"
    assert multiprocessing.active_children() == []  # the child was killed and reaped


# ── 6. Result carries computed value ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_computed_field_populated_on_success():
    result = await verify_expression("x**2 - 5*x + 6 = 0", "2, 3")
    assert result.computed != ""


# ── value lists as the expression (answer handoff) ───────────────────────────


@pytest.mark.parametrize("answer", ["2, 3", "2,3", "x = 2 or x = 3", "2 and 3", "[2, 3]"])
async def test_value_list_verifies_against_itself_and_matching_list(answer: str) -> None:
    result = await verify_expression(answer, "3, 2")
    assert result.verified is True and result.error is None


async def test_value_list_mismatch_and_prose_are_not_verified() -> None:
    assert (await verify_expression("2, 4", "2, 3")).verified is False
    prose = await verify_expression("The values of x are 2 and 3", "2, 3")
    assert prose.verified is False and prose.error is not None


@pytest.mark.parametrize(
    ("expression", "expected"), [("binomial(5, 2)", "10"), ("gcd(12, 18)", "6")]
)
async def test_multi_argument_function_calls_are_not_value_lists(
    expression: str, expected: str
) -> None:
    result = await verify_expression(expression, expected)
    assert result.verified is True and result.error is None
