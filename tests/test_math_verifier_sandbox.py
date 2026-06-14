"""
Tests for the math verifier sandbox (sympy-backed).

Behaviors under test (from issue 04 acceptance criteria):
  1. Correct quadratic equation verified
  2. Incorrect answer not verified
  3. Arithmetic expression evaluated and verified
  4. Malformed expression returns parse error (no crash)
  5. Timeout returns timeout error (no crash)
"""

import asyncio
import pytest
from unittest.mock import patch

from services.math_verifier.src.math_verifier.sandbox import (
    VerificationResult,
    verify_expression,
)


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
async def test_timeout_returns_timeout_error():
    # Simulate slow execution by patching the sync verify function to sleep
    import time

    def slow_verify(expression, expected):
        time.sleep(10)  # much longer than timeout
        return VerificationResult(verified=True, computed="")

    with patch(
        "services.math_verifier.src.math_verifier.sandbox._sync_verify",
        side_effect=slow_verify,
    ):
        result = await verify_expression("x = 1", "1", timeout=0.1)

    assert result.verified is False
    assert result.error == "timeout"


# ── 6. Result carries computed value ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_computed_field_populated_on_success():
    result = await verify_expression("x**2 - 5*x + 6 = 0", "2, 3")
    assert result.computed != ""
