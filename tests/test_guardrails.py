"""
Tests for the five-layer guardrail pipeline (issue 10).

Behaviors under test:
  1. Unsafe input → blocked, no LLM call
  2. Incomplete question → ClarificationTurn (Orchestrator enrichment)
  3. Output PII redaction: email and phone stripped
  4. Tutor-mode answer leak → detected, regenerate once, log event
  5. Rate limit: >60 turns/min → 429
  6. Daily quota: >200 turns/day → 429
  7. Every guardrail emits structured log event with trace_id
"""

import pytest
import logging
from unittest.mock import AsyncMock, patch

from services.gateway.src.gateway.guardrails.input import (
    check_content_safety,
    check_completeness,
)
from services.gateway.src.gateway.guardrails.output import (
    check_answer_leak,
    redact_pii,
)
from services.gateway.src.gateway.guardrails.rate_limit import (
    RateLimiter,
)
from services.gateway.src.gateway.guardrails.events import emit_guardrail_event
from shared.src.shared.models import Mode, Session


def make_session(mode: Mode = Mode.tutor) -> Session:
    return Session(candidate_id="c1", test_type="gre", section="quant", mode=mode)


# ── 1. Input content safety ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unsafe_input_blocked():
    with patch("services.gateway.src.gateway.guardrails.input._call_moderation_api",
               new_callable=AsyncMock) as mock_mod:
        mock_mod.return_value = {"flagged": True, "categories": {"violence": True}}
        safe, reason = await check_content_safety("I want to hurt someone", "trace-1")
    assert safe is False
    assert reason != ""


@pytest.mark.asyncio
async def test_safe_input_passes():
    with patch("services.gateway.src.gateway.guardrails.input._call_moderation_api",
               new_callable=AsyncMock) as mock_mod:
        mock_mod.return_value = {"flagged": False, "categories": {}}
        safe, reason = await check_content_safety("Solve x^2 - 5x + 6 = 0", "trace-2")
    assert safe is True


# ── 2. Incomplete question enrichment ────────────────────────────────────────

@pytest.mark.asyncio
async def test_incomplete_quant_question_returns_clarification():
    with patch("services.gateway.src.gateway.guardrails.input._call_enrichment_llm",
               new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {"complete": False, "clarification": "What is the full problem?"}
        complete, prompt = await check_completeness("solve this", make_session(), "trace-3")
    assert complete is False
    assert len(prompt) > 0


@pytest.mark.asyncio
async def test_complete_question_passes():
    with patch("services.gateway.src.gateway.guardrails.input._call_enrichment_llm",
               new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {"complete": True, "clarification": ""}
        complete, prompt = await check_completeness(
            "Solve x^2 - 5x + 6 = 0 for all real values of x.", make_session(), "trace-4"
        )
    assert complete is True


# ── 3. Output PII redaction ───────────────────────────────────────────────────

def test_email_redacted():
    text = "Contact me at alice@example.com for help."
    result = redact_pii(text)
    assert "alice@example.com" not in result
    assert "[REDACTED]" in result


def test_phone_redacted():
    text = "Call me at 555-123-4567 anytime."
    result = redact_pii(text)
    assert "555-123-4567" not in result
    assert "[REDACTED]" in result


def test_clean_text_unchanged():
    text = "The answer to the quadratic equation is x = 2 or x = 3."
    assert redact_pii(text) == text


def test_multiple_pii_all_redacted():
    text = "Email: bob@test.org, Phone: 123-456-7890"
    result = redact_pii(text)
    assert "bob@test.org" not in result
    assert "123-456-7890" not in result


# ── 4. Answer leak detection ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_answer_leak_detected_in_tutor_mode():
    with patch("services.gateway.src.gateway.guardrails.output._call_leak_checker",
               new_callable=AsyncMock) as mock_check:
        mock_check.return_value = True  # leak detected
        leaked = await check_answer_leak(
            "The answer is x = 2.", mode=Mode.tutor, trace_id="trace-5"
        )
    assert leaked is True


@pytest.mark.asyncio
async def test_no_leak_check_in_solve_mode():
    with patch("services.gateway.src.gateway.guardrails.output._call_leak_checker",
               new_callable=AsyncMock) as mock_check:
        leaked = await check_answer_leak(
            "The answer is x = 2.", mode=Mode.solve, trace_id="trace-6"
        )
    mock_check.assert_not_called()
    assert leaked is False


@pytest.mark.asyncio
async def test_no_leak_in_safe_hint():
    with patch("services.gateway.src.gateway.guardrails.output._call_leak_checker",
               new_callable=AsyncMock) as mock_check:
        mock_check.return_value = False
        leaked = await check_answer_leak(
            "What happens when you factor the left side?", mode=Mode.tutor, trace_id="trace-7"
        )
    assert leaked is False


# ── 5. Rate limit ─────────────────────────────────────────────────────────────

def test_rate_limit_allows_under_threshold():
    limiter = RateLimiter(max_per_minute=60, max_per_day=200)
    for _ in range(59):
        allowed, _ = limiter.check("cand-1")
    allowed, _ = limiter.check("cand-1")
    assert allowed is True


def test_rate_limit_blocks_over_per_minute():
    limiter = RateLimiter(max_per_minute=3, max_per_day=200)
    limiter.check("cand-2")
    limiter.check("cand-2")
    limiter.check("cand-2")
    allowed, msg = limiter.check("cand-2")
    assert allowed is False
    assert "rate limit" in msg.lower() or "429" in msg or "too many" in msg.lower()


def test_daily_quota_blocks_over_limit():
    limiter = RateLimiter(max_per_minute=1000, max_per_day=3)
    limiter.check("cand-3")
    limiter.check("cand-3")
    limiter.check("cand-3")
    allowed, msg = limiter.check("cand-3")
    assert allowed is False
    assert len(msg) > 0


def test_different_candidates_tracked_independently():
    limiter = RateLimiter(max_per_minute=2, max_per_day=200)
    limiter.check("cand-a")
    limiter.check("cand-a")
    # cand-a is at limit; cand-b should still be allowed
    allowed, _ = limiter.check("cand-b")
    assert allowed is True


# ── 6. Structured log events ──────────────────────────────────────────────────

def test_guardrail_event_logged(caplog):
    with caplog.at_level(logging.INFO):
        emit_guardrail_event(
            layer="input", name="content_safety", result="pass", trace_id="trace-99"
        )
    assert any(
        "guardrail" in r.message and "content_safety" in r.message and "trace-99" in r.message
        for r in caplog.records
    )


def test_guardrail_event_includes_fail_result(caplog):
    with caplog.at_level(logging.INFO):
        emit_guardrail_event(
            layer="generation", name="tutor_mode_answer_leak", result="fail", trace_id="trace-100"
        )
    assert any("fail" in r.message for r in caplog.records)
