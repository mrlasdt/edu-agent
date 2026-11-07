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

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.gateway.src.gateway.guardrails.events import emit_guardrail_event
from services.gateway.src.gateway.guardrails.input import (
    COMPLETENESS_MAX_TOKENS,
    DEFAULT_CLARIFICATION,
    CompletenessVerdict,
    check_completeness,
    check_content_safety,
)
from services.gateway.src.gateway.guardrails.output import (
    LEAK_CHECK_MAX_TOKENS,
    check_answer_leak,
    redact_pii,
)
from services.gateway.src.gateway.guardrails.rate_limit import (
    RateLimiter,
)
from services.gateway.src.gateway.guardrails.verdict import Verdict
from services.gateway.src.gateway.settings import GatewaySettings as Settings
from shared.src.shared.config import clear_settings_caches
from shared.src.shared.models import Message, MessageRole, Mode, Session
from shared.src.shared.structured import StructuredOutputError
from shared.src.shared.turn_meter import TurnMeter
from shared.src.shared.turn_outcome import Outcome

_INPUT = "services.gateway.src.gateway.guardrails.input"


def make_session(mode: Mode = Mode.tutor) -> Session:
    return Session(candidate_id="c1", test_type="gre", section="quant", mode=mode)


# ── 1. Input content safety ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unsafe_input_blocked():
    with patch(f"{_INPUT}.get_settings", return_value=Settings(env="dev", openai_api_key="k")), \
         patch(f"{_INPUT}._call_moderation_api", new_callable=AsyncMock) as mock_mod:
        mock_mod.return_value = {"flagged": True, "categories": {"violence": True}}
        verdict = await check_content_safety("I want to hurt someone", "trace-1")
    assert verdict == Verdict.stop(
        Outcome.content_blocked, "I can't help with that request (content flagged: violence)."
    )


@pytest.mark.asyncio
async def test_safe_input_passes():
    with patch(f"{_INPUT}.get_settings", return_value=Settings(env="dev", openai_api_key="k")), \
         patch(f"{_INPUT}._call_moderation_api", new_callable=AsyncMock) as mock_mod:
        mock_mod.return_value = {"flagged": False, "categories": {}}
        verdict = await check_content_safety("Solve x^2 - 5x + 6 = 0", "trace-2")
    assert verdict == Verdict.allow()


@pytest.mark.asyncio
async def test_moderation_is_skipped_and_recorded_in_dev_without_an_openai_key():
    meter = TurnMeter("trace-skip", "gateway", "gre/quant", "tutor")
    with patch(f"{_INPUT}.get_settings", return_value=Settings(env="dev", openai_api_key="")), \
         patch(f"{_INPUT}._call_moderation_api", new_callable=AsyncMock) as mock_mod:
        verdict = await check_content_safety("Solve x + 1 = 2", "trace-skip", meter)
    mock_mod.assert_not_called()
    assert verdict == Verdict.allow()
    assert meter.snapshot()["moderation_skipped"] is True


@pytest.mark.asyncio
async def test_moderation_still_runs_in_dev_when_the_key_is_set():
    meter = TurnMeter("trace-key", "gateway", "gre/quant", "tutor")
    with patch(f"{_INPUT}.get_settings", return_value=Settings(env="dev", openai_api_key="k")), \
         patch(f"{_INPUT}._call_moderation_api", new_callable=AsyncMock) as mock_mod:
        mock_mod.return_value = {"flagged": False, "categories": {}}
        await check_content_safety("Solve x + 1 = 2", "trace-key", meter)
    mock_mod.assert_awaited_once()
    assert "moderation_skipped" not in meter.snapshot()


@pytest.mark.asyncio
@pytest.mark.parametrize("env", ["prod", "staging"])
async def test_moderation_is_never_skipped_outside_dev(env):
    # Production behaviour is unchanged: no key means the moderation call is made (and fails).
    meter = TurnMeter("trace-prod", "gateway", "gre/quant", "tutor")
    with patch(f"{_INPUT}.get_settings", return_value=Settings(env=env, openai_api_key="")), \
         patch(f"{_INPUT}._call_moderation_api", new_callable=AsyncMock) as mock_mod:
        mock_mod.side_effect = RuntimeError("401 no key")
        with pytest.raises(RuntimeError):
            await check_content_safety("Solve x + 1 = 2", "trace-prod", meter)
    mock_mod.assert_awaited_once()
    assert "moderation_skipped" not in meter.snapshot()


# ── 2. Incomplete question enrichment ────────────────────────────────────────

@pytest.mark.asyncio
async def test_incomplete_quant_question_returns_clarification():
    with patch("services.gateway.src.gateway.guardrails.input._call_enrichment_llm",
               new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = CompletenessVerdict(
            complete=False, clarification="What is the full problem?"
        )
        verdict = await check_completeness("solve this", make_session(), "trace-3")
    assert verdict == Verdict.stop(Outcome.clarification_requested, "What is the full problem?")


@pytest.mark.asyncio
async def test_incomplete_question_without_a_clarification_gets_the_default_one():
    with patch("services.gateway.src.gateway.guardrails.input._call_enrichment_llm",
               new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = CompletenessVerdict(complete=False)
        verdict = await check_completeness("solve this", make_session(), "trace-3b")
    assert verdict == Verdict.stop(Outcome.clarification_requested, DEFAULT_CLARIFICATION)


@pytest.mark.asyncio
async def test_complete_question_passes():
    with patch("services.gateway.src.gateway.guardrails.input._call_enrichment_llm",
               new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = CompletenessVerdict(complete=True)
        verdict = await check_completeness(
            "Solve x^2 - 5x + 6 = 0 for all real values of x.", make_session(), "trace-4"
        )
    assert verdict.allowed is True


def _llm_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


@pytest.mark.asyncio
async def test_enrichment_llm_sees_history_and_carries_trace_metadata():
    session = make_session()
    session.history = [
        Message(role=MessageRole.candidate, content="Solve 2x + 3 = 11."),
        Message(role=MessageRole.agent, content="Subtract 3 from both sides: 2x = 8."),
    ]
    with patch(
        "litellm.acompletion",
        new_callable=AsyncMock,
    ) as mock_llm:
        mock_llm.return_value = _llm_response('{"complete": true, "clarification": ""}')
        verdict = await check_completeness("why did we subtract 3?", session, "trace-hist")
    assert verdict.allowed is True
    kwargs = mock_llm.call_args.kwargs
    assert [m["role"] for m in kwargs["messages"]] == ["system", "user", "assistant", "user"]
    assert kwargs["messages"][1]["content"] == "Solve 2x + 3 = 11."
    assert kwargs["messages"][-1]["content"] == "why did we subtract 3?"
    assert kwargs["metadata"]["trace_id"] == "trace-hist"
    assert "prompt_version" in kwargs["metadata"]


@pytest.mark.asyncio
async def test_completeness_check_caps_its_output_tokens():
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _llm_response('{"complete": true, "clarification": ""}')
        await check_completeness("Solve 2x = 8.", make_session(), "trace-max")
    assert mock_llm.call_args.kwargs["max_tokens"] == COMPLETENESS_MAX_TOKENS


@pytest.mark.asyncio
async def test_enrichment_llm_history_is_capped_to_recent_turns():
    session = make_session()
    session.history = [
        Message(role=MessageRole.candidate, content=f"turn {i}") for i in range(20)
    ]
    with patch(
        "litellm.acompletion",
        new_callable=AsyncMock,
    ) as mock_llm:
        mock_llm.return_value = _llm_response('{"complete": true, "clarification": ""}')
        await check_completeness("next", session, "trace-cap")
    contents = [m["content"] for m in mock_llm.call_args.kwargs["messages"][1:]]
    assert contents == [f"turn {i}" for i in range(14, 20)] + ["next"]


# A reply that is not a valid CompletenessVerdict must never read as "complete": the check raises
# (and logs a guardrail "error" event) so the caller sees a failure, not a pass.
_BAD_REPLIES = [
    pytest.param('{"complete": "false", "clarification": "Which problem?"}', id="false-as-string"),
    pytest.param('{"complete": "true"}', id="true-as-string"),
    pytest.param('{"complete": 0}', id="number-not-bool"),
    pytest.param('{"clarification": "Which problem?"}', id="complete-missing"),
    pytest.param('{"complete": null}', id="complete-null"),
    pytest.param("Sure! The question looks complete.", id="prose"),
    pytest.param('{"complete": false, "clarification": "Wh', id="cut-off"),
    pytest.param("", id="empty"),
    pytest.param(None, id="no-content"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", _BAD_REPLIES)
async def test_an_invalid_completeness_reply_is_an_error_never_complete(reply, caplog):
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm, \
            caplog.at_level(logging.INFO, logger="guardrails"):
        mock_llm.return_value = _llm_response(reply)
        with pytest.raises(StructuredOutputError):
            await check_completeness("solve this", make_session(), "trace-bad")
    assert "guardrail.input.completeness.error trace_id=trace-bad" in caplog.text
    assert "guardrail.input.completeness.pass" not in caplog.text


@pytest.mark.asyncio
async def test_a_fenced_completeness_reply_is_read():
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _llm_response(
            '```json\n{"complete": false, "clarification": "Which problem?"}\n```'
        )
        verdict = await check_completeness("solve this", make_session(), "trace-fence")
    assert verdict == Verdict.stop(Outcome.clarification_requested, "Which problem?")


@pytest.mark.asyncio
async def test_a_model_with_schema_support_is_asked_for_the_schema():
    # The pinned dev config model is gpt-4o-mini.
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _llm_response('{"complete": true}')
        await check_completeness("Solve 2x = 8.", make_session(), "trace-schema")
    assert mock_llm.call_args.kwargs["response_format"] is CompletenessVerdict


@pytest.mark.asyncio
async def test_a_model_without_schema_support_gets_json_object_and_the_same_validation(
    monkeypatch,
):
    # The default config's free OpenRouter model: litellm does not list it as schema-capable.
    monkeypatch.setenv("MODEL_CONFIG_PATH", "config/model_config.free.yaml")
    clear_settings_caches()
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _llm_response('{"complete": "false"}')
        with pytest.raises(StructuredOutputError):
            await check_completeness("solve this", make_session(), "trace-free")
    assert mock_llm.call_args.kwargs["response_format"] == {"type": "json_object"}


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


@pytest.mark.parametrize(
    "phone",
    [
        "555-123-4567",
        "555.123.4567",
        "555 123 4567",
        "(555) 123-4567",
        "(555)123-4567",
        "+1 555 123 4567",
        "+1-555-123-4567",
        "+15551234567",
        "0912345678",
    ],
)
def test_formatted_phone_numbers_redacted(phone):
    result = redact_pii(f"Call {phone}, thanks.")
    assert result == "Call [REDACTED], thanks."


@pytest.mark.parametrize(
    "text",
    [
        "2^33 = 8589934592 and 3141592653",
        "The product is 1234567890.",
        "x = 5551234567",
        "5551234567*2 = 11102469134",
        "12345678901234",
        "1.5551234567",
    ],
)
def test_bare_digit_runs_are_not_phone_numbers(text):
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
    assert leaked.allowed is False


@pytest.mark.asyncio
async def test_no_leak_check_in_solve_mode():
    with patch("services.gateway.src.gateway.guardrails.output._call_leak_checker",
               new_callable=AsyncMock) as mock_check:
        leaked = await check_answer_leak(
            "The answer is x = 2.", mode=Mode.solve, trace_id="trace-6"
        )
    mock_check.assert_not_called()
    assert leaked.allowed is True


@pytest.mark.asyncio
async def test_no_leak_in_safe_hint():
    with patch("services.gateway.src.gateway.guardrails.output._call_leak_checker",
               new_callable=AsyncMock) as mock_check:
        mock_check.return_value = False
        leaked = await check_answer_leak(
            "What happens when you factor the left side?", mode=Mode.tutor, trace_id="trace-7"
        )
    assert leaked.allowed is True


@pytest.mark.asyncio
async def test_regex_stage_flags_leak_without_calling_the_llm():
    with patch("services.gateway.src.gateway.guardrails.output._call_leak_checker",
               new_callable=AsyncMock) as mock_check:
        leaked = await check_answer_leak(
            "Therefore x equals four.", mode=Mode.tutor, trace_id="trace-8"
        )
    mock_check.assert_not_called()
    assert leaked.allowed is False


@pytest.mark.asyncio
async def test_llm_stage_is_skipped_when_provider_has_no_api_key():
    settings = Settings(openai_api_key="", anthropic_api_key="")
    with patch("services.gateway.src.gateway.guardrails.output.get_settings",
               return_value=settings), \
         patch("litellm.acompletion",
               new_callable=AsyncMock) as mock_llm:
        leaked = await check_answer_leak(
            "Try isolating the variable.", mode=Mode.tutor, trace_id="trace-9"
        )
    mock_llm.assert_not_called()
    assert leaked.allowed is True


@pytest.mark.asyncio
async def test_leak_check_caps_its_output_tokens():
    settings = Settings(openai_api_key="k", anthropic_api_key="k")
    with patch("services.gateway.src.gateway.guardrails.output.get_settings",
               return_value=settings), \
         patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _llm_response("SAFE")
        leaked = await check_answer_leak(
            "Try isolating the variable.", mode=Mode.tutor, trace_id="trace-cap"
        )
    assert mock_llm.call_args.kwargs["max_tokens"] == LEAK_CHECK_MAX_TOKENS
    assert leaked.allowed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, "", "  "])
async def test_leak_check_cut_off_before_a_verdict_is_not_a_leak_or_a_crash(content):
    # A reasoning model can spend the whole cap thinking and return no content at all.
    settings = Settings(openai_api_key="k", anthropic_api_key="k")
    with patch("services.gateway.src.gateway.guardrails.output.get_settings",
               return_value=settings), \
         patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _llm_response(content)
        leaked = await check_answer_leak(
            "Try isolating the variable.", mode=Mode.tutor, trace_id="trace-cut"
        )
    assert leaked.allowed is True


@pytest.mark.asyncio
async def test_llm_stage_failure_is_not_a_leak():
    settings = Settings(openai_api_key="k", anthropic_api_key="k")
    with patch("services.gateway.src.gateway.guardrails.output.get_settings",
               return_value=settings), \
         patch("litellm.acompletion",
               new_callable=AsyncMock, side_effect=RuntimeError("boom")):
        leaked = await check_answer_leak(
            "Try isolating the variable.", mode=Mode.tutor, trace_id="trace-10"
        )
    assert leaked.allowed is True


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


@pytest.mark.parametrize("reply", ["So x = 3", "Therefore x = 3", "Then y = -2"])
def test_a_stated_answer_with_no_closing_punctuation_is_still_a_leak(reply):
    """Found by the failure-seeded eval: the clause-end check required punctuation or a newline,
    so a reply that simply ended on the answer got through."""
    from services.gateway.src.gateway.guardrails.output import detect_answer_leak

    assert detect_answer_leak(reply)
