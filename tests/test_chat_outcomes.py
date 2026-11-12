"""
/chat's wire output and recorded outcome, pinned for every way a turn can end.

Each case runs the whole route (rate limiter, safety, completeness, orchestrator, agent stream,
reply guard) with only the model calls and the agent faked at their lowest seams, then compares
the exact streamed bytes and the gateway's turn record against literals. The literals are the
behaviour before `Outcome`, `Verdict` and the stage list replaced the free strings: a turn
report groups and old logs hold these exact strings, so they must not drift.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from services.gateway.src.gateway.guardrails.input import CompletenessVerdict
from services.gateway.src.gateway.guardrails.rate_limit import RateLimiter
from services.gateway.src.gateway.guardrails.stream import AGENT_UNAVAILABLE_NOTICE
from services.gateway.src.gateway.guardrails.supportive import SUPPORTIVE_MESSAGE
from services.gateway.src.gateway.main import app
from services.gateway.src.gateway.orchestrator import ClarificationTurn
from services.gateway.src.gateway.settings import GatewaySettings
from shared.src.shared.turn_outcome import Outcome
from tests.gateway_fakes import agent_replies, orchestrator_of, rate_limiter_of

_INPUT = "services.gateway.src.gateway.guardrails.input"
_OUTPUT = "services.gateway.src.gateway.guardrails.output"
_BODY = {"candidate_id": "c1", "test_type": "gre", "section": "quant", "message": "Solve 2x+3=11"}
_SAFE = {"flagged": False, "categories": {}}


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app, raise_server_exceptions=False)


def _run(
    client: TestClient,
    *,
    body: dict[str, Any] | None = None,
    reply: list[str] | Exception = ("Isolate x first.",),
    then_fail: Exception | None = None,
    moderation: dict[str, Any] | Exception = _SAFE,
    completeness: CompletenessVerdict | Exception = CompletenessVerdict(complete=True),
    llm_leak: bool = False,
    limiter: RateLimiter | None = None,
    env: str = "prod",
) -> Any:
    async def agent(payload: dict[str, Any], meta: dict[str, Any] | None = None) -> Any:
        if isinstance(reply, Exception):
            raise reply
        for token in reply:
            yield token.encode()
        if then_fail is not None:
            raise then_fail  # the connection to the agent breaks after some of the reply

    moderation_mock = (
        AsyncMock(side_effect=moderation)
        if isinstance(moderation, Exception)
        else AsyncMock(return_value=moderation)
    )
    completeness_mock = (
        AsyncMock(side_effect=completeness)
        if isinstance(completeness, Exception)
        else AsyncMock(return_value=completeness)
    )
    with (
        patch(f"{_INPUT}.get_settings", return_value=GatewaySettings(openai_api_key="k", env=env)),
        patch(f"{_INPUT}._call_moderation_api", moderation_mock),
        patch(f"{_INPUT}._call_enrichment_llm", completeness_mock),
        patch(f"{_OUTPUT}._call_leak_checker", AsyncMock(return_value=llm_leak)),
        agent_replies(agent),
        rate_limiter_of(limiter or RateLimiter()),
    ):
        return client.post("/chat", json={**_BODY, **(body or {})})


# (id, how to run it, exact streamed text, recorded outcome, recorded flags)
CASES = [
    pytest.param(
        {}, "Isolate x first.", "ok", [], id="ok",
    ),
    pytest.param(
        {"body": {"message": "mail me at jo@example.com, solve 2x+3=11"}},
        "[Privacy: removed an email address or phone number from your message before "
        "processing it.]\n\nIsolate x first.",
        "ok", ["input_pii_redacted"], id="ok-with-input-pii-notice",
    ),
    pytest.param(
        {"moderation": {"flagged": True, "categories": {"violence": True}}},
        "I can't help with that request (content flagged: violence).",
        "content_blocked", [], id="content_blocked",
    ),
    pytest.param(
        {"moderation": {"flagged": True, "categories": {"self-harm/intent": True}}},
        SUPPORTIVE_MESSAGE, "supportive_reply", [], id="supportive_reply",
    ),
    pytest.param(
        {"moderation": RuntimeError("moderation down")},
        "The safety check is unavailable right now. Please try again in a moment.",
        "moderation_unavailable", [], id="moderation_unavailable",
    ),
    pytest.param(
        {"completeness": CompletenessVerdict(complete=False, clarification="Which equation?")},
        "Which equation?", "clarification_requested", [], id="clarification-from-the-check",
    ),
    pytest.param(
        {"completeness": CompletenessVerdict(complete=False)},
        "Could you share the full question so I can help?",
        "clarification_requested", [], id="clarification-default-prompt",
    ),
    pytest.param(
        {"body": {"section": "verbal", "message": "hi"}},
        "I'm not set up to help with GRE verbal yet. "
        "Currently available: gre/aw, gre/quant, ielts/writing. "
        "Please start a new session with one of these.",
        "clarification_requested", [], id="clarification-from-the-orchestrator",
    ),
    pytest.param(
        {"completeness": RuntimeError("model down")},
        "Isolate x first.", "ok", ["completeness_check_failed"], id="completeness-fails-open",
    ),
    pytest.param(
        {"reply": ["Isolate x first.\n", "Subtract 3, then divide.\nThe ans", "wer is 4.\nMore"]},
        "Isolate x first.\nSubtract 3, then divide.\n\n\n[Answer withheld: in tutor mode I "
        "don't state the final answer. Ask for a hint, or use \"Show me the answer\".]",
        "answer_leak_blocked", [], id="answer_leak_blocked",
    ),
    pytest.param(
        {"llm_leak": True, "reply": ["Divide both sides by two."]},
        "Divide both sides by two.\n\n[Answer check: this reply may state the final answer. "
        "Ask for a hint if you want to work it out yourself.]",
        "ok", ["answer_leak_flagged"], id="llm-leak-flag-keeps-outcome-ok",
    ),
    pytest.param(
        {"reply": RuntimeError("agent exploded")},
        AGENT_UNAVAILABLE_NOTICE.lstrip(), "agent_unavailable", [], id="agent-unavailable",
    ),
    pytest.param(
        {"reply": ["Isolate x first.\n", "Subtract 3, then"], "then_fail": RuntimeError("reset")},
        "Isolate x first.\nSubtract 3, then" + AGENT_UNAVAILABLE_NOTICE,
        "agent_unavailable", [], id="agent-breaks-midstream",
    ),
]


@pytest.mark.parametrize("kwargs,text,outcome,flags", CASES)
def test_a_turn_streams_the_same_bytes_and_records_the_same_outcome(
    client, turn_log, kwargs, text, outcome, flags
):
    resp = _run(client, **kwargs)

    if text is not None:
        assert resp.status_code == 200
        assert resp.text == text
    (record,) = turn_log.read()
    assert record["outcome"] == outcome
    assert record["flags"] == flags


def test_rate_limited_turn_streams_the_limit_message_and_records_rate_limited(client, turn_log):
    limiter = RateLimiter(max_per_minute=1)
    _run(client, limiter=limiter)
    resp = _run(client, limiter=limiter)

    assert resp.text == (
        "Too many requests \u2014 rate limit exceeded (1/min). Please wait a moment and try again."
    )
    first, second = turn_log.read()
    assert (first["outcome"], second["outcome"]) == ("ok", "rate_limited")


def test_an_injected_orchestrator_decides_the_route(client, turn_log):
    class Unroutable:
        async def process(self, session: Any, message: str) -> ClarificationTurn:
            return ClarificationTurn(prompt="Pick a section first.", reason="test")

    with orchestrator_of(Unroutable()):
        resp = _run(client)

    assert resp.text == "Pick a section first."
    (record,) = turn_log.read()
    assert record["outcome"] == "clarification_requested"


def test_the_outcome_vocabulary_is_the_one_old_turn_logs_hold():
    assert {o.value for o in Outcome} == {
        "ok",
        "rate_limited",
        "moderation_unavailable",
        "supportive_reply",
        "content_blocked",
        "clarification_requested",
        "answer_leak_blocked",
        "agent_unavailable",
        "error",
        "client_cancelled",
        "no gateway record",
    }


def test_a_failing_agent_is_a_notice_and_the_record_says_where_it_failed(client, turn_log):
    """The learner used to get HTTP 200 and then a dropped connection (nothing to read); now the
    reply is a sentence, and the turn record keeps the failure for the operator."""
    resp = _run(client, reply=RuntimeError("agent exploded"))
    assert resp.status_code == 200 and "Tutor unavailable" in resp.text
    (record,) = turn_log.read()
    assert record["outcome"] == "agent_unavailable"
    assert record["error"] == "RuntimeError" and record["degraded_stage"] == "before_first_byte"

    _run(client, reply=["Isolate x first.\n", "Then"], then_fail=ConnectionError("reset"))
    midstream = turn_log.read()[-1]
    assert midstream["degraded_stage"] == "midstream" and midstream["error"] == "ConnectionError"


def test_a_midstream_failure_whose_flushed_reply_trips_the_leak_check_keeps_the_leak_outcome(
    client, turn_log
):
    # the leak check that runs when the reply ends (it sees the text plus a final newline) fires
    # on the partial reply the failed stream left behind
    guard = "services.gateway.src.gateway.guardrails.stream.detect_answer_leak"
    with patch(guard, side_effect=lambda text: text.endswith("\n")):
        resp = _run(
            client, reply=["Isolate x first. ", "Then"], then_fail=ConnectionError("reset")
        )
    assert "Answer withheld" in resp.text and resp.text.endswith(AGENT_UNAVAILABLE_NOTICE)
    (record,) = turn_log.read()
    assert record["outcome"] == "answer_leak_blocked"
    assert record["error"] == "ConnectionError" and record["degraded_stage"] == "midstream"


def test_a_bug_in_the_guards_is_still_an_error_not_a_notice(client, turn_log):
    # only the agent call is guarded: a failure elsewhere in the turn must stay loud
    with patch(
        "services.gateway.src.gateway.chat.answer.ReplyGuard", side_effect=RuntimeError("guard bug")
    ):
        resp = _run(client)
    assert "Tutor unavailable" not in resp.text
    (record,) = turn_log.read()
    assert record["outcome"] == "error"
