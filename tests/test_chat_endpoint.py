"""
Tests for the Gateway /chat endpoint.

Behaviors under test:
  1. Unsafe content is refused (agent never called)
  2. Unregistered (test_type, section) yields the orchestrator's clarification
  3. A safe, routable turn proxies the agent's streamed tokens back verbatim
  4. Guardrails on the stream (answer leak, PII, rate limit, citations) are applied and
     announced to the Candidate in the stream; all offline with the agent stream mocked
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from services.gateway.src.gateway.guardrails.rate_limit import RateLimiter
from services.gateway.src.gateway.main import app

client = TestClient(app)

_SAFE = (True, "")


def test_chat_refuses_unsafe_content():
    with patch(
        "services.gateway.src.gateway.main.check_content_safety",
        new_callable=AsyncMock,
    ) as mock_safety:
        mock_safety.return_value = (False, "content flagged: violence")
        resp = client.post(
            "/chat",
            json={"candidate_id": "c1", "test_type": "gre", "section": "quant", "message": "x"},
        )
    assert resp.status_code == 200
    assert "can't help" in resp.text.lower()


def test_chat_clarifies_unregistered_section():
    with patch(
        "services.gateway.src.gateway.main.check_content_safety",
        new_callable=AsyncMock,
    ) as mock_safety:
        mock_safety.return_value = _SAFE
        resp = client.post(
            "/chat",
            json={"candidate_id": "c1", "test_type": "gre", "section": "verbal", "message": "hi"},
        )
    assert resp.status_code == 200
    assert "not set up" in resp.text.lower()


def test_chat_proxies_agent_stream_on_route():
    async def fake_stream(payload, meta=None):
        for token in (b"Hello", b", ", b"world"):
            yield token

    with patch(
        "services.gateway.src.gateway.main.check_content_safety",
        new_callable=AsyncMock,
    ) as mock_safety, patch(
        "services.gateway.src.gateway.main._stream_from_agent", fake_stream
    ):
        mock_safety.return_value = _SAFE
        resp = client.post(
            "/chat",
            json={
                "candidate_id": "c1",
                "test_type": "gre",
                "section": "quant",
                "message": "Solve x^2 - 5x + 6 = 0",
            },
        )
    assert resp.status_code == 200
    assert resp.text == "Hello, world"


# ── Guardrails wired into /chat ───────────────────────────────────────────────

_BODY = {"candidate_id": "c1", "test_type": "gre", "section": "quant", "message": "Solve 2x+3=11"}


def _chat(tokens, body=None, num_sources=None, sent=None):
    async def fake_stream(payload, meta=None):
        if sent is not None:
            sent.append(payload)
        if num_sources is not None and meta is not None:
            meta["num_sources"] = num_sources
        for token in tokens:
            yield token.encode()

    with patch(
        "services.gateway.src.gateway.main.check_content_safety", new_callable=AsyncMock
    ) as mock_safety, patch("services.gateway.src.gateway.main._stream_from_agent", fake_stream):
        mock_safety.return_value = _SAFE
        return client.post("/chat", json={**_BODY, **(body or {})})


def test_tutor_reply_that_leaks_the_answer_is_blocked_with_a_notice():
    # The answer line is split across chunks to prove buffering catches it.
    resp = _chat(
        ["Isolate x first.\n", "Subtract 3, then divide.\nThe ans", "wer is 4.\nMore text"]
    )
    assert "Isolate x first." in resp.text
    assert "answer is 4" not in resp.text
    assert "More text" not in resp.text
    assert "[Answer withheld" in resp.text


@pytest.mark.parametrize(
    "leak",
    ["**Answer: 4**", "Final Answer: x = 4", "So the answer is (C).", "So the answer is x = 4."],
)
def test_tutor_leak_variants_are_blocked(leak):
    resp = _chat(["Think about inverse operations.\n", leak])
    assert "[Answer withheld" in resp.text
    assert leak not in resp.text


def test_solve_mode_may_state_the_answer():
    resp = _chat(["Steps...\n", "**Answer: 4**"], body={"mode": "solve"})
    assert "**Answer: 4**" in resp.text
    assert "withheld" not in resp.text


def test_tutor_hint_mentioning_answer_without_value_passes():
    resp = _chat(["The answer is found by isolating x.\n", "Try that."])
    assert resp.text == "The answer is found by isolating x.\nTry that."


def test_output_pii_is_redacted_with_a_notice():
    resp = _chat(["Email me at jo@example.com or call 555-123-4567.\n", "Good luck."])
    assert "jo@example.com" not in resp.text
    assert "555-123-4567" not in resp.text
    assert "[REDACTED]" in resp.text
    assert "[Privacy: removed" in resp.text


def test_input_pii_is_redacted_before_the_agent_and_announced():
    sent: list[dict] = []
    resp = _chat(
        ["Hint."],
        body={
            "message": "I'm jo@example.com, solve 2x+3=11",
            "history": [{"role": "candidate", "content": "call 555-123-4567"}],
        },
        sent=sent,
    )
    assert resp.text.startswith("[Privacy: removed")
    assert "jo@example.com" not in str(sent[0])
    assert "555-123-4567" not in str(sent[0])
    assert resp.text.endswith("Hint.")


def test_rate_limited_candidate_gets_a_clear_message_and_agent_is_not_called():
    sent: list[dict] = []
    limiter = RateLimiter(max_per_minute=2)
    with patch("services.gateway.src.gateway.main._rate_limiter", limiter):
        _chat(["a"], body={"candidate_id": "rl"}, sent=sent)
        _chat(["a"], body={"candidate_id": "rl"}, sent=sent)
        resp = _chat(["a"], body={"candidate_id": "rl"}, sent=sent)
        other = _chat(["a"], body={"candidate_id": "someone-else"}, sent=sent)
    assert len(sent) == 3
    assert "rate limit exceeded" in resp.text
    assert other.text == "a"


def test_reply_citing_unretrieved_sources_has_those_sentences_removed_and_announced():
    resp = _chat(
        ["Planning matters [1]. Planning also wins [7]. Plan well.\n", "Next line [2].\n"],
        body={"section": "aw"},
        num_sources=2,
    )
    assert "[7]" not in resp.text
    assert "Planning matters [1]." in resp.text
    assert "Plan well." in resp.text
    assert "Next line [2]." in resp.text
    assert "[Citation check: removed 1 sentence(s)" in resp.text


def test_citations_are_not_checked_when_agent_reports_no_source_count():
    resp = _chat(["Try isolating x first [1]."])
    assert resp.text == "Try isolating x first [1]."
