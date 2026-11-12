"""
Tests for the gateway's request size caps and the bounded rate limiter (slice E1).

Behaviors under test:
  1. A message over 4,000 characters, a history over 20 items, a history item over 4,000
     characters, or a candidate_id over 128 characters is rejected with 422 before any guardrail
     or model call; requests exactly at each cap are accepted
  2. The rate limiter keeps at most `max_candidates` ids and evicts the least recently used
     first; an id that keeps sending is not evicted
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from services.gateway.src.gateway.guardrails.rate_limit import RateLimiter
from services.gateway.src.gateway.guardrails.verdict import Verdict
from services.gateway.src.gateway.main import app
from shared.src.shared.wire import (
    MAX_CANDIDATE_ID_CHARS,
    MAX_HISTORY_ITEM_CHARS,
    MAX_HISTORY_ITEMS,
    MAX_MESSAGE_CHARS,
    GenerateRequest,
)
from tests.gateway.gateway_fakes import agent_replies, rate_limiter_of

client = TestClient(app)
BODY = {"candidate_id": "c1", "test_type": "gre", "section": "quant", "message": "Solve 2x+3=11"}
TURN = {"role": "candidate", "content": "hi"}


@pytest.fixture
def pipeline():
    """Everything /chat calls after validation, mocked; `agent` records what was forwarded."""
    sent: list[GenerateRequest] = []

    async def fake_stream(payload, meta=None):
        sent.append(payload)
        yield b"ok"

    with patch(
        "services.gateway.src.gateway.chat.gates.check_content_safety", new_callable=AsyncMock
    ) as safety, patch(
        "services.gateway.src.gateway.chat.gates.check_completeness", new_callable=AsyncMock
    ) as completeness, patch(
        "services.gateway.src.gateway.chat.answer.check_answer_leak", new_callable=AsyncMock
    ) as leak, agent_replies(fake_stream), rate_limiter_of(RateLimiter()):
        safety.return_value = Verdict.allow()
        completeness.return_value = Verdict.allow()
        leak.return_value = Verdict.allow()
        yield sent, safety


# ── 1. Size caps ──────────────────────────────────────────────────────────────

# name -> (body override, the field the 422 must point at)
OVERSIZE = {
    "message": ({"message": "x" * (MAX_MESSAGE_CHARS + 1)}, ["body", "message"]),
    "five-megabyte-message": ({"message": "x" * 5_000_000}, ["body", "message"]),
    "history-items": ({"history": [TURN] * (MAX_HISTORY_ITEMS + 1)}, ["body", "history"]),
    "history-item-length": (
        {"history": [{"role": "agent", "content": "x" * (MAX_HISTORY_ITEM_CHARS + 1)}]},
        ["body", "history", 0, "content"],
    ),
    "candidate-id": (
        {"candidate_id": "c" * (MAX_CANDIDATE_ID_CHARS + 1)},
        ["body", "candidate_id"],
    ),
}


@pytest.mark.parametrize("override,loc", OVERSIZE.values(), ids=OVERSIZE.keys())
def test_oversized_request_is_rejected_before_any_guardrail_or_model_call(
    pipeline, override, loc
):
    sent, safety = pipeline

    resp = client.post("/chat", json={**BODY, **override})

    assert resp.status_code == 422
    assert [e["loc"] for e in resp.json()["detail"]] == [loc]
    safety.assert_not_called()
    assert sent == []


def test_rejection_does_not_echo_the_oversized_input_back():
    resp = client.post("/chat", json={**BODY, "message": "x" * 5_000_000})

    assert resp.status_code == 422
    assert len(resp.content) < 1_000
    assert "xxxx" not in resp.text


def test_requests_exactly_at_the_caps_are_accepted(pipeline):
    sent, _ = pipeline
    body = {
        **BODY,
        "candidate_id": "c" * MAX_CANDIDATE_ID_CHARS,
        "message": "x" * MAX_MESSAGE_CHARS,
        "history": [
            {"role": "candidate", "content": "y" * MAX_HISTORY_ITEM_CHARS}
        ] * MAX_HISTORY_ITEMS,
    }

    resp = client.post("/chat", json=body)

    assert resp.status_code == 200
    assert len(sent) == 1
    assert len(sent[0].history) == MAX_HISTORY_ITEMS


def test_redaction_that_lengthens_text_past_a_cap_is_not_rejected(pipeline):
    sent, safety = pipeline
    emails = "a@b.co " * 571
    assert len(emails) < MAX_MESSAGE_CHARS
    body = {
        **BODY,
        "message": emails,
        "history": [{"role": "candidate", "content": emails}],
    }

    resp = client.post("/chat", json=body)

    assert resp.status_code == 200
    assert resp.text.endswith("ok")
    assert len(sent) == 1
    for redacted in (sent[0].message, sent[0].history[0].content):
        assert "a@b.co" not in redacted
        assert redacted.count("[REDACTED]") == 571
        assert len(redacted) > MAX_HISTORY_ITEM_CHARS
    assert "a@b.co" not in safety.call_args.args[0]


# ── 2. Bounded rate limiter ───────────────────────────────────────────────────


def test_limiter_never_tracks_more_than_max_candidates():
    limiter = RateLimiter(max_candidates=100)
    for i in range(10_000):
        limiter.check(f"rotating-{i}")
    assert len(limiter._states) == 100


def test_least_recently_used_candidate_is_evicted_first():
    limiter = RateLimiter(max_per_minute=2, max_candidates=2)
    limiter.check("a")
    limiter.check("a")  # a is at its limit
    limiter.check("b")
    limiter.check("c")  # over capacity: a is the least recently used and goes
    assert "a" not in limiter._states
    assert set(limiter._states) == {"b", "c"}
    # a came back as a fresh id, so its counters started over
    assert limiter.check("a") == (True, "")


def test_a_candidate_that_keeps_sending_is_not_evicted_and_stays_limited():
    limiter = RateLimiter(max_per_minute=2, max_candidates=3)
    assert limiter.check("busy")[0] is True
    for i in range(20):
        limiter.check(f"noise-{i}")
        limiter.check("busy")  # touched every round: always the most recent but one
    assert "busy" in limiter._states
    allowed, message = limiter.check("busy")
    assert allowed is False
    assert "rate limit" in message.lower()


def test_max_candidates_must_be_positive():
    with pytest.raises(ValueError):
        RateLimiter(max_candidates=0)
