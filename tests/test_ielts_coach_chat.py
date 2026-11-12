"""
/chat for ("ielts", "writing"): the gateway finds the learner's graded report in the attempt
store, hands the agent only the digest, and keeps its gates (moderation still runs). No model
call is made; the agent is a fake.
"""
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from scripts.seed_ielts_attempt import ESSAY, seed
from services.gateway.src.gateway import ielts_api
from services.gateway.src.gateway.guardrails.stream import LLM_LEAK_NOTICE
from services.gateway.src.gateway.guardrails.verdict import Outcome, Verdict
from services.gateway.src.gateway.ielts_coach import (
    NO_REPORT_NOTICE,
    SOLVE_MODE_NOTICE,
    UNKNOWN_ATTEMPT_NOTICE,
)
from services.gateway.src.gateway.ielts_store import AttemptStore
from services.gateway.src.gateway.main import app
from shared.src.shared.wire import GenerateRequest
from tests.gateway_fakes import agent_replies

client = TestClient(app)
GATEWAY = "services.gateway.src.gateway.main"

_BODY = {
    "candidate_id": "c1", "test_type": "ielts", "section": "writing",
    "message": "why is my Coherence 6.0 and what do I fix first?",
}


@pytest.fixture
def store(tmp_path: Path):
    store = AttemptStore(tmp_path / "attempts.sqlite3")
    app.dependency_overrides[ielts_api.get_store] = lambda: store
    yield store
    app.dependency_overrides.pop(ielts_api.get_store, None)


@pytest.fixture(autouse=True)
def gates():
    """Moderation and completeness are mocked (no model call); the moderation mock is returned
    so a test can assert the gate still ran."""
    with (
        patch(f"{GATEWAY}.check_content_safety", new_callable=AsyncMock) as safety,
        patch(f"{GATEWAY}.check_completeness", new_callable=AsyncMock) as completeness,
        patch(
            "services.gateway.src.gateway.guardrails.output._call_leak_checker",
            new_callable=AsyncMock, return_value=False,
        ),
    ):
        safety.return_value = Verdict.allow()
        completeness.return_value = Verdict.allow()
        yield safety


def chat(body: dict | None = None):
    seen: list[GenerateRequest] = []

    async def agent(payload, meta=None):
        seen.append(payload)
        yield b"Your linking is mechanical."

    with agent_replies(agent):
        resp = client.post("/chat", json={**_BODY, **(body or {})})
    return resp, seen


def test_the_latest_graded_report_reaches_the_agent_as_a_digest_without_the_essay(store, gates):
    seed(store)
    resp, seen = chat()
    assert resp.text == "Your linking is mechanical."
    assert len(seen) == 1
    digest = seen[0].report_digest
    assert "<<<REPORT" in digest and "CC (Coherence and Cohesion): band 5.5" in digest
    assert not any(paragraph in digest for paragraph in ESSAY.split("\n\n"))
    gates.assert_awaited_once()  # OpenAI moderation (the existing gate) still runs on IELTS chat


def test_a_named_attempt_is_coached_over_the_latest_one(store):
    first = seed(store)
    seed(store)
    _, seen = chat({"attempt_id": first.id})
    assert seen and seen[0].report_digest


def test_no_graded_attempt_is_answered_without_calling_the_agent_or_a_model(store, gates):
    resp, seen = chat()
    assert resp.text == NO_REPORT_NOTICE
    assert seen == []
    gates.assert_not_awaited()


def test_an_unknown_attempt_is_answered_with_a_notice(store):
    seed(store)
    resp, seen = chat({"attempt_id": "nope"})
    assert resp.text == UNKNOWN_ATTEMPT_NOTICE and seen == []


def test_an_ungraded_attempt_is_not_coached(store):
    attempt = store.create("p", "prompt")
    resp, seen = chat({"attempt_id": attempt.id})
    assert resp.text == UNKNOWN_ATTEMPT_NOTICE and seen == []


def test_solve_mode_is_answered_with_a_clarification(store):
    seed(store)
    resp, seen = chat({"mode": "solve"})
    assert resp.text == SOLVE_MODE_NOTICE and seen == []


def test_a_gre_turn_sends_no_digest_and_never_opens_the_store(store):
    seed(store)
    resp, seen = chat({"test_type": "gre", "section": "quant", "message": "Solve x+1=2"})
    assert resp.status_code == 200
    assert seen[0].report_digest == ""


def test_a_coaching_reply_passes_the_tutor_answer_leak_guard(store):
    seed(store)
    reply = (
        'Your quote "Moreover, they have no fear of making mistakes" opens with a linker. '
        "Try this ten-minute drill: rewrite it as: They have no fear of making mistakes, "
        "so they speak freely. Which link did you drop?"
    )

    async def agent(payload, meta=None):
        yield reply.encode()

    with agent_replies(agent):
        resp = client.post("/chat", json=_BODY)
    assert resp.text == reply


_GRE = {"test_type": "gre", "section": "quant", "message": "Solve x+1=2"}
_LEAK = "services.gateway.src.gateway.guardrails.output._call_leak_checker"


def test_a_report_question_reaches_the_coach_when_the_completeness_check_would_call_it_incomplete(
    store,
):
    seed(store)
    with patch(f"{GATEWAY}.check_completeness", new_callable=AsyncMock) as completeness:
        completeness.return_value = Verdict.stop(
            Outcome.clarification_requested, "Could you share the full question?"
        )
        resp, seen = chat()
    assert resp.text == "Your linking is mechanical."
    assert len(seen) == 1
    completeness.assert_not_awaited()


def test_a_gre_turn_still_runs_the_completeness_check():
    with patch(f"{GATEWAY}.check_completeness", new_callable=AsyncMock) as completeness:
        completeness.return_value = Verdict.stop(
            Outcome.clarification_requested, "Could you share the full question?"
        )
        resp, seen = chat(_GRE)
    assert resp.text == "Could you share the full question?"
    assert seen == []
    completeness.assert_awaited_once()


def test_a_coaching_reply_is_not_checked_for_a_gre_final_answer(store):
    seed(store)
    reply = "Answer: they have no fear of making mistakes, so they speak freely."

    async def agent(payload, meta=None):
        yield reply.encode()

    with agent_replies(agent), patch(_LEAK, new_callable=AsyncMock, return_value=True) as llm:
        resp = client.post("/chat", json=_BODY)
    assert resp.text == reply
    llm.assert_not_awaited()


def test_a_gre_reply_still_runs_both_answer_leak_checks():
    async def stating(payload, meta=None):
        yield b"Answer: 1"

    async def hinting(payload, meta=None):
        yield b"Think about isolating x."

    with agent_replies(stating):
        regex = client.post("/chat", json={**_BODY, **_GRE})
    assert "Answer: 1" not in regex.text

    with agent_replies(hinting), patch(_LEAK, new_callable=AsyncMock, return_value=True) as llm:
        resp = client.post("/chat", json={**_BODY, **_GRE})
    assert LLM_LEAK_NOTICE in resp.text
    llm.assert_awaited_once()
