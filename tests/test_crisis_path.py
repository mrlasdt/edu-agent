"""A self-harm moderation hit gets a fixed supportive reply and no model call (ADR-0008)."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.gateway.src.gateway.guardrails.input import check_content_safety
from services.gateway.src.gateway.guardrails.supportive import (
    SUPPORTIVE_MESSAGE,
    is_self_harm_reason,
)
from services.gateway.src.gateway.guardrails.verdict import Verdict
from services.gateway.src.gateway.main import app
from services.gateway.src.gateway.settings import GatewaySettings
from shared.src.shared.turn_outcome import Outcome
from tests.gateway_fakes import agent_replies

_INPUT = "services.gateway.src.gateway.guardrails.input"
_MAIN = "services.gateway.src.gateway.main"
_TRIGGER = "__test_self_harm__"
_BODY = {"candidate_id": "c1", "test_type": "gre", "section": "quant", "message": "hello"}


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app, raise_server_exceptions=False)


def _agent() -> MagicMock:
    """A stand-in for the agent stream that answers once and records its calls."""

    async def stream(payload: dict[str, Any], meta: dict[str, Any] | None = None) -> Any:
        yield b"Isolate x first."

    return MagicMock(side_effect=stream)


def _post(
    client: TestClient,
    moderation: dict[str, Any],
    settings: GatewaySettings | None = None,
    message: str = "hello",
) -> Any:
    """POST /chat; returns (response, completeness mock, agent mock)."""
    completeness = AsyncMock(return_value=Verdict.allow())
    agent = _agent()
    settings = settings or GatewaySettings(openai_api_key="k", env="prod")
    with (
        patch(f"{_INPUT}.get_settings", return_value=settings),
        patch(f"{_INPUT}._call_moderation_api", new_callable=AsyncMock, return_value=moderation),
        patch(f"{_MAIN}.check_completeness", completeness),
        agent_replies(agent),
        patch(
            "services.gateway.src.gateway.guardrails.output._call_leak_checker",
            new_callable=AsyncMock, return_value=False,
        ),
    ):
        return client.post("/chat", json={**_BODY, "message": message}), completeness, agent


@pytest.mark.parametrize("category", ["self-harm", "self-harm/intent", "self-harm/instructions"])
def test_self_harm_hit_gets_the_supportive_reply_and_no_model_call(client, turn_log, category):
    response, completeness, agent = _post(
        client, {"flagged": True, "categories": {category: True, "violence": False}}
    )
    assert response.status_code == 200
    assert response.text == SUPPORTIVE_MESSAGE
    assert "can't help" not in response.text
    completeness.assert_not_called()
    agent.assert_not_called()
    (record,) = turn_log.read()
    assert record["outcome"] == "supportive_reply"
    assert record["llm_calls"] == 0


def test_self_harm_reply_with_pii_in_the_message_is_only_the_template(client, turn_log):
    response, _, agent = _post(
        client,
        {"flagged": True, "categories": {"self-harm/intent": True}},
        message="call me at 555-123-4567, I want to end it",
    )
    assert response.text == SUPPORTIVE_MESSAGE
    agent.assert_not_called()


def test_pii_notice_still_precedes_a_normal_answer(client, turn_log):
    response, _, _ = _post(
        client, {"flagged": False, "categories": {}}, message="call me at 555-123-4567"
    )
    assert response.text.startswith("[Privacy:")
    assert response.text.endswith("Isolate x first.")


def test_the_template_names_emergency_services_and_988():
    assert "local emergency services" in SUPPORTIVE_MESSAGE
    assert "988" in SUPPORTIVE_MESSAGE
    assert "come back" in SUPPORTIVE_MESSAGE
    assert "\u2014" not in SUPPORTIVE_MESSAGE


def test_other_moderation_categories_keep_the_existing_reply(client, turn_log):
    response, _, _ = _post(
        client, {"flagged": True, "categories": {"violence": True, "self-harm": False}}
    )
    assert response.text == "I can't help with that request (content flagged: violence)."
    (record,) = turn_log.read()
    assert record["outcome"] == "content_blocked"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("content flagged: self-harm", True),
        ("content flagged: violence, self-harm/intent", True),
        ("content flagged: self_harm/instructions", True),
        ("content flagged: violence", False),
        ("content flagged: harassment/threatening", False),
        ("", False),
    ],
)
def test_is_self_harm_reason(reason, expected):
    assert is_self_harm_reason(reason) is expected


# ── the dev-only trigger ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trigger_reports_a_self_harm_hit_in_dev_without_a_moderation_call():
    settings = GatewaySettings(env="dev", safety_test_trigger=_TRIGGER)
    with patch(f"{_INPUT}.get_settings", return_value=settings), \
         patch(f"{_INPUT}._call_moderation_api", new_callable=AsyncMock) as moderation:
        verdict = await check_content_safety(f"help {_TRIGGER.upper()}", "t")
    assert verdict == Verdict.stop(Outcome.supportive_reply, SUPPORTIVE_MESSAGE)
    moderation.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("env", ["prod", "staging"])
async def test_trigger_is_inert_outside_dev(env):
    settings = GatewaySettings(env=env, safety_test_trigger=_TRIGGER, openai_api_key="k")
    clean = {"flagged": False, "categories": {}}
    with patch(f"{_INPUT}.get_settings", return_value=settings), \
         patch(f"{_INPUT}._call_moderation_api", new_callable=AsyncMock, return_value=clean) as mod:
        assert await check_content_safety(f"help {_TRIGGER}", "t") == Verdict.allow()
    mod.assert_awaited_once()


@pytest.mark.asyncio
async def test_trigger_is_off_when_unset():
    settings = GatewaySettings(env="dev")
    with patch(f"{_INPUT}.get_settings", return_value=settings):
        assert await check_content_safety("help __test_self_harm__", "t") == Verdict.allow()


def test_chat_with_the_trigger_in_dev_gets_the_supportive_reply(client, turn_log):
    settings = GatewaySettings(env="dev", safety_test_trigger=_TRIGGER)
    completeness, agent = AsyncMock(), _agent()
    with (
        patch(f"{_INPUT}.get_settings", return_value=settings),
        patch(f"{_MAIN}.check_completeness", completeness),
        agent_replies(agent),
    ):
        response = client.post("/chat", json={**_BODY, "message": f"{_TRIGGER} please"})
    assert response.text == SUPPORTIVE_MESSAGE
    completeness.assert_not_called()
    agent.assert_not_called()
    (record,) = turn_log.read()
    assert record["outcome"] == "supportive_reply"
    assert record["llm_calls"] == 0


def test_chat_with_the_trigger_outside_dev_is_not_a_supportive_reply(client, turn_log):
    response, _, _ = _post(
        client,
        {"flagged": False, "categories": {}},
        GatewaySettings(env="prod", safety_test_trigger=_TRIGGER, openai_api_key="k"),
        message=f"{_TRIGGER} please",
    )
    assert response.text == "Isolate x first."
