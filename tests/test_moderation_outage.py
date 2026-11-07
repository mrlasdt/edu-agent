"""A moderation outage follows an explicit policy instead of failing the turn (ADR-0008)."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from services.gateway.src.gateway.guardrails.input import (
    ModerationUnavailableError,
    check_content_safety,
)
from services.gateway.src.gateway.guardrails.verdict import Verdict
from services.gateway.src.gateway.main import app
from services.gateway.src.gateway.settings import GatewaySettings
from shared.src.shared.turn_meter import TurnMeter
from tests.gateway_fakes import agent_replies

_INPUT = "services.gateway.src.gateway.guardrails.input"
_MAIN = "services.gateway.src.gateway.main"
_BODY = {"candidate_id": "c1", "test_type": "gre", "section": "quant", "message": "Solve x + 1 = 2"}
_OUTAGE = httpx.ConnectError("All connection attempts failed")


def _settings(**kw: Any) -> GatewaySettings:
    return GatewaySettings(openai_api_key="k", **kw)


# ── check_content_safety: the policy ─────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("env", ["prod", "staging"])
async def test_outage_fails_closed_outside_dev_by_default(env):
    with patch(f"{_INPUT}.get_settings", return_value=_settings(env=env)), \
         patch(f"{_INPUT}._call_moderation_api", new_callable=AsyncMock, side_effect=_OUTAGE):
        with pytest.raises(ModerationUnavailableError):
            await check_content_safety("Solve x + 1 = 2", "t")


@pytest.mark.asyncio
async def test_outage_fails_open_in_dev_by_default_and_is_recorded():
    meter = TurnMeter("t", "gateway", "gre/quant", "tutor")
    with patch(f"{_INPUT}.get_settings", return_value=_settings(env="dev")), \
         patch(f"{_INPUT}._call_moderation_api", new_callable=AsyncMock, side_effect=_OUTAGE):
        assert await check_content_safety("Solve x + 1 = 2", "t", meter) == Verdict.allow()
    assert meter.snapshot()["moderation_failed_open"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("env", "policy", "fails_open"),
    [("prod", "open", True), ("dev", "closed", False)],
)
async def test_explicit_policy_overrides_the_env_default(env, policy, fails_open):
    settings = _settings(env=env, moderation_on_error=policy)
    with patch(f"{_INPUT}.get_settings", return_value=settings), \
         patch(f"{_INPUT}._call_moderation_api", new_callable=AsyncMock, side_effect=_OUTAGE):
        if fails_open:
            assert await check_content_safety("hi", "t") == Verdict.allow()
        else:
            with pytest.raises(ModerationUnavailableError):
                await check_content_safety("hi", "t")


# ── /chat: the learner sees a message, not a broken stream ───────────────────

@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app, raise_server_exceptions=False)


def _chat(client: TestClient, settings: GatewaySettings) -> Any:
    async def agent_stream(payload: dict[str, Any], meta: dict[str, Any] | None = None) -> Any:
        yield b"Isolate x first."

    with (
        patch(f"{_INPUT}.get_settings", return_value=settings),
        patch(f"{_INPUT}._call_moderation_api", new_callable=AsyncMock, side_effect=_OUTAGE),
        patch(f"{_MAIN}.check_completeness", new_callable=AsyncMock, return_value=Verdict.allow()),
        agent_replies(agent_stream),
        patch("services.gateway.src.gateway.guardrails.output._call_leak_checker",
              new_callable=AsyncMock, return_value=False),
    ):
        return client.post("/chat", json=_BODY)


def test_chat_answers_safety_check_unavailable_when_closed(client, turn_log):
    response = _chat(client, _settings(env="prod"))
    assert response.status_code == 200
    assert "safety check" in response.text.lower()
    assert "Isolate x first." not in response.text
    (record,) = turn_log.read()
    assert record["outcome"] == "moderation_unavailable"


def test_chat_continues_when_the_policy_is_open(client, turn_log):
    response = _chat(client, _settings(env="dev"))
    assert response.status_code == 200
    assert "Isolate x first." in response.text
    (record,) = turn_log.read()
    assert record["outcome"] == "ok"
