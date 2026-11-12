"""
The guardrails' three outbound calls, run for real against a fake provider.

The policy tests in `test_guardrails.py`, `test_crisis_path.py` and `test_moderation_outage.py`
patch the call seams (`_call_moderation_api`, `_call_leak_checker`, `_call_enrichment_llm`) with
`AsyncMock`, so the request each seam builds and the way it reads the reply run in no test. A typo
in the moderation URL (a 404 in production, so every turn refused) or `flagged` read inverted
(crisis messages pass, normal ones blocked) passed every one of them. These tests leave the seams
alone and fake the provider instead:

  - moderation is a plain HTTP call, so `respx` answers it at the HTTP layer and the tests assert
    the URL, the `Authorization` header, the JSON body and the mapping of the reply;
  - the leak checker and the completeness check go through the litellm Router, not plain HTTP, so
    they use the repo's scripted-router pattern (`tests/llm_fakes.py`): a real Router whose
    deployment answers with litellm's `mock_response`, or a `ScriptedRouter` that captures the
    request.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import litellm
import pytest
import respx

from services.gateway.src.gateway.guardrails.input import (
    ModerationUnavailableError,
    _call_moderation_api,
    check_content_safety,
)
from services.gateway.src.gateway.guardrails.output import check_answer_leak
from services.gateway.src.gateway.guardrails.supportive import SUPPORTIVE_MESSAGE
from services.gateway.src.gateway.guardrails.verdict import Verdict
from services.gateway.src.gateway.orchestrator.enrich import (
    DEFAULT_CLARIFICATION,
    CompletenessVerdict,
    check_completeness,
)
from services.gateway.src.gateway.settings import GatewaySettings
from shared.src.shared.llm_router import use_router
from shared.src.shared.models import Message, MessageRole, Mode, Session
from shared.src.shared.structured import StructuredOutputError
from shared.src.shared.turn_outcome import Outcome
from tests.llm_fakes import SERVER_ERROR, ScriptedRouter, calls, mock_router

_INPUT = "services.gateway.src.gateway.guardrails.input"
_OUTPUT = "services.gateway.src.gateway.guardrails.output"
_MODERATIONS_URL = "https://api.openai.com/v1/moderations"
_MODEL = "openai/gpt-4o-mini"  # the orchestrator role in config/model_config.dev.yaml


# ── moderation: an HTTP call, faked with respx ───────────────────────────────

def _moderation_reply(flagged: bool, **categories: bool) -> httpx.Response:
    """OpenAI's reply shape, with the extra fields the real API sends that we ignore."""
    return httpx.Response(200, json={
        "id": "modr-1",
        "model": "omni-moderation-latest",
        "results": [{
            "flagged": flagged,
            "categories": categories,
            "category_scores": {name: 0.9 if hit else 0.01 for name, hit in categories.items()},
        }],
    })


@pytest.fixture
def moderation_settings():
    """A prod-like setup with a key; the settings the seam reads are the ones patched here."""
    settings = GatewaySettings(openai_api_key="sk-test-123", env="prod")
    with patch(f"{_INPUT}.get_settings", return_value=settings):
        yield settings


@pytest.mark.asyncio
@respx.mock
async def test_moderation_request_has_the_right_url_header_and_body(moderation_settings):
    route = respx.post(_MODERATIONS_URL).mock(
        return_value=_moderation_reply(False, violence=False)
    )
    await _call_moderation_api("Solve x + 1 = 2")
    assert route.call_count == 1
    request = route.calls.last.request
    assert request.method == "POST"
    assert str(request.url) == _MODERATIONS_URL
    assert request.headers["Authorization"] == "Bearer sk-test-123"
    assert json.loads(request.content) == {"input": "Solve x + 1 = 2"}


@pytest.mark.asyncio
@respx.mock
async def test_moderation_reply_is_mapped_to_flagged_and_categories(moderation_settings):
    respx.post(_MODERATIONS_URL).mock(
        return_value=_moderation_reply(True, violence=False, **{"self-harm": True})
    )
    assert await _call_moderation_api("anything") == {
        "flagged": True,
        "categories": {"violence": False, "self-harm": True},
    }


@pytest.mark.asyncio
@respx.mock
async def test_flagged_message_is_blocked_with_its_categories_as_the_reason(moderation_settings):
    respx.post(_MODERATIONS_URL).mock(
        return_value=_moderation_reply(
            True, violence=False, **{"self-harm": True, "self-harm/intent": True}
        )
    )
    verdict = await check_content_safety("I want to hurt myself", "t-flagged")
    assert verdict == Verdict.stop(Outcome.supportive_reply, SUPPORTIVE_MESSAGE)


@pytest.mark.asyncio
@respx.mock
async def test_clean_message_passes(moderation_settings):
    route = respx.post(_MODERATIONS_URL).mock(
        return_value=_moderation_reply(False, violence=False, **{"self-harm": False})
    )
    assert await check_content_safety("Solve x + 1 = 2", "t-clean") == Verdict.allow()
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("status", [401, 429, 500, 503])
async def test_a_bad_status_raises_instead_of_reading_the_error_body(moderation_settings, status):
    respx.post(_MODERATIONS_URL).mock(
        return_value=httpx.Response(status, json={"error": {"message": "nope"}})
    )
    with pytest.raises(httpx.HTTPStatusError) as caught:
        await _call_moderation_api("anything")
    assert caught.value.response.status_code == status


@pytest.mark.asyncio
@respx.mock
async def test_a_bad_status_fails_closed_outside_dev(moderation_settings):
    respx.post(_MODERATIONS_URL).mock(return_value=httpx.Response(404))
    with pytest.raises(ModerationUnavailableError):
        await check_content_safety("Solve x + 1 = 2", "t-404")


@pytest.mark.asyncio
@respx.mock
async def test_a_transport_error_fails_closed_outside_dev(moderation_settings):
    respx.post(_MODERATIONS_URL).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ModerationUnavailableError):
        await check_content_safety("Solve x + 1 = 2", "t-down")


@pytest.mark.asyncio
@respx.mock
async def test_a_reply_without_results_fails_closed_outside_dev(moderation_settings):
    respx.post(_MODERATIONS_URL).mock(return_value=httpx.Response(200, json={"results": []}))
    with pytest.raises(ModerationUnavailableError):
        await check_content_safety("Solve x + 1 = 2", "t-empty")


@pytest.mark.asyncio
@respx.mock
async def test_a_bad_status_fails_open_when_the_policy_says_open():
    settings = GatewaySettings(openai_api_key="k", env="prod", moderation_on_error="open")
    respx.post(_MODERATIONS_URL).mock(return_value=httpx.Response(500))
    with patch(f"{_INPUT}.get_settings", return_value=settings):
        assert await check_content_safety("Solve x + 1 = 2", "t-open") == Verdict.allow()


def _reply(content: str) -> Any:
    return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": content}}])


# ── leak checker: a Router call, faked with a mock_response ──────────────────

@pytest.fixture
def leak_settings():
    """Both provider keys set, so the LLM stage is not skipped for a missing key."""
    with patch(f"{_OUTPUT}.get_settings",
               return_value=GatewaySettings(openai_api_key="k", anthropic_api_key="k")):
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verdict", "leaked"),
    [("LEAK", True), ("leak", True), (" Leak: it states x = 2", True), ("SAFE", False),
     ("safe", False)],
)
async def test_leak_verdict_is_read_from_the_model_reply(leak_settings, verdict, leaked):
    router = mock_router({_MODEL: verdict})
    with use_router(router):
        got = await check_answer_leak(
            "Try isolating the variable.", Mode.tutor, "t-verdict"
        )
    assert got.allowed is not leaked
    assert calls(router) == {_MODEL: 1}  # the orchestrator model answered, once


@pytest.mark.asyncio
async def test_leak_check_sends_the_reply_under_review_with_the_leak_instruction(leak_settings):
    seen: list[dict[str, Any]] = []

    async def behave(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return _reply("SAFE")

    text = "Try isolating the variable."
    with use_router(ScriptedRouter(behave)):
        assert (await check_answer_leak(text, Mode.tutor, "t-request")).allowed
    (request,) = seen
    assert request["model"] == _MODEL
    assert request["temperature"] == 0
    assert request["stream"] is False
    roles = [m["role"] for m in request["messages"]]
    assert roles == ["system", "user"]
    assert "LEAK or SAFE" in request["messages"][0]["content"]
    assert request["messages"][1]["content"] == text


@pytest.mark.asyncio
async def test_a_provider_error_in_the_leak_check_is_not_a_leak(leak_settings):
    router = mock_router({_MODEL: SERVER_ERROR})
    with use_router(router):
        assert (await check_answer_leak("Try isolating the variable.", Mode.tutor, "t-err")).allowed


# ── completeness check: a Router call, faked with a mock_response ────────────

def _session() -> Session:
    return Session(candidate_id="c1", test_type="gre", section="quant", mode=Mode.tutor)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            '{"complete": false, "clarification": "Which problem?"}',
            Verdict.stop(Outcome.clarification_requested, "Which problem?"),
        ),
        ('{"complete": true, "clarification": ""}', Verdict.allow()),
        (
            '{"complete": false}',
            Verdict.stop(Outcome.clarification_requested, DEFAULT_CLARIFICATION),
        ),
    ],
)
async def test_completeness_reply_is_parsed_from_the_model_reply(reply, expected):
    router = mock_router({_MODEL: reply})
    with use_router(router):
        assert await check_completeness("solve this", _session(), "t-complete") == expected
    assert calls(router) == {_MODEL: 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",  # unparseable
        '{"clarification": "Which problem?"}',  # no verdict: never read as complete
        '{"complete": "false", "clarification": "Which problem?"}',  # a string is not a boolean
    ],
)
async def test_a_completeness_reply_that_is_not_a_verdict_is_an_error_not_a_pass(reply):
    router = mock_router({_MODEL: reply})
    with use_router(router), pytest.raises(StructuredOutputError):
        await check_completeness("solve this", _session(), "t-bad-reply")
    assert calls(router) == {_MODEL: 1}


@pytest.mark.asyncio
async def test_completeness_check_sends_the_history_and_asks_for_the_verdict_schema():
    seen: list[dict[str, Any]] = []

    async def behave(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return _reply('{"complete": true, "clarification": ""}')

    session = _session()
    session.history = [
        Message(role=MessageRole.candidate, content="Solve 2x + 3 = 11."),
        Message(role=MessageRole.agent, content="Subtract 3 from both sides: 2x = 8."),
    ]
    with use_router(ScriptedRouter(behave)):
        await check_completeness("why did we subtract 3?", session, "t-hist")
    (request,) = seen
    assert request["model"] == _MODEL
    assert request["temperature"] == 0
    assert request["stream"] is False
    assert request["response_format"] is CompletenessVerdict
    assert [m["role"] for m in request["messages"]] == ["system", "user", "assistant", "user"]
    assert "GRE QUANT" in request["messages"][0]["content"]
    assert request["messages"][-1]["content"] == "why did we subtract 3?"
