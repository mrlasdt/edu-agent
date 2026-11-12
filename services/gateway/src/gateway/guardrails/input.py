"""Input guardrails: the content safety check."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from services.gateway.src.gateway.guardrails.events import emit_guardrail_event
from services.gateway.src.gateway.guardrails.supportive import (
    SUPPORTIVE_MESSAGE,
    flagged_reason,
    is_self_harm_reason,
)
from services.gateway.src.gateway.guardrails.verdict import Verdict
from services.gateway.src.gateway.settings import get_settings
from shared.src.shared.turn_meter import TurnMeter
from shared.src.shared.turn_outcome import Outcome

_log = logging.getLogger(__name__)


class ModerationUnavailableError(RuntimeError):
    """The moderation call failed and the fail policy is "closed": the turn must not proceed."""


def _blocked(reason: str) -> Verdict:
    """The verdict for a moderation hit: the supportive text for self-harm, a refusal otherwise."""
    if is_self_harm_reason(reason):
        return Verdict.stop(Outcome.supportive_reply, SUPPORTIVE_MESSAGE)
    return Verdict.stop(Outcome.content_blocked, f"I can't help with that request ({reason}).")


def _moderation_fails_open() -> bool:
    """MODERATION_ON_ERROR, defaulting to closed everywhere except ENV=dev."""
    settings = get_settings()
    if settings.moderation_on_error:
        return settings.moderation_on_error == "open"
    return settings.env.lower() == "dev"


async def check_content_safety(
    text: str, trace_id: str, meter: TurnMeter | None = None
) -> Verdict:
    """
    Call the OpenAI Moderation API.
    Returns `Verdict.allow()` when safe. A hit stops the turn: `supportive_reply` with the fixed
    supportive text for a self-harm category, `content_blocked` with a refusal for any other.

    In ENV=dev with no OpenAI key (a free-model setup, see config/model_config.free.yaml) the
    call is skipped, the turn goes on, and `moderation_skipped` is recorded in the turn record
    when a `meter` is given. In any other ENV a missing key is not special-cased: the call fails.

    If the call fails (outage, timeout, bad status), MODERATION_ON_ERROR decides: "open" lets the
    turn go on and records `moderation_failed_open`; "closed" raises ModerationUnavailableError. The
    default is closed, except open in ENV=dev. Either way the failure is logged as an event.

    In ENV=dev, a non-empty SAFETY_TEST_TRIGGER found in the text reports a self-harm hit with no
    moderation call, so the supportive path can be tried without a paid key. Outside dev it is
    ignored.
    """
    settings = get_settings()
    trigger = settings.safety_test_trigger.strip().lower()
    if settings.env.lower() == "dev" and trigger and trigger in text.lower():
        reason = flagged_reason(["self-harm"])
        emit_guardrail_event(
            "input", "content_safety", "fail", trace_id, reason=reason, source="test_trigger"
        )
        return _blocked(reason)
    if settings.env.lower() == "dev" and not settings.openai_api_key:
        emit_guardrail_event(
            "input", "content_safety", "skipped", trace_id, reason="no_openai_key_in_dev"
        )
        if meter is not None:
            meter.annotate(moderation_skipped=True)
        return Verdict.allow()
    try:
        result = await _call_moderation_api(text)
    except Exception as exc:
        fail_open = _moderation_fails_open()
        _log.warning("moderation call failed (%s); failing %s", type(exc).__name__,
                     "open" if fail_open else "closed", exc_info=True)
        emit_guardrail_event(
            "input", "content_safety", "error", trace_id,
            reason=type(exc).__name__, policy="open" if fail_open else "closed",
        )
        if not fail_open:
            raise ModerationUnavailableError("content-safety check unavailable") from exc
        if meter is not None:
            meter.annotate(moderation_failed_open=True)
        return Verdict.allow()
    if result.get("flagged"):
        flagged_cats = [k for k, v in result.get("categories", {}).items() if v]
        reason = flagged_reason(flagged_cats)
        emit_guardrail_event("input", "content_safety", "fail", trace_id, reason=reason)
        return _blocked(reason)
    emit_guardrail_event("input", "content_safety", "pass", trace_id)
    return Verdict.allow()


async def _call_moderation_api(text: str) -> dict[str, Any]:
    """Call OpenAI Moderation API. Seam for unit test patching."""
    settings = get_settings()
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/moderations",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={"input": text},
        )
        response.raise_for_status()
        data = response.json()
        result = data["results"][0]
        return {"flagged": result["flagged"], "categories": result["categories"]}
