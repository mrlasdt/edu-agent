"""Input guardrails: content safety check and completeness enrichment."""

from __future__ import annotations

import logging
from typing import Any, cast

import httpx

from services.gateway.src.gateway.guardrails.events import emit_guardrail_event
from shared.src.shared.config import get_settings, load_model_config
from shared.src.shared.models import MessageRole, Session
from shared.src.shared.observability import build_llm_metadata
from shared.src.shared.turn_meter import TurnMeter

_log = logging.getLogger(__name__)

_HISTORY_TURNS = 6

# The completeness check answers with one short JSON object (a boolean and, when incomplete, one
# clarifying question). A reasoning model also spends hidden thinking tokens against this cap, so
# it is sized well above the visible answer: low enough to stop a runaway reply from delaying
# every turn before the tutor starts, high enough that a reasoning model still finishes. If the
# reply is cut off it does not parse and the check fails open (the question is treated as
# complete).
COMPLETENESS_MAX_TOKENS = 512


class ModerationUnavailableError(RuntimeError):
    """The moderation call failed and the fail policy is "closed": the turn must not proceed."""


def _moderation_fails_open() -> bool:
    """MODERATION_ON_ERROR, defaulting to closed everywhere except ENV=dev."""
    settings = get_settings()
    if settings.moderation_on_error:
        return settings.moderation_on_error == "open"
    return settings.env.lower() == "dev"


async def check_content_safety(
    text: str, trace_id: str, meter: TurnMeter | None = None
) -> tuple[bool, str]:
    """
    Call the OpenAI Moderation API.
    Returns (safe, reason) — reason is empty when safe=True.

    In ENV=dev with no OpenAI key (a free-model setup, see config/model_config.free.yaml) the
    call is skipped, the turn goes on, and `moderation_skipped` is recorded in the turn record
    when a `meter` is given. In any other ENV a missing key is not special-cased: the call fails.

    If the call fails (outage, timeout, bad status), MODERATION_ON_ERROR decides: "open" lets the
    turn go on and records `moderation_failed_open`; "closed" raises ModerationUnavailableError. The
    default is closed, except open in ENV=dev. Either way the failure is logged as an event.
    """
    settings = get_settings()
    if settings.env.lower() == "dev" and not settings.openai_api_key:
        emit_guardrail_event(
            "input", "content_safety", "skipped", trace_id, reason="no_openai_key_in_dev"
        )
        if meter is not None:
            meter.annotate(moderation_skipped=True)
        return True, ""
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
        return True, ""
    if result.get("flagged"):
        flagged_cats = [k for k, v in result.get("categories", {}).items() if v]
        reason = f"content flagged: {', '.join(flagged_cats)}"
        emit_guardrail_event("input", "content_safety", "fail", trace_id, reason=reason)
        return False, reason
    emit_guardrail_event("input", "content_safety", "pass", trace_id)
    return True, ""


async def check_completeness(
    text: str, session: Session, trace_id: str, meter: TurnMeter | None = None
) -> tuple[bool, str]:
    """
    Haiku-tier LLM check: is the question complete enough to answer?
    Returns (complete, clarification_prompt). With a `meter`, its model call is measured.
    """
    result = await _call_enrichment_llm(text, session, trace_id, meter)
    complete = result.get("complete", True)
    clarification = result.get("clarification", "")
    label = "pass" if complete else "fail"
    emit_guardrail_event("input", "completeness", label, trace_id)
    return complete, clarification


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


async def _call_enrichment_llm(
    text: str, session: Session, trace_id: str, meter: TurnMeter | None = None
) -> dict[str, Any]:
    """
    Haiku-tier LLM call to detect incomplete questions.
    Returns {"complete": bool, "clarification": str}.
    Seam for unit test patching.
    """
    cfg = load_model_config()
    model = cfg.litellm_model("orchestrator")
    system = (
        f"You are checking if the latest candidate message in a GRE {session.section.upper()} "
        "session is complete enough to answer. Judge it in the context of the earlier "
        "conversation: a follow-up, clarification or next-step request that refers to earlier "
        "turns is complete. A message that states a problem or asks about a topic is complete, "
        "even if it asks for a hint or does not say which method to use. Mark it incomplete "
        "only when it refers to material it does not contain (for example 'this problem' or "
        "'the figure' with nothing attached). "
        'Reply with JSON: {"complete": true/false, '
        '"clarification": "<question to ask if incomplete>"}'
    )
    history = [
        {
            "role": "user" if m.role == MessageRole.candidate else "assistant",
            "content": m.content,
        }
        for m in session.history[-_HISTORY_TURNS:]
    ]
    version = cfg.prompt_version("orchestrator", "completeness")
    response = await (meter or TurnMeter.detached()).complete(
        model=model,
        messages=[
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=COMPLETENESS_MAX_TOKENS,
        response_format={"type": "json_object"},
        **build_llm_metadata(trace_id=trace_id, prompt_version=version),
    )
    import json

    content = response.choices[0].message.content
    try:
        return cast(dict[str, Any], json.loads(content))
    except Exception:
        return {"complete": True, "clarification": ""}
