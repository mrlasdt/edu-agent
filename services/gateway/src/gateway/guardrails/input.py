"""Input guardrails: content safety check and completeness enrichment."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from pydantic import BaseModel, StrictBool

from services.gateway.src.gateway.guardrails.events import emit_guardrail_event
from services.gateway.src.gateway.guardrails.supportive import (
    SUPPORTIVE_MESSAGE,
    flagged_reason,
    is_self_harm_reason,
)
from services.gateway.src.gateway.guardrails.verdict import Verdict
from services.gateway.src.gateway.settings import get_settings
from shared.src.shared.config import PROJECT_ROOT, load_model_config
from shared.src.shared.models import MessageRole, Session
from shared.src.shared.observability import build_llm_metadata
from shared.src.shared.structured import parse_structured, response_format_for
from shared.src.shared.turn_meter import TurnMeter
from shared.src.shared.turn_outcome import Outcome

_log = logging.getLogger(__name__)

_HISTORY_TURNS = 6

# The completeness check answers with one short JSON object (a boolean and, when incomplete, one
# clarifying question). A reasoning model also spends hidden thinking tokens against this cap, so
# it is sized well above the visible answer: low enough to stop a runaway reply from delaying
# every turn before the tutor starts, high enough that a reasoning model still finishes. If the
# reply is cut off it does not validate: check_completeness raises, and /chat skips the check
# and answers the turn (flag `completeness_check_failed`), same as any other failure of it.
COMPLETENESS_MAX_TOKENS = 512

_COMPLETENESS_PROMPT_DIR = PROJECT_ROOT / "prompts" / "orchestrator" / "completeness"


def _load_completeness_prompt(version: str) -> str:
    """The completeness prompt file for `version`, without its YAML frontmatter."""
    raw = (_COMPLETENESS_PROMPT_DIR / f"{version}.md").read_text()
    return re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL).strip()


# Sent when the check finds the question incomplete but writes no clarifying question.
DEFAULT_CLARIFICATION = "Could you share the full question so I can help?"


class CompletenessVerdict(BaseModel):
    """The completeness check's reply. `complete` must be a real JSON boolean: the string
    "false" is truthy, so accepting it would read an incomplete question as complete."""

    complete: StrictBool
    clarification: str = ""


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


async def check_completeness(
    text: str, session: Session, trace_id: str, meter: TurnMeter | None = None
) -> Verdict:
    """
    Haiku-tier LLM check: is the question complete enough to answer?
    Returns `Verdict.allow()` when it is, else a `clarification_requested` stop whose message is
    the clarifying question (DEFAULT_CLARIFICATION when the model wrote none). With a `meter`,
    its model call is measured.

    A reply that is not valid JSON of the CompletenessVerdict shape never counts as complete:
    the failure is logged as a guardrail "error" event and raised, like any other failure of the
    call, for the caller to handle.
    """
    try:
        verdict = await _call_enrichment_llm(text, session, trace_id, meter)
    except Exception as exc:
        emit_guardrail_event(
            "input", "completeness", "error", trace_id, reason=type(exc).__name__
        )
        raise
    label = "pass" if verdict.complete else "fail"
    emit_guardrail_event("input", "completeness", label, trace_id)
    if verdict.complete:
        return Verdict.allow()
    return Verdict.stop(
        Outcome.clarification_requested, verdict.clarification or DEFAULT_CLARIFICATION
    )


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
) -> CompletenessVerdict:
    """
    Haiku-tier LLM call to detect incomplete questions. Returns the validated reply; raises
    StructuredOutputError if the model's reply is not a CompletenessVerdict.
    Seam for unit test patching.
    """
    cfg = load_model_config()
    model = cfg.litellm_model("orchestrator")
    history = [
        {
            "role": "user" if m.role == MessageRole.candidate else "assistant",
            "content": m.content,
        }
        for m in session.history[-_HISTORY_TURNS:]
    ]
    version = cfg.prompt_version("orchestrator", "completeness")
    system = _load_completeness_prompt(version).replace("{section}", session.section.upper())
    response = await (meter or TurnMeter.detached()).complete(
        model=model,
        messages=[
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=COMPLETENESS_MAX_TOKENS,
        response_format=response_format_for(model, CompletenessVerdict),
        **build_llm_metadata(trace_id=trace_id, prompt_version=version),
    )
    return parse_structured(response.choices[0].message.content, CompletenessVerdict)
