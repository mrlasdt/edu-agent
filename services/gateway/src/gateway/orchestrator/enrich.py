"""
The Orchestrator's enrichment step: a cheap model asks whether the learner's question is complete
enough to answer, and if not, what to ask them. It runs in `/chat` after the Orchestrator has
validated and routed the turn.
"""

from __future__ import annotations

from pydantic import BaseModel, StrictBool

from services.gateway.src.gateway.guardrails.events import emit_guardrail_event
from services.gateway.src.gateway.guardrails.verdict import Verdict
from shared.src.shared.config import load_model_config
from shared.src.shared.history import history_to_messages
from shared.src.shared.models import Session
from shared.src.shared.observability import build_llm_metadata
from shared.src.shared.prompts import read_prompt
from shared.src.shared.structured import parse_structured, response_format_for
from shared.src.shared.turn_meter import TurnMeter
from shared.src.shared.turn_outcome import Outcome

_HISTORY_TURNS = 6

# The completeness check answers with one short JSON object (a boolean and, when incomplete, one
# clarifying question). A reasoning model also spends hidden thinking tokens against this cap, so
# it is sized well above the visible answer: low enough to stop a runaway reply from delaying
# every turn before the tutor starts, high enough that a reasoning model still finishes. If the
# reply is cut off it does not validate: check_completeness raises, and /chat skips the check
# and answers the turn (flag `completeness_check_failed`), same as any other failure of it.
COMPLETENESS_MAX_TOKENS = 512


# Sent when the check finds the question incomplete but writes no clarifying question.
DEFAULT_CLARIFICATION = "Could you share the full question so I can help?"


class CompletenessVerdict(BaseModel):
    """The completeness check's reply. `complete` must be a real JSON boolean: the string
    "false" is truthy, so accepting it would read an incomplete question as complete."""

    complete: StrictBool
    clarification: str = ""


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
    history = history_to_messages(session.history[-_HISTORY_TURNS:])
    version = cfg.prompt_version("orchestrator", "completeness")
    system = read_prompt("orchestrator", "completeness", version)
    system = system.replace("{section}", session.section.upper())
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
