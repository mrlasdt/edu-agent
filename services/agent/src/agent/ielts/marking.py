"""
Mark one Writing Task 2 response end to end: checks, sampled grading, feedback, report.

Two outcomes are final and returned as an `ungraded` report (the response cannot be marked);
a model outage raises `GradingError` and a budget stop `BudgetExceededError`, at any stage, so
the caller can keep the essay and offer a retry.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from services.agent.src.agent.ielts import llm
from services.agent.src.agent.ielts.checks import check_essay
from services.agent.src.agent.ielts.feedback import Feedback, generate_feedback
from services.agent.src.agent.ielts.gate import load_calibration
from services.agent.src.agent.ielts.grading import (
    EssayNotMarkableError,
    GradeOutcome,
    grade_essay,
)
from services.agent.src.agent.ielts.models import (
    Calibration,
    CriterionReport,
    ErrorSpan,
    EstimatedBand,
    Flag,
    ModelInfo,
    PreviousAttempt,
    Priority,
    Report,
    TaskCheck,
    Timing,
)
from services.agent.src.agent.ielts.scoring import CRITERIA, CRITERION_NAMES, MAX_ESTIMATED_BAND
from shared.src.shared.budget import BudgetExceededError

LOW_CONFIDENCE_FLAGS = frozenset(
    {"possibly_off_topic", "copies_prompt", "addressed_to_marker", "repetitive_text"}
)
ERROR_GROUPS = {
    "CC": ("linking",),
    "LR": ("collocation", "spelling", "word_form"),
    "GRA": ("agreement", "tense", "article", "preposition", "punctuation", "other"),
}


def _confidence(outcome: GradeOutcome) -> str:
    n = len(outcome.samples)
    if n < 2 or outcome.spread > 1.0 or any(f.code in LOW_CONFIDENCE_FLAGS for f in outcome.flags):
        return "low"
    if any(outcome.caps.get(c, 9.0) < outcome.median[c] for c in CRITERIA):
        return "low"  # a rule overrode the marking passes, so their agreement says little
    return "high" if n >= 3 and outcome.spread <= 0.5 else "medium"


def _estimated_band(outcome: GradeOutcome, calibration: Calibration) -> EstimatedBand:
    overall = outcome.overall
    overalls = outcome.sample_overalls
    low = max(0.0, min([overall - 0.5, *overalls]))
    high = min(MAX_ESTIMATED_BAND, max([overall + 0.5, *overalls]))
    return EstimatedBand(
        overall=overall, range=(low, high), confidence=_confidence(outcome),  # type: ignore[arg-type]
        samples=len(outcome.samples), sample_spread=outcome.spread, calibration=calibration,
    )


def _criteria(outcome: GradeOutcome, feedback: Feedback | None) -> list[CriterionReport]:
    errors = feedback.errors if feedback else []
    out: list[CriterionReport] = []
    for crit in CRITERIA:
        fb = feedback.criteria.get(crit) if feedback else None
        table = Counter(e.type for e in errors if e.type in ERROR_GROUPS.get(crit, ()))
        out.append(CriterionReport(
            id=crit, name=CRITERION_NAMES[crit], band=outcome.criteria[crit],
            sample_bands=[s[crit] for s in outcome.samples],
            capped_from=outcome.median[crit] if outcome.criteria[crit] < outcome.median[crit]
            else None,
            verdict=fb.verdict if fb else "", evidence=fb.evidence if fb else [],
            holding_back=fb.holding_back if fb else "",
            to_reach_next_band=fb.to_reach_next_band if fb else "",
            actions=fb.actions if fb else [], paragraph_map=fb.paragraph_map if fb else [],
            error_table=dict(table) if crit in ERROR_GROUPS and feedback else {},
        ))
    return out


def top_priorities(criteria: list[CriterionReport]) -> list[Priority]:
    """The three lowest-banded criteria, each with its first action (ties keep TR, CC, LR, GRA)."""
    lowest = sorted(criteria, key=lambda c: c.band)[:3]
    priorities: list[Priority] = []
    for c in lowest:
        if c.actions:
            do, why = c.actions[0].do, c.actions[0].why
        elif c.to_reach_next_band:
            do, why = c.to_reach_next_band, c.holding_back
        else:
            continue
        priorities.append(Priority(priority=len(priorities) + 1, criterion=c.id, do=do, why=why))
    return priorities


def _task_checks(outcome: GradeOutcome, feedback: Feedback | None) -> list[TaskCheck]:
    checks = outcome.checks
    model: dict[str, bool | None] = feedback.checks if feedback else {}
    return [
        TaskCheck(check="at least 250 words", passed=checks.word_count >= 250),
        TaskCheck(check="four or more paragraphs", passed=checks.paragraph_count >= 4),
        TaskCheck(check="uses the key words of the question", passed=not checks.has(
            "possibly_off_topic")),
        TaskCheck(check="answers every part of the question",
                  passed=model.get("addresses_all_parts")),
        TaskCheck(check="clear position in the introduction",
                  passed=model.get("position_in_introduction")),
        TaskCheck(check="clear position in the conclusion",
                  passed=model.get("position_in_conclusion")),
    ]


def _feedback_status(feedback: Feedback | None) -> str:
    if feedback is None:
        return "unavailable"
    complete = not feedback.unverifiable and all(c.evidence for c in feedback.criteria.values())
    return "complete" if complete else "partial"


def ungraded_report(
    essay_words: int, flags: list[Flag], reason: str, timing: Timing | None
) -> Report:
    return Report(
        status="ungraded", word_count=essay_words, timing=timing, flags=flags,
        ungraded_reason=reason,
    )


async def mark_essay(
    prompt: str, essay: str, *, completer: llm.Completer, timing: Timing | None = None,
    previous: Mapping[str, Any] | None = None, samples: int = 3, trace_id: str = "",
    calibration: Calibration | None = None,
) -> Report:
    """`previous` is the earlier graded attempt: {"overall": float, "criteria": {id: band}}."""
    try:
        outcome = await grade_essay(
            prompt, essay, completer=completer, samples=samples, trace_id=trace_id
        )
    except EssayNotMarkableError as exc:
        checks = check_essay(prompt, essay)
        return ungraded_report(checks.word_count, checks.flags, exc.reason, timing)

    notes = [f.message for f in outcome.flags if f.severity != "info"]
    try:
        feedback: Feedback | None = await generate_feedback(
            prompt, essay, outcome.criteria, notes, completer=completer, trace_id=trace_id
        )
    except BudgetExceededError:
        raise
    except Exception:  # noqa: BLE001 - the bands stand without feedback; the report says so
        feedback = None

    criteria = _criteria(outcome, feedback)
    errors: list[ErrorSpan] = feedback.errors if feedback else []
    delta: PreviousAttempt | None = None
    if previous:
        delta = PreviousAttempt(
            overall_delta=outcome.overall - float(previous["overall"]),
            criteria_deltas={
                c: outcome.criteria[c] - float(previous["criteria"][c])
                for c in CRITERIA if c in previous["criteria"]
            },
        )
    return Report(
        status="graded", word_count=outcome.checks.word_count, timing=timing,
        estimated_band=_estimated_band(outcome, calibration or load_calibration()),
        criteria=criteria, error_spans=errors, task_checks=_task_checks(outcome, feedback),
        top_priorities=top_priorities(criteria) if feedback else [], flags=outcome.flags,
        feedback_status=_feedback_status(feedback),  # type: ignore[arg-type]
        vs_previous_attempt=delta,
        prompt_versions=ModelInfo(
            grader_models=[llm.role_model(llm.GRADER_ROLE)],
            grader_prompt=llm.prompt_version(llm.GRADER_ROLE),
            feedback_prompt=llm.prompt_version(llm.FEEDBACK_ROLE),
        ),
    )
