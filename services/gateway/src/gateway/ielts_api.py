"""
HTTP API for the IELTS Writing Task 2 mock (`/ielts/...`, JSON).

Deliberately not routed through `/chat`: a graded submission is not a tutoring turn, so the
content-safety and completeness guardrails (and the PII redaction that would corrupt an
essay) do not apply. The rate limiter does. Essay text is never logged.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from services.gateway.src.gateway.ielts.checks import words
from services.gateway.src.gateway.ielts.grading import GradingError
from services.gateway.src.gateway.ielts.llm import Completer, LiteLLMCompleter
from services.gateway.src.gateway.ielts.marking import mark_essay
from services.gateway.src.gateway.ielts.models import DISCLAIMER, Report, Timing
from services.gateway.src.gateway.ielts.prompts import WritingPrompt, load_prompts
from services.gateway.src.gateway.ielts_store import (
    ALLOWED_S,
    DRAFT_MAX_CHARS,
    MODULE_WRITING_T2,
    Attempt,
    AttemptNotFoundError,
    AttemptStateError,
    AttemptStore,
)
from services.gateway.src.gateway.limits import ielts_limiter
from services.gateway.src.gateway.settings import get_settings
from shared.src.shared.budget import (
    BudgetExceededError,
    Ledger,
    UnreadableLedgerError,
    spend_ledger,
)

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/ielts", tags=["ielts"])

_CANDIDATE = "ielts-local"  # single local user

# One mock costs about 0.01 to 0.02 USD with the default model. A new attempt (or a retry) is
# refused unless at least this much of the spend cap is left, so a candidate is not asked to
# write for 40 minutes and then told the marking cannot be paid for.
MOCK_RESERVE_USD = 0.05


@lru_cache
def get_store() -> AttemptStore:
    return AttemptStore(Path(get_settings().ielts_data_dir) / "attempts.sqlite3")


@lru_cache
def get_completer() -> Completer:
    return LiteLLMCompleter()


@lru_cache
def get_prompts() -> dict[str, WritingPrompt]:
    return {p.id: p for p in load_prompts()}


def get_ledger() -> Ledger:
    return spend_ledger()


Store = Annotated[AttemptStore, Depends(get_store)]
Marker = Annotated[Completer, Depends(get_completer)]
SpendLedger = Annotated[Ledger, Depends(get_ledger)]


class PromptOut(BaseModel):
    id: str
    essay_type: str
    essay_type_label: str
    text: str


class BudgetOut(BaseModel):
    cap_usd: float
    spent_usd: float
    remaining_usd: float
    exhausted: bool  # true when a new mock would be refused
    message: str | None  # the explanation shown to the candidate when exhausted


class CreateAttempt(BaseModel):
    prompt_id: str


class EssayText(BaseModel):
    text: str = Field(max_length=DRAFT_MAX_CHARS)


class AttemptOut(BaseModel):
    id: str
    prompt_id: str
    prompt: str
    status: str
    allowed_s: int
    remaining_s: int | None  # server-computed; None until the clock starts
    started_at: float | None
    submitted_at: float | None
    late: bool
    draft: str
    word_count: int
    error: str | None
    retryable: bool
    overall: float | None = None
    disclaimer: str = DISCLAIMER


def _overall_of(report_json: str | None) -> float | None:
    if not report_json:
        return None
    band = json.loads(report_json).get("estimated_band")
    return band["overall"] if band else None


def _view(attempt: Attempt, store: AttemptStore) -> AttemptOut:
    remaining = None
    if attempt.deadline_at is not None and attempt.status == "in_progress":
        remaining = max(0, int(attempt.deadline_at - store.clock()))
    return AttemptOut(
        id=attempt.id, prompt_id=attempt.prompt_id, prompt=attempt.prompt_text,
        status=attempt.status, allowed_s=ALLOWED_S, remaining_s=remaining,
        started_at=attempt.started_at, submitted_at=attempt.submitted_at, late=attempt.late,
        draft=attempt.essay, word_count=len(words(attempt.essay)), error=attempt.error,
        retryable=attempt.retryable, overall=_overall_of(attempt.report_json),
    )


INTERRUPTED_MESSAGE = "Marking was interrupted. Your essay is saved; try marking again."


def recover_interrupted_attempts(store: AttemptStore) -> int:
    """
    Run once at startup. Marking runs as a background task of this process, so an attempt still
    'grading' now lost its worker with the previous process: keep the essay, offer a retry.
    """
    settled = store.fail_interrupted(INTERRUPTED_MESSAGE)
    if settled:
        _log.warning("marked %d attempt(s) interrupted by a restart as retryable", settled)
    return settled


def _load(store: AttemptStore, attempt_id: str, module: str = MODULE_WRITING_T2) -> Attempt:
    try:
        attempt = store.get(attempt_id)
    except AttemptNotFoundError:
        raise HTTPException(404, "attempt not found") from None
    if attempt.module != module:
        raise HTTPException(404, "attempt not found")
    return attempt


def _budget(ledger: Ledger) -> BudgetOut:
    try:
        spent, remaining = ledger.spent(), ledger.remaining()
    except UnreadableLedgerError as exc:
        return BudgetOut(
            cap_usd=ledger.ceiling_usd, spent_usd=0.0, remaining_usd=0.0,
            exhausted=True, message=str(exc),
        )
    exhausted = remaining < MOCK_RESERVE_USD
    return BudgetOut(
        cap_usd=ledger.ceiling_usd, spent_usd=spent, remaining_usd=remaining,
        exhausted=exhausted, message=cap_message(ledger) if exhausted else None,
    )


def cap_message(ledger: Ledger) -> str:
    try:
        spent = ledger.spent()
    except UnreadableLedgerError as exc:
        return str(exc)
    return (
        f"The AI spend cap of ${ledger.ceiling_usd:.2f} is used up (${spent:.2f} spent), "
        "so no new mock can be marked. Earlier attempts and saved essays stay available. "
        "Raise AI_SPEND_CAP_USD to continue."
    )


def _require_budget(ledger: Ledger) -> None:
    try:
        remaining = ledger.remaining()
    except UnreadableLedgerError as exc:
        raise HTTPException(402, str(exc)) from exc
    if remaining < MOCK_RESERVE_USD:
        raise HTTPException(402, cap_message(ledger))


def _limit() -> None:
    allowed, message = ielts_limiter().check(_CANDIDATE)
    if not allowed:
        raise HTTPException(429, message)


def _previous_of(report: Report | None) -> dict[str, Any] | None:
    """A graded report as `mark_essay`'s `previous` argument: what a new mark is compared with."""
    if report is None or report.estimated_band is None:
        return None
    return {
        "overall": report.estimated_band.overall,
        "criteria": {c.id: c.band for c in report.criteria},
    }


async def grade_attempt(
    attempt_id: str, store: AttemptStore, completer: Completer, ledger: Ledger | None = None,
) -> None:
    """
    Mark an attempt the caller already moved to 'grading' and store the outcome; a model outage
    leaves it retryable. Only the call that made that transition may queue this.
    """
    try:
        attempt = store.get(attempt_id)
        assert attempt.started_at is not None and attempt.submitted_at is not None
        previous = store.previous_graded(attempt)
        prev_report = Report.model_validate_json(previous.report_json) if (
            previous and previous.report_json
        ) else None
        timing = Timing(
            allowed_s=ALLOWED_S,
            used_s=int(attempt.submitted_at - attempt.started_at),
            late=attempt.late,
        )
        report = await mark_essay(
            attempt.prompt_text, attempt.essay, completer=completer, timing=timing,
            previous=_previous_of(prev_report),
            samples=get_settings().ielts_grader_samples, trace_id=attempt_id,
        )
        store.finish(attempt_id, report.model_dump_json(), graded=report.status == "graded")
    except GradingError as exc:
        store.fail(attempt_id, exc.reason, retryable=True)
    except BudgetExceededError:  # the cap was hit mid-attempt: the essay stays, the retry waits
        store.fail(attempt_id, cap_message(ledger or spend_ledger()), retryable=True)
    except Exception:  # noqa: BLE001 - keep the essay and offer a retry; the cause goes to the log
        _log.exception("marking attempt %s failed", attempt_id)
        store.fail(attempt_id, "Marking failed unexpectedly. Your essay is saved; try again.")


@router.get("/prompts")
def list_prompts() -> list[PromptOut]:
    return [PromptOut(**vars(p)) for p in get_prompts().values()]


@router.get("/budget")
def get_budget(ledger: SpendLedger) -> BudgetOut:
    return _budget(ledger)


@router.post("/attempts", status_code=201)
def create_attempt(body: CreateAttempt, store: Store, ledger: SpendLedger) -> AttemptOut:
    prompt = get_prompts().get(body.prompt_id)
    if prompt is None:
        raise HTTPException(404, "unknown prompt")
    _require_budget(ledger)
    _limit()
    return _view(store.create(prompt.id, prompt.text), store)


@router.get("/attempts")
def list_attempts(store: Store) -> list[AttemptOut]:
    return [_view(a, store) for a in store.recent()]


@router.get("/attempts/{attempt_id}")
def get_attempt(attempt_id: str, store: Store) -> AttemptOut:
    return _view(_load(store, attempt_id), store)


@router.post("/attempts/{attempt_id}/start")
def start_attempt(attempt_id: str, store: Store) -> AttemptOut:
    _load(store, attempt_id)
    return _view(store.start(attempt_id), store)


@router.put("/attempts/{attempt_id}/draft")
def save_draft(attempt_id: str, body: EssayText, store: Store) -> AttemptOut:
    _load(store, attempt_id)
    try:
        return _view(store.save_draft(attempt_id, body.text), store)
    except AttemptStateError as exc:
        raise HTTPException(409, str(exc)) from None


@router.post("/attempts/{attempt_id}/submit")
def submit_attempt(
    attempt_id: str, body: EssayText, background: BackgroundTasks, store: Store, completer: Marker,
    ledger: SpendLedger,
) -> AttemptOut:
    _load(store, attempt_id)
    try:
        attempt, submitted = store.submit(attempt_id, body.text)
    except AttemptStateError as exc:
        raise HTTPException(409, str(exc)) from None
    if submitted:  # only the call that made the transition marks; a repeated submit does not
        attempt, _ = store.begin_grading(attempt_id)
        background.add_task(grade_attempt, attempt_id, store, completer, ledger)
    return _view(attempt, store)


@router.post("/attempts/{attempt_id}/regrade")
def regrade_attempt(
    attempt_id: str, background: BackgroundTasks, store: Store, completer: Marker,
    ledger: SpendLedger,
) -> AttemptOut:
    attempt = _load(store, attempt_id)
    if attempt.status != "ungraded" or not attempt.retryable:
        raise HTTPException(409, f"attempt is {attempt.status}; nothing to retry")
    _require_budget(ledger)
    _limit()
    try:
        attempt, started = store.begin_grading(attempt_id)
    except AttemptStateError as exc:  # settled by another request since the check above
        raise HTTPException(409, str(exc)) from None
    if not started:
        raise HTTPException(409, f"attempt is {attempt.status}; nothing to retry")
    background.add_task(grade_attempt, attempt_id, store, completer, ledger)
    return _view(attempt, store)


PRACTICE_LABEL = "Practice re-mark (untimed, not saved as a mock attempt)"


class RemarkRequest(BaseModel):
    attempt_id: str = Field(max_length=64)  # the graded attempt the revision is compared with
    essay: str = Field(max_length=DRAFT_MAX_CHARS)


class CriterionRemark(BaseModel):
    id: str
    name: str
    band: float
    delta: float | None  # against the same criterion of the attempt; None when it has no band


class RemarkOut(BaseModel):
    """A practice re-mark: nothing about it is stored, and it is not a mock attempt."""

    status: str  # "graded" or "ungraded"
    label: str = PRACTICE_LABEL
    ungraded_reason: str | None = None
    word_count: int
    overall: float | None = None
    range: tuple[float, float] | None = None
    confidence: str | None = None
    overall_delta: float | None = None
    criteria: list[CriterionRemark] = []
    disclaimer: str = DISCLAIMER


@router.post("/remark")
async def remark_essay(
    body: RemarkRequest, store: Store, completer: Marker, ledger: SpendLedger
) -> RemarkOut:
    """
    Practice re-mark of a revised essay for the /chat coach (its `mark_revision` tool calls this;
    the agent service cannot reach the grader, the attempt store or the spend ledger). Same
    marking and the same spend cap and rate limit as a mock, but untimed and stored nowhere. The
    comparison is with the graded attempt the learner is being coached from. Refusals carry a
    `detail` the learner can read: 402 the cap is used up, 429 too many requests, 502 the model
    is unavailable, 409 the attempt has no graded report, 422 nothing to mark.
    """
    attempt = _load(store, body.attempt_id)
    previous_report = (
        Report.model_validate_json(attempt.report_json) if attempt.report_json else None
    )
    previous = _previous_of(previous_report)
    if previous is None:
        raise HTTPException(409, "that attempt has no graded report to compare with")
    if not body.essay.strip():
        raise HTTPException(422, "there is no essay text to mark")
    _require_budget(ledger)
    _limit()
    try:
        report = await mark_essay(
            attempt.prompt_text, body.essay, completer=completer, previous=previous,
            samples=get_settings().ielts_grader_samples, trace_id=f"remark-{attempt.id}",
        )
    except BudgetExceededError:  # the cap was hit mid-marking
        raise HTTPException(402, cap_message(ledger)) from None
    except GradingError as exc:
        raise HTTPException(502, exc.reason) from None
    band, delta = report.estimated_band, report.vs_previous_attempt
    if report.status != "graded" or band is None:
        return RemarkOut(
            status="ungraded", word_count=report.word_count,
            ungraded_reason=report.ungraded_reason,
        )
    return RemarkOut(
        status="graded", word_count=report.word_count, overall=band.overall, range=band.range,
        confidence=band.confidence, overall_delta=delta.overall_delta if delta else None,
        criteria=[
            CriterionRemark(
                id=c.id, name=c.name, band=c.band,
                delta=delta.criteria_deltas.get(c.id) if delta else None,
            )
            for c in report.criteria
        ],
    )


@router.get("/attempts/{attempt_id}/report")
def get_report(attempt_id: str, store: Store) -> Report:
    attempt = _load(store, attempt_id)
    if not attempt.report_json:
        raise HTTPException(409, f"attempt is {attempt.status}; no report yet")
    return Report.model_validate_json(attempt.report_json)
