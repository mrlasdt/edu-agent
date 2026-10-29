"""
HTTP API for the IELTS-style Writing Task 2 mock (`/ielts/...`, JSON).

Deliberately not routed through `/chat`: a graded submission is not a tutoring turn, so the
content-safety and completeness guardrails (and the PII redaction that would corrupt an
essay) do not apply. The rate limiter does. Essay text is never logged.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from services.agent.src.agent.ielts.checks import words
from services.agent.src.agent.ielts.grading import GradingError
from services.agent.src.agent.ielts.llm import Completer, LiteLLMCompleter
from services.agent.src.agent.ielts.marking import mark_essay
from services.agent.src.agent.ielts.models import DISCLAIMER, Report, Timing
from services.agent.src.agent.ielts.prompts import WritingPrompt, load_prompts
from services.gateway.src.gateway.guardrails.rate_limit import RateLimiter
from services.gateway.src.gateway.ielts_store import (
    ALLOWED_S,
    DRAFT_MAX_CHARS,
    Attempt,
    AttemptNotFoundError,
    AttemptStateError,
    AttemptStore,
)
from shared.src.shared.config import get_settings

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/ielts", tags=["ielts"])

_CANDIDATE = "ielts-local"  # single local user
_limiter = RateLimiter(max_per_minute=10, max_per_day=60)
_inflight: set[str] = set()  # attempts being marked by this process


@lru_cache
def get_store() -> AttemptStore:
    return AttemptStore(Path(get_settings().ielts_data_dir) / "attempts.sqlite3")


@lru_cache
def get_completer() -> Completer:
    return LiteLLMCompleter()


@lru_cache
def get_prompts() -> dict[str, WritingPrompt]:
    return {p.id: p for p in load_prompts()}


Store = Annotated[AttemptStore, Depends(get_store)]
Marker = Annotated[Completer, Depends(get_completer)]


class PromptOut(BaseModel):
    id: str
    essay_type: str
    essay_type_label: str
    text: str


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
    status = attempt.status
    error, retryable = attempt.error, attempt.retryable
    if status == "grading" and attempt.id not in _inflight:
        # the process restarted mid-marking: the essay is safe, marking can be retried
        attempt = store.fail(
            attempt.id, "Marking was interrupted. Your essay is saved; try marking again."
        )
        status, error, retryable = attempt.status, attempt.error, attempt.retryable
    remaining = None
    if attempt.deadline_at is not None and status == "in_progress":
        remaining = max(0, int(attempt.deadline_at - store.clock()))
    return AttemptOut(
        id=attempt.id, prompt_id=attempt.prompt_id, prompt=attempt.prompt_text, status=status,
        allowed_s=ALLOWED_S, remaining_s=remaining, started_at=attempt.started_at,
        submitted_at=attempt.submitted_at, late=attempt.late, draft=attempt.essay,
        word_count=len(words(attempt.essay)), error=error, retryable=retryable,
        overall=_overall_of(attempt.report_json),
    )


def _load(store: AttemptStore, attempt_id: str) -> Attempt:
    try:
        return store.get(attempt_id)
    except AttemptNotFoundError:
        raise HTTPException(404, "attempt not found") from None


def _limit() -> None:
    allowed, message = _limiter.check(_CANDIDATE)
    if not allowed:
        raise HTTPException(429, message)


async def grade_attempt(attempt_id: str, store: AttemptStore, completer: Completer) -> None:
    """Mark a submitted attempt and store the outcome; a model outage leaves it retryable."""
    _inflight.add(attempt_id)
    try:
        attempt = store.begin_grading(attempt_id)
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
            previous=(
                {"overall": prev_report.estimated_band.overall,
                 "criteria": {c.id: c.band for c in prev_report.criteria}}
                if prev_report and prev_report.estimated_band else None
            ),
            samples=get_settings().ielts_grader_samples, trace_id=attempt_id,
        )
        store.finish(attempt_id, report.model_dump_json(), graded=report.status == "graded")
    except GradingError as exc:
        store.fail(attempt_id, exc.reason, retryable=True)
    except Exception:  # noqa: BLE001 - keep the essay and offer a retry; the cause goes to the log
        _log.exception("marking attempt %s failed", attempt_id)
        store.fail(attempt_id, "Marking failed unexpectedly. Your essay is saved; try again.")
    finally:
        _inflight.discard(attempt_id)


@router.get("/prompts")
def list_prompts() -> list[PromptOut]:
    return [PromptOut(**vars(p)) for p in get_prompts().values()]


@router.post("/attempts", status_code=201)
def create_attempt(body: CreateAttempt, store: Store) -> AttemptOut:
    prompt = get_prompts().get(body.prompt_id)
    if prompt is None:
        raise HTTPException(404, "unknown prompt")
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
) -> AttemptOut:
    _load(store, attempt_id)
    try:
        before = store.get(attempt_id).status
        attempt = store.submit(attempt_id, body.text)
    except AttemptStateError as exc:
        raise HTTPException(409, str(exc)) from None
    if before == "in_progress":  # a repeated submit must not mark twice
        background.add_task(grade_attempt, attempt_id, store, completer)
    return _view(attempt, store)


@router.post("/attempts/{attempt_id}/regrade")
def regrade_attempt(
    attempt_id: str, background: BackgroundTasks, store: Store, completer: Marker,
) -> AttemptOut:
    attempt = _load(store, attempt_id)
    view = _view(attempt, store)  # settles a stale 'grading' state first
    attempt = store.get(attempt_id)
    if attempt.status != "ungraded" or not attempt.retryable:
        raise HTTPException(409, f"attempt is {view.status}; nothing to retry")
    _limit()
    attempt = store.begin_grading(attempt_id)
    _inflight.add(attempt_id)
    background.add_task(grade_attempt, attempt_id, store, completer)
    return _view(attempt, store)


@router.get("/attempts/{attempt_id}/report")
def get_report(attempt_id: str, store: Store) -> Report:
    attempt = _load(store, attempt_id)
    if not attempt.report_json:
        raise HTTPException(409, f"attempt is {attempt.status}; no report yet")
    return Report.model_validate_json(attempt.report_json)
