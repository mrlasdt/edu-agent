"""
HTTP API for the IELTS General Training Writing module (`/ielts/writing/...`, JSON).

One attempt is a Task 1 letter and a Task 2 essay on one 60 minute clock. It reuses the Writing
attempt store (module `ielts_writing`), its server-side clock, `mark_essay` for each task, the
spend cap and the rate limiter. The two texts are stored as one JSON object in the attempt's text
columns and the two prompts as one JSON object in `prompt_text`; the report is a
`WritingModuleReport` holding both task reports and the combined band. Essay text is never logged.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from services.gateway.src.gateway.ielts.checks import words
from services.gateway.src.gateway.ielts.grading import GradingError
from services.gateway.src.gateway.ielts.llm import Completer
from services.gateway.src.gateway.ielts.marking import mark_essay
from services.gateway.src.gateway.ielts.models import (
    DISCLAIMER,
    Report,
    Timing,
    WritingModuleReport,
)
from services.gateway.src.gateway.ielts.prompts import LetterPrompt, load_letter_prompts
from services.gateway.src.gateway.ielts.scoring import (
    TASK_SOURCE_URLS,
    TASK_WEIGHTS_NOTE,
    writing_band,
)
from services.gateway.src.gateway.ielts_api import (
    _limit,
    _load,
    _require_budget,
    cap_message,
    get_completer,
    get_ledger,
    get_prompts,
    get_store,
)
from services.gateway.src.gateway.ielts_store import (
    DRAFT_MAX_CHARS,
    MODULE_WRITING,
    WRITING_ALLOWED_S,
    Attempt,
    AttemptStateError,
    AttemptStore,
)
from services.gateway.src.gateway.settings import get_settings
from shared.src.shared.budget import BudgetExceededError, Ledger, spend_ledger

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/ielts/writing", tags=["ielts-writing"])

Store = Annotated[AttemptStore, Depends(get_store)]
Marker = Annotated[Completer, Depends(get_completer)]
SpendLedger = Annotated[Ledger, Depends(get_ledger)]

TASK_KEYS = ("t1", "t2")
MODULE_TITLE = "IELTS General Training Writing"


def get_letters() -> dict[str, LetterPrompt]:
    return {p.id: p for p in load_letter_prompts()}


class TaskText(BaseModel):
    """One task of the module as the candidate sees it."""

    prompt_id: str
    prompt: str
    text: str
    word_count: int
    band: float | None = None


class ModuleTexts(BaseModel):
    t1: str = Field(default="", max_length=DRAFT_MAX_CHARS // 2)
    t2: str = Field(default="", max_length=DRAFT_MAX_CHARS // 2)


class CreateModuleAttempt(BaseModel):
    t1_prompt_id: str | None = None  # a random one when omitted
    t2_prompt_id: str | None = None


class ModuleAttemptOut(BaseModel):
    id: str
    status: str
    allowed_s: int
    remaining_s: int | None  # server-computed; None until the clock starts
    started_at: float | None
    submitted_at: float | None
    late: bool
    t1: TaskText
    t2: TaskText
    error: str | None
    retryable: bool
    writing_band: float | None = None
    disclaimer: str = DISCLAIMER


def _decode(raw: str) -> dict[str, str]:
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        data = {}
    return {k: str(data.get(k, "")) for k in TASK_KEYS} if isinstance(data, dict) else {
        k: "" for k in TASK_KEYS
    }


def _encode(texts: ModuleTexts) -> str:
    return json.dumps({"t1": texts.t1, "t2": texts.t2}, ensure_ascii=False)


def _prompts_of(attempt: Attempt) -> dict[str, dict[str, str]]:
    prompts: dict[str, dict[str, str]] = json.loads(attempt.prompt_text)
    return prompts  # {"t1": {"id", "text"}, "t2": {...}}


def _report_of(attempt: Attempt) -> WritingModuleReport | None:
    if not attempt.report_json:
        return None
    return WritingModuleReport.model_validate_json(attempt.report_json)


def _view(attempt: Attempt, store: AttemptStore) -> ModuleAttemptOut:
    remaining = None
    if attempt.deadline_at is not None and attempt.status == "in_progress":
        remaining = max(0, int(attempt.deadline_at - store.clock()))
    prompts, texts, report = _prompts_of(attempt), _decode(attempt.essay), _report_of(attempt)

    def task(key: str, sub: Report | None) -> TaskText:
        band = sub.estimated_band.overall if sub and sub.estimated_band else None
        return TaskText(
            prompt_id=prompts[key]["id"], prompt=prompts[key]["text"], text=texts[key],
            word_count=len(words(texts[key])), band=band,
        )

    return ModuleAttemptOut(
        id=attempt.id, status=attempt.status, allowed_s=WRITING_ALLOWED_S, remaining_s=remaining,
        started_at=attempt.started_at, submitted_at=attempt.submitted_at, late=attempt.late,
        t1=task("t1", report.t1 if report else None), t2=task("t2", report.t2 if report else None),
        error=attempt.error, retryable=attempt.retryable,
        writing_band=report.writing_band if report else None,
    )


def combine(t1: Report, t2: Report, timing: Timing) -> WritingModuleReport:
    """The module report from the two task reports; the band needs both to be graded."""
    both = (
        t1.status == "graded" and t2.status == "graded"
        and t1.estimated_band is not None and t2.estimated_band is not None
    )
    band = None
    if both:
        assert t1.estimated_band is not None and t2.estimated_band is not None
        band = writing_band({"T1": t1.estimated_band.overall, "T2": t2.estimated_band.overall})
    return WritingModuleReport(
        status="graded" if both else "partial", t1=t1, t2=t2, writing_band=band,
        weights_note=TASK_WEIGHTS_NOTE, weights_source_urls=list(TASK_SOURCE_URLS), timing=timing,
    )


async def grade_module_attempt(
    attempt_id: str, store: AttemptStore, completer: Completer, ledger: Ledger | None = None,
) -> None:
    """
    Mark both tasks of an attempt the caller already moved to 'grading', at the same time, and
    store the combined report. A model outage in either task leaves the attempt retryable.
    """
    try:
        attempt = store.get(attempt_id)
        assert attempt.started_at is not None and attempt.submitted_at is not None
        prompts, texts = _prompts_of(attempt), _decode(attempt.essay)
        timing = Timing(
            allowed_s=WRITING_ALLOWED_S,
            used_s=int(attempt.submitted_at - attempt.started_at),
            late=attempt.late,
        )
        samples = get_settings().ielts_grader_samples
        t1, t2 = await asyncio.gather(
            mark_essay(
                prompts["t1"]["text"], texts["t1"], completer=completer, timing=timing,
                samples=samples, trace_id=f"{attempt_id}-t1", task="T1",
            ),
            mark_essay(
                prompts["t2"]["text"], texts["t2"], completer=completer, timing=timing,
                samples=samples, trace_id=f"{attempt_id}-t2", task="T2",
            ),
        )
        report = combine(t1, t2, timing)
        store.finish(attempt_id, report.model_dump_json(), graded=report.status == "graded")
    except GradingError as exc:
        store.fail(attempt_id, exc.reason, retryable=True)
    except BudgetExceededError:
        store.fail(attempt_id, cap_message(ledger or spend_ledger()), retryable=True)
    except Exception:  # noqa: BLE001 - keep the texts and offer a retry; the cause goes to the log
        _log.exception("marking writing attempt %s failed", attempt_id)
        store.fail(attempt_id, "Marking failed unexpectedly. Your writing is saved; try again.")


@router.get("/info")
def info() -> dict[str, object]:
    return {
        "allowed_s": WRITING_ALLOWED_S, "t1_minutes": 20, "t2_minutes": 40,
        "t1_min_words": 150, "t2_min_words": 250,
        "weights_note": TASK_WEIGHTS_NOTE, "source_urls": list(TASK_SOURCE_URLS),
    }


@router.get("/attempts")
def list_attempts(store: Store) -> list[ModuleAttemptOut]:
    return [_view(a, store) for a in store.recent(module=MODULE_WRITING)]


@router.post("/attempts", status_code=201)
def create_attempt(
    body: CreateModuleAttempt, store: Store, ledger: SpendLedger
) -> ModuleAttemptOut:
    letters, essays = get_letters(), get_prompts()
    letter = letters.get(body.t1_prompt_id) if body.t1_prompt_id else random.choice(
        list(letters.values())
    )
    essay = essays.get(body.t2_prompt_id) if body.t2_prompt_id else random.choice(
        list(essays.values())
    )
    if letter is None or essay is None:
        raise HTTPException(404, "unknown prompt")
    _require_budget(ledger)
    _limit()
    prompt_text = json.dumps({
        "t1": {"id": letter.id, "text": letter.text},
        "t2": {"id": essay.id, "text": essay.text},
    })
    return _view(store.create(f"{letter.id}+{essay.id}", prompt_text, MODULE_WRITING), store)


def _load_module(store: AttemptStore, attempt_id: str) -> Attempt:
    return _load(store, attempt_id, MODULE_WRITING)


@router.get("/attempts/{attempt_id}")
def get_attempt(attempt_id: str, store: Store) -> ModuleAttemptOut:
    return _view(_load_module(store, attempt_id), store)


@router.post("/attempts/{attempt_id}/start")
def start_attempt(attempt_id: str, store: Store) -> ModuleAttemptOut:
    _load_module(store, attempt_id)
    return _view(store.start(attempt_id), store)


@router.put("/attempts/{attempt_id}/draft")
def save_draft(attempt_id: str, body: ModuleTexts, store: Store) -> ModuleAttemptOut:
    _load_module(store, attempt_id)
    try:
        return _view(store.save_draft(attempt_id, _encode(body)), store)
    except AttemptStateError as exc:
        raise HTTPException(409, str(exc)) from None


@router.post("/attempts/{attempt_id}/submit")
def submit_attempt(
    attempt_id: str, body: ModuleTexts, background: BackgroundTasks, store: Store,
    completer: Marker, ledger: SpendLedger,
) -> ModuleAttemptOut:
    _load_module(store, attempt_id)
    try:
        attempt, submitted = store.submit(attempt_id, _encode(body))
    except AttemptStateError as exc:
        raise HTTPException(409, str(exc)) from None
    if submitted:  # only the call that made the transition marks
        attempt, _ = store.begin_grading(attempt_id)
        background.add_task(grade_module_attempt, attempt_id, store, completer, ledger)
    return _view(attempt, store)


@router.post("/attempts/{attempt_id}/regrade")
def regrade_attempt(
    attempt_id: str, background: BackgroundTasks, store: Store, completer: Marker,
    ledger: SpendLedger,
) -> ModuleAttemptOut:
    attempt = _load_module(store, attempt_id)
    if attempt.status != "ungraded" or not attempt.retryable:
        raise HTTPException(409, f"attempt is {attempt.status}; nothing to retry")
    _require_budget(ledger)
    _limit()
    try:
        attempt, started = store.begin_grading(attempt_id)
    except AttemptStateError as exc:
        raise HTTPException(409, str(exc)) from None
    if not started:
        raise HTTPException(409, f"attempt is {attempt.status}; nothing to retry")
    background.add_task(grade_module_attempt, attempt_id, store, completer, ledger)
    return _view(attempt, store)


@router.get("/attempts/{attempt_id}/report")
def get_report(attempt_id: str, store: Store) -> WritingModuleReport:
    attempt = _load_module(store, attempt_id)
    report = _report_of(attempt)
    if report is None:
        raise HTTPException(409, f"attempt is {attempt.status}; no report yet")
    return report
