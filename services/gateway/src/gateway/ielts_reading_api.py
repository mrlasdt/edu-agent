"""
HTTP API for the IELTS Reading mock (`/ielts/reading/...`, JSON).

Reading is marked in code against the committed answer keys, so there is no model call, no spend
ledger and no background task: a submit marks the answers and returns the report in the same
response. It reuses the Writing attempt store (module `ielts_reading`), its server-side clock and
its rate limiter. Before submit nothing it returns carries a key or an evidence sentence.
"""
from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from services.gateway.src.gateway.ielts import reading
from services.gateway.src.gateway.ielts.models import DISCLAIMER, ReadingReport, Timing
from services.gateway.src.gateway.ielts_api import _limit, _load, get_store
from services.gateway.src.gateway.ielts_store import (
    DRAFT_MAX_CHARS,
    MODULE_READING,
    READING_ALLOWED_S,
    Attempt,
    AttemptStateError,
    AttemptStore,
    allowed_s_for,
)

router = APIRouter(prefix="/ielts/reading", tags=["ielts-reading"])

Store = Annotated[AttemptStore, Depends(get_store)]


class PassageOut(BaseModel):
    id: str
    title: str
    paragraphs: list[str]


class QuestionOut(BaseModel):
    """A question as the candidate sees it: no key and no evidence."""

    id: str
    number: int
    passage_id: str
    task_type: str
    task_label: str
    instruction: str
    prompt: str
    paragraph: int | None
    choices: dict[str, str]


class InfoOut(BaseModel):
    set_id: str
    allowed_s: int
    question_count: int
    task_types: dict[str, str]
    timing_source_url: str
    band_source_url: str
    band_note: str
    set_note: str
    keys_note: str


class PaperOut(BaseModel):
    passages: list[PassageOut]
    questions: list[QuestionOut]


class Answers(BaseModel):
    answers: dict[str, str] = Field(default_factory=dict)


class ReadingAttemptOut(BaseModel):
    id: str
    set_id: str
    status: str
    allowed_s: int
    remaining_s: int | None  # server-computed; None until the clock starts
    started_at: float | None
    submitted_at: float | None
    late: bool
    answers: dict[str, str]
    answered: int
    band: float | None = None
    raw: int | None = None
    scaled_40: int | None = None
    paper: PaperOut | None = None  # only while the attempt is in progress
    sample: bool = False  # the quick sample test: a few items on a short clock
    disclaimer: str = DISCLAIMER


def _paper_of(attempt: Attempt) -> reading.ReadingSet:
    return reading.get_sample_set() if attempt.sample else reading.get_set()


def _paper_out(paper: reading.ReadingSet) -> PaperOut:
    return PaperOut(
        passages=[PassageOut(id=p.id, title=p.title, paragraphs=list(p.paragraphs))
                  for p in paper.passages],
        questions=[
            QuestionOut(
                id=q.id, number=q.number, passage_id=q.passage_id, task_type=q.task_type,
                task_label=reading.TASK_LABELS.get(q.task_type, q.task_type),
                instruction=q.instruction, prompt=q.prompt, paragraph=q.paragraph,
                choices=q.choices,
            )
            for q in paper.questions
        ],
    )


def _report_of(attempt: Attempt) -> ReadingReport | None:
    return ReadingReport.model_validate_json(attempt.report_json) if attempt.report_json else None


def _view(attempt: Attempt, store: AttemptStore) -> ReadingAttemptOut:
    remaining = None
    if attempt.deadline_at is not None and attempt.status == "in_progress":
        remaining = max(0, int(attempt.deadline_at - store.clock()))
    answers = reading.read_answers(attempt.essay)
    report = _report_of(attempt)
    return ReadingAttemptOut(
        id=attempt.id, set_id=attempt.prompt_id, status=attempt.status,
        allowed_s=allowed_s_for(MODULE_READING, attempt.sample), remaining_s=remaining,
        started_at=attempt.started_at,
        submitted_at=attempt.submitted_at, late=attempt.late, answers=answers,
        answered=sum(1 for v in answers.values() if v.strip()),
        band=report.band if report else None, raw=report.raw if report else None,
        scaled_40=report.scaled_40 if report else None,
        paper=_paper_out(_paper_of(attempt)) if attempt.status == "in_progress" else None,
        sample=attempt.sample,
    )


def _load_reading(store: AttemptStore, attempt_id: str) -> Attempt:
    return _load(store, attempt_id, MODULE_READING)


def _encode(attempt: Attempt, answers: dict[str, str]) -> str:
    known = {q.id for q in _paper_of(attempt).questions}
    text = json.dumps({k: v for k, v in answers.items() if k in known}, ensure_ascii=False)
    if len(text) > DRAFT_MAX_CHARS:
        raise HTTPException(422, "answers are too long")
    return text


@router.get("/info")
def info() -> InfoOut:
    paper = reading.get_set()
    return InfoOut(
        set_id=paper.id, allowed_s=READING_ALLOWED_S, question_count=len(paper.questions),
        task_types=reading.TASK_LABELS, timing_source_url=reading.TIMING_SOURCE_URL,
        band_source_url=reading.BAND_SOURCE_URL, band_note=reading.BAND_TABLE_NOTE,
        set_note=reading.SET_NOTE, keys_note=reading.KEYS_NOTE,
    )


@router.get("/attempts")
def list_attempts(store: Store) -> list[ReadingAttemptOut]:
    return [_view(a, store) for a in store.recent(module=MODULE_READING)]


@router.post("/attempts", status_code=201)
def create_attempt(store: Store, sample: bool = False) -> ReadingAttemptOut:
    """`sample=true` starts the quick sample test: a few of the same items on a short clock."""
    _limit()
    paper = reading.get_sample_set() if sample else reading.get_set()
    attempt = store.create(paper.id, "IELTS General Training Reading", MODULE_READING, sample)
    return _view(attempt, store)


@router.get("/attempts/{attempt_id}")
def get_attempt(attempt_id: str, store: Store) -> ReadingAttemptOut:
    return _view(_load_reading(store, attempt_id), store)


@router.post("/attempts/{attempt_id}/start")
def start_attempt(attempt_id: str, store: Store) -> ReadingAttemptOut:
    _load_reading(store, attempt_id)
    return _view(store.start(attempt_id), store)


@router.put("/attempts/{attempt_id}/draft")
def save_draft(attempt_id: str, body: Answers, store: Store) -> ReadingAttemptOut:
    current = _load_reading(store, attempt_id)
    try:
        return _view(store.save_draft(attempt_id, _encode(current, body.answers)), store)
    except AttemptStateError as exc:
        raise HTTPException(409, str(exc)) from None


@router.post("/attempts/{attempt_id}/submit")
def submit_attempt(attempt_id: str, body: Answers, store: Store) -> ReadingAttemptOut:
    """Mark in code and return the report's headline; repeating a submit changes nothing."""
    current = _load_reading(store, attempt_id)
    try:
        attempt, submitted = store.submit(attempt_id, _encode(current, body.answers))
    except AttemptStateError as exc:
        raise HTTPException(409, str(exc)) from None
    if submitted:  # only the call that made the transition marks
        attempt, _ = store.begin_grading(attempt_id)
        assert attempt.started_at is not None and attempt.submitted_at is not None
        timing = Timing(
            allowed_s=allowed_s_for(MODULE_READING, attempt.sample),
            used_s=int(attempt.submitted_at - attempt.started_at),
            late=attempt.late,
        )
        report = reading.mark(
            _paper_of(attempt), reading.read_answers(attempt.essay), timing, attempt.sample
        )
        attempt = store.finish(attempt_id, report.model_dump_json(), graded=True)
    return _view(attempt, store)


@router.get("/attempts/{attempt_id}/report")
def get_report(attempt_id: str, store: Store) -> ReadingReport:
    attempt = _load_reading(store, attempt_id)
    report = _report_of(attempt)
    if report is None:
        raise HTTPException(409, f"attempt is {attempt.status}; no report yet")
    return report
