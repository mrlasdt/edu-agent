"""
HTTP API for the IELTS Listening mock (`/ielts/listening/...`, JSON plus the audio files).

Listening is marked in code against the committed answer keys, so there is no model call, no
spend ledger and no background task: a submit marks the answers and returns the report in the
same response. The recordings were made once, at build time, by `scripts/build_listening_audio.py`;
this API only serves those files, so an attempt never calls text-to-speech. It reuses the attempt
store (module `ielts_listening`), its server-side clock and its rate limiter. Before submit
nothing it returns carries a key, an evidence sentence or a transcript.
"""
from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from services.gateway.src.gateway.ielts import listening
from services.gateway.src.gateway.ielts.models import DISCLAIMER, ListeningReport, Timing
from services.gateway.src.gateway.ielts_api import _limit, _load, get_store
from services.gateway.src.gateway.ielts_store import (
    DRAFT_MAX_CHARS,
    LISTENING_ALLOWED_S,
    MODULE_LISTENING,
    Attempt,
    AttemptStateError,
    AttemptStore,
)

router = APIRouter(prefix="/ielts/listening", tags=["ielts-listening"])

Store = Annotated[AttemptStore, Depends(get_store)]


class SectionOut(BaseModel):
    id: str
    number: int
    title: str
    context: str
    audio_url: str
    audio_s: float | None  # length of the recording, None if it was never built


class QuestionOut(BaseModel):
    """A question as the candidate sees it: no key and no evidence."""

    id: str
    number: int
    section_id: str
    task_type: str
    task_label: str
    instruction: str
    prompt: str
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
    sections: list[SectionOut]
    questions: list[QuestionOut]


class Answers(BaseModel):
    answers: dict[str, str] = Field(default_factory=dict)


class ListeningAttemptOut(BaseModel):
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
    disclaimer: str = DISCLAIMER


def _paper_out(paper: listening.ListeningSet, attempt_id: str) -> PaperOut:
    return PaperOut(
        sections=[
            SectionOut(
                id=s.id, number=s.number, title=s.title, context=s.context,
                audio_url=f"/ielts/listening/attempts/{attempt_id}/audio/{s.id}",
                audio_s=listening.audio_duration(s.id),
            )
            for s in paper.sections
        ],
        questions=[
            QuestionOut(
                id=q.id, number=q.number, section_id=q.section_id, task_type=q.task_type,
                task_label=listening.TASK_LABELS.get(q.task_type, q.task_type),
                instruction=q.instruction, prompt=q.prompt, choices=q.choices,
            )
            for q in paper.questions
        ],
    )


def _report_of(attempt: Attempt) -> ListeningReport | None:
    return ListeningReport.model_validate_json(attempt.report_json) if attempt.report_json else None


def _view(attempt: Attempt, store: AttemptStore) -> ListeningAttemptOut:
    remaining = None
    if attempt.deadline_at is not None and attempt.status == "in_progress":
        remaining = max(0, int(attempt.deadline_at - store.clock()))
    answers = listening.read_answers(attempt.essay)
    report = _report_of(attempt)
    return ListeningAttemptOut(
        id=attempt.id, set_id=attempt.prompt_id, status=attempt.status,
        allowed_s=LISTENING_ALLOWED_S, remaining_s=remaining, started_at=attempt.started_at,
        submitted_at=attempt.submitted_at, late=attempt.late, answers=answers,
        answered=sum(1 for v in answers.values() if v.strip()),
        band=report.band if report else None, raw=report.raw if report else None,
        scaled_40=report.scaled_40 if report else None,
        paper=(
            _paper_out(listening.get_set(), attempt.id) if attempt.status == "in_progress" else None
        ),
    )


def _load_listening(store: AttemptStore, attempt_id: str) -> Attempt:
    return _load(store, attempt_id, MODULE_LISTENING)


def _encode(answers: dict[str, str]) -> str:
    known = {q.id for q in listening.get_set().questions}
    text = json.dumps({k: v for k, v in answers.items() if k in known}, ensure_ascii=False)
    if len(text) > DRAFT_MAX_CHARS:
        raise HTTPException(422, "answers are too long")
    return text


@router.get("/info")
def info() -> InfoOut:
    paper = listening.get_set()
    return InfoOut(
        set_id=paper.id, allowed_s=LISTENING_ALLOWED_S, question_count=len(paper.questions),
        task_types=listening.TASK_LABELS, timing_source_url=listening.TIMING_SOURCE_URL,
        band_source_url=listening.BAND_SOURCE_URL, band_note=listening.BAND_TABLE_NOTE,
        set_note=listening.SET_NOTE, keys_note=listening.KEYS_NOTE,
    )


@router.get("/attempts")
def list_attempts(store: Store) -> list[ListeningAttemptOut]:
    return [_view(a, store) for a in store.recent(module=MODULE_LISTENING)]


@router.post("/attempts", status_code=201)
def create_attempt(store: Store) -> ListeningAttemptOut:
    _limit()
    paper = listening.get_set()
    attempt = store.create(paper.id, "IELTS General Training Listening", MODULE_LISTENING)
    return _view(attempt, store)


@router.get("/attempts/{attempt_id}")
def get_attempt(attempt_id: str, store: Store) -> ListeningAttemptOut:
    return _view(_load_listening(store, attempt_id), store)


@router.post("/attempts/{attempt_id}/start")
def start_attempt(attempt_id: str, store: Store) -> ListeningAttemptOut:
    _load_listening(store, attempt_id)
    return _view(store.start(attempt_id), store)


@router.get("/attempts/{attempt_id}/audio/{section_id}")
def get_audio(attempt_id: str, section_id: str, store: Store) -> FileResponse:
    """A recording, only once the attempt has started. Range requests let the player seek."""
    attempt = _load_listening(store, attempt_id)
    if attempt.status == "created":
        raise HTTPException(409, "start the attempt to hear the recordings")
    path = listening.audio_path(section_id)
    if path is None:
        raise HTTPException(404, "recording not found")
    return FileResponse(
        path, media_type="audio/mpeg", headers={"Cache-Control": "private, max-age=3600"}
    )


@router.put("/attempts/{attempt_id}/draft")
def save_draft(attempt_id: str, body: Answers, store: Store) -> ListeningAttemptOut:
    _load_listening(store, attempt_id)
    try:
        return _view(store.save_draft(attempt_id, _encode(body.answers)), store)
    except AttemptStateError as exc:
        raise HTTPException(409, str(exc)) from None


@router.post("/attempts/{attempt_id}/submit")
def submit_attempt(attempt_id: str, body: Answers, store: Store) -> ListeningAttemptOut:
    """Mark in code and return the report's headline; repeating a submit changes nothing."""
    _load_listening(store, attempt_id)
    try:
        attempt, submitted = store.submit(attempt_id, _encode(body.answers))
    except AttemptStateError as exc:
        raise HTTPException(409, str(exc)) from None
    if submitted:  # only the call that made the transition marks
        attempt, _ = store.begin_grading(attempt_id)
        assert attempt.started_at is not None and attempt.submitted_at is not None
        timing = Timing(
            allowed_s=LISTENING_ALLOWED_S,
            used_s=int(attempt.submitted_at - attempt.started_at),
            late=attempt.late,
        )
        report = listening.mark(listening.get_set(), listening.read_answers(attempt.essay), timing)
        attempt = store.finish(attempt_id, report.model_dump_json(), graded=True)
    return _view(attempt, store)


@router.get("/attempts/{attempt_id}/report")
def get_report(attempt_id: str, store: Store) -> ListeningReport:
    attempt = _load_listening(store, attempt_id)
    report = _report_of(attempt)
    if report is None:
        raise HTTPException(409, f"attempt is {attempt.status}; no report yet")
    return report
