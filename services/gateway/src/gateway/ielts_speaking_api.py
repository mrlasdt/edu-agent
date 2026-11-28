"""
HTTP API for the IELTS Speaking mock (`/ielts/speaking/...`).

The browser records each part and uploads it as the raw request body (no multipart). The
recordings are files under `<ielts_data_dir>/speaking/<attempt_id>/part<n>`; no new storage engine
is used: the attempt store keeps the attempt, its clock and (as the draft) a small JSON record of
what was uploaded for each part. A submit needs all three parts. Marking runs in the background
(a model hears the audio and the code counts the fluency figures, see `ielts.speaking_grading`)
and the learner's voice is deleted once the report is stored: only the transcript stays. A marking
that fails keeps the recordings so a retry does not ask the learner to speak again.
"""
from __future__ import annotations

import json
import logging
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel

from services.gateway.src.gateway.ielts import speaking
from services.gateway.src.gateway.ielts.grading import GradingError
from services.gateway.src.gateway.ielts.llm import ModelUnavailableError, role_temperature
from services.gateway.src.gateway.ielts.models import DISCLAIMER, SpeakingReport, Timing
from services.gateway.src.gateway.ielts.speaking_grading import PartAudio, mark_speaking
from services.gateway.src.gateway.ielts.speaking_llm import (
    SPEAKING_ROLE,
    AudioAssessor,
    GeminiAudioAssessor,
)
from services.gateway.src.gateway.ielts_api import (
    _limit,
    _load,
    _require_budget,
    cap_message,
    get_ledger,
    get_store,
)
from services.gateway.src.gateway.ielts_store import (
    MODULE_SPEAKING,
    SPEAKING_ALLOWED_S,
    Attempt,
    AttemptStateError,
    AttemptStore,
)
from services.gateway.src.gateway.settings import get_settings
from shared.src.shared.budget import BudgetExceededError, Ledger, spend_ledger
from shared.src.shared.config import get_llm_settings, load_model_config

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/ielts/speaking", tags=["ielts-speaking"])

MODULE_TITLE = "IELTS Speaking"
EXTENSIONS = {
    "audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/wav": "wav",
    "audio/x-wav": "wav", "audio/mpeg": "mp3",
}


@lru_cache
def get_assessor() -> AudioAssessor:
    return GeminiAudioAssessor()


def get_audio_dir() -> Path:
    return Path(get_settings().ielts_data_dir) / "speaking"


Store = Annotated[AttemptStore, Depends(get_store)]
Assessor = Annotated[AudioAssessor, Depends(get_assessor)]
SpendLedger = Annotated[Ledger, Depends(get_ledger)]
AudioDir = Annotated[Path, Depends(get_audio_dir)]


class UploadedPart(BaseModel):
    part: int
    mime: str
    bytes: int
    duration_s: float
    file: str


class SpeakingAttemptOut(BaseModel):
    id: str
    set_id: str
    status: str
    allowed_s: int
    remaining_s: int | None
    started_at: float | None
    submitted_at: float | None
    late: bool
    uploaded: list[int]  # the parts that have a recording
    paper: dict[str, object]  # the questions and cue card, shown from the start
    band: float | None
    partial: bool | None
    error: str | None
    retryable: bool


class InfoOut(BaseModel):
    allowed_s: int
    part2_prep_s: int
    part2_talk_s: int
    sample: dict[str, object]  # the quick sample set, shown on the start page
    not_assessed: dict[str, str]
    descriptors_source_url: str
    format_source_url: str
    model: str
    set_note: str


SET_NOTE = (
    "The questions are the repository's own original prompts, written by an agent and not "
    "checked by a human; they are not real IELTS material."
)


def _parts_of(attempt: Attempt) -> dict[int, UploadedPart]:
    try:
        raw = json.loads(attempt.essay or "{}")
    except json.JSONDecodeError:
        return {}
    return {int(k): UploadedPart.model_validate(v) for k, v in raw.items()}


def _encode(parts: dict[int, UploadedPart]) -> str:
    return json.dumps({str(k): v.model_dump() for k, v in sorted(parts.items())})


def _report_of(attempt: Attempt) -> SpeakingReport | None:
    return SpeakingReport.model_validate_json(attempt.report_json) if attempt.report_json else None


def _view(attempt: Attempt, store: AttemptStore) -> SpeakingAttemptOut:
    remaining = None
    if attempt.deadline_at is not None and attempt.status == "in_progress":
        remaining = max(0, int(attempt.deadline_at - store.clock()))
    report = _report_of(attempt)
    return SpeakingAttemptOut(
        id=attempt.id, set_id=attempt.prompt_id, status=attempt.status,
        allowed_s=SPEAKING_ALLOWED_S, remaining_s=remaining, started_at=attempt.started_at,
        submitted_at=attempt.submitted_at, late=attempt.late, uploaded=sorted(_parts_of(attempt)),
        paper=speaking.public_set(speaking.get_set(attempt.prompt_id)),
        band=report.band if report else None, partial=report.partial if report else None,
        error=attempt.error, retryable=attempt.retryable,
    )


def _load_speaking(store: AttemptStore, attempt_id: str) -> Attempt:
    return _load(store, attempt_id, MODULE_SPEAKING)


def _dir(audio_dir: Path, attempt_id: str) -> Path:
    return audio_dir / attempt_id


def _delete_audio(audio_dir: Path, attempt_id: str) -> None:
    shutil.rmtree(_dir(audio_dir, attempt_id), ignore_errors=True)


async def grade_speaking_attempt(
    attempt_id: str, store: AttemptStore, assessor: AudioAssessor, audio_dir: Path,
    ledger: Ledger | None = None,
) -> None:
    """
    Mark an attempt the caller already moved to 'grading' and store the report. A model outage
    leaves it retryable with the recordings kept; once a report is stored the audio is deleted.
    """
    try:
        attempt = store.get(attempt_id)
        assert attempt.started_at is not None and attempt.submitted_at is not None
        uploaded = _parts_of(attempt)
        audio = [
            PartAudio(
                part=n, data=(_dir(audio_dir, attempt_id) / p.file).read_bytes(), mime=p.mime,
                duration_s=p.duration_s,
            )
            for n, p in sorted(uploaded.items())
        ]
        timing = Timing(
            allowed_s=SPEAKING_ALLOWED_S, used_s=int(attempt.submitted_at - attempt.started_at),
            late=attempt.late,
        )
        config = load_model_config(get_llm_settings().model_config_path)
        report = await mark_speaking(
            speaking.get_set(attempt.prompt_id), audio, assessor=assessor, timing=timing,
            run_id=f"ielts-speaking-{attempt_id}", temperature=role_temperature(SPEAKING_ROLE),
            prompt_version=config.prompt_version("ielts_speaking", "assess"),
        )
        store.finish(attempt_id, report.model_dump_json(), graded=True)
        _delete_audio(audio_dir, attempt_id)
    except GradingError as exc:
        store.fail(attempt_id, exc.reason, retryable=True)
    except BudgetExceededError:
        store.fail(attempt_id, cap_message(ledger or spend_ledger()), retryable=True)
    except ModelUnavailableError as exc:
        _log.warning("speaking attempt %s: no audio model answered: %s", attempt_id, exc)
        store.fail(
            attempt_id,
            "The marking model is busy or unavailable. Your recordings are saved; try again.",
            retryable=True,
        )
    except Exception:  # noqa: BLE001 - keep the recordings and offer a retry; cause goes to the log
        _log.exception("marking speaking attempt %s failed", attempt_id)
        store.fail(attempt_id, "Marking failed unexpectedly. Your recordings are saved; try again.")


@router.get("/info")
def info() -> InfoOut:
    config = load_model_config(get_llm_settings().model_config_path)
    return InfoOut(
        allowed_s=SPEAKING_ALLOWED_S, part2_prep_s=speaking.PART2_PREP_S,
        part2_talk_s=speaking.PART2_TALK_S,
        not_assessed=speaking.UNASSESSED, descriptors_source_url=speaking.DESCRIPTORS_URL,
        format_source_url=speaking.FORMAT_URL, model=config.role(SPEAKING_ROLE).model,
        set_note=SET_NOTE, sample=speaking.public_set(speaking.SAMPLE_SET),
    )


@router.get("/attempts")
def list_attempts(store: Store) -> list[SpeakingAttemptOut]:
    return [_view(a, store) for a in store.recent(module=MODULE_SPEAKING)]


@router.post("/attempts", status_code=201)
def create_attempt(
    store: Store, ledger: SpendLedger, sample: bool = False,
) -> SpeakingAttemptOut:
    """`?sample=true` makes the quick sample set: one question per part and short timings."""
    item = speaking.SAMPLE_SET if sample else speaking.pick_set()
    _require_budget(ledger)
    _limit()
    return _view(store.create(item.id, MODULE_TITLE, MODULE_SPEAKING), store)


@router.get("/attempts/{attempt_id}")
def get_attempt(attempt_id: str, store: Store) -> SpeakingAttemptOut:
    return _view(_load_speaking(store, attempt_id), store)


@router.post("/attempts/{attempt_id}/start")
def start_attempt(attempt_id: str, store: Store) -> SpeakingAttemptOut:
    _load_speaking(store, attempt_id)
    return _view(store.start(attempt_id), store)


@router.put("/attempts/{attempt_id}/parts/{part}/audio")
async def upload_audio(
    attempt_id: str, part: int, request: Request, store: Store, audio_dir: AudioDir,
) -> SpeakingAttemptOut:
    """Store one part's recording (the raw body); a second upload for a part replaces it."""
    attempt = _load_speaking(store, attempt_id)
    if part not in (1, 2, 3):
        raise HTTPException(404, "no such part")
    if attempt.status != "in_progress":
        raise HTTPException(409, f"attempt is {attempt.status}; recordings are taken while it runs")
    mime = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if mime not in EXTENSIONS:
        raise HTTPException(415, f"unsupported audio type {mime or '(none)'}")
    try:
        duration = float(request.headers.get("x-duration-s", ""))
    except ValueError:
        raise HTTPException(422, "X-Duration-S (seconds) is required") from None
    limit = speaking.get_set(attempt.prompt_id).max_clip_s[part - 1]
    if not 0 < duration <= limit:
        raise HTTPException(422, f"part {part} must be at most {limit:.0f}s")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > speaking.MAX_CLIP_BYTES:
        raise HTTPException(413, "recording is too large")
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > speaking.MAX_CLIP_BYTES:
            raise HTTPException(413, "recording is too large")
    if not data:
        raise HTTPException(422, "recording is empty")

    folder = _dir(audio_dir, attempt_id)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"part{part}.{EXTENSIONS[mime]}"
    uploaded = _parts_of(attempt)
    staged = folder / f"{name}.upload"
    staged.write_bytes(bytes(data))
    uploaded[part] = UploadedPart(
        part=part, mime=mime, bytes=len(data), duration_s=round(duration, 1), file=name,
    )
    try:
        saved = store.save_draft(attempt_id, _encode(uploaded))
    except AttemptStateError as exc:
        staged.unlink(missing_ok=True)
        raise HTTPException(409, str(exc)) from None
    staged.replace(folder / name)
    for old in folder.glob(f"part{part}.*"):
        if old.name != name:
            old.unlink()
    return _view(saved, store)


@router.post("/attempts/{attempt_id}/submit")
def submit_attempt(
    attempt_id: str, background: BackgroundTasks, store: Store, assessor: Assessor,
    ledger: SpendLedger, audio_dir: AudioDir,
) -> SpeakingAttemptOut:
    attempt = _load_speaking(store, attempt_id)
    if attempt.status == "in_progress":
        missing = [n for n in (1, 2, 3) if n not in _parts_of(attempt)]
        if missing:
            raise HTTPException(
                409, "record every part before submitting; missing part "
                + ", ".join(str(n) for n in missing),
            )
    try:
        attempt, submitted = store.submit(attempt_id, attempt.draft_text)
    except AttemptStateError as exc:
        raise HTTPException(409, str(exc)) from None
    if submitted:  # only the call that made the transition marks
        attempt, _ = store.begin_grading(attempt_id)
        background.add_task(grade_speaking_attempt, attempt_id, store, assessor, audio_dir, ledger)
    return _view(attempt, store)


@router.post("/attempts/{attempt_id}/regrade")
def regrade_attempt(
    attempt_id: str, background: BackgroundTasks, store: Store, assessor: Assessor,
    ledger: SpendLedger, audio_dir: AudioDir,
) -> SpeakingAttemptOut:
    attempt = _load_speaking(store, attempt_id)
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
    background.add_task(grade_speaking_attempt, attempt_id, store, assessor, audio_dir, ledger)
    return _view(attempt, store)


@router.get("/attempts/{attempt_id}/report")
def get_report(attempt_id: str, store: Store) -> SpeakingReport:
    attempt = _load_speaking(store, attempt_id)
    report = _report_of(attempt)
    if report is None:
        raise HTTPException(409, f"attempt is {attempt.status}; no report yet")
    return report


__all__ = ["DISCLAIMER", "router"]
