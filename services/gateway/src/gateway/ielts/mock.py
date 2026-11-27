"""
The full IELTS mock: an ordered list of modules and one combined result, worked out in code.

A mock is a parent record (`ielts_store.Mock`) that links the attempts of its modules; the
modules are the same Reading and Writing attempts the standalone pages use. Nothing here is
stored: the status and band of each module are read from its attempt every time, so the mock can
never disagree with the module page.

Only Reading and Writing can be sat so far. Listening and Speaking are listed as "not assessed"
and the overall band is marked partial for as long as any skill is missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from services.gateway.src.gateway.ielts.models import DISCLAIMER, ReadingReport, WritingModuleReport
from services.gateway.src.gateway.ielts_store import (
    MODULE_READING,
    MODULE_WRITING,
    Attempt,
    AttemptNotFoundError,
    AttemptStore,
    Mock,
)
from shared.src.shared.stats import round_half_band

# IELTS: "The Overall Band Score is the average of the four test components (Listening, Reading,
# Writing and Speaking), rounded to the nearest whole or half band." A mean ending in .25 rounds
# up to the next half band and one ending in .75 up to the next whole band, which is what
# `round_half_band` does.
SCORING_SOURCE_URL = "https://ielts.org/take-a-test/your-results/ielts-scoring-in-detail"

ModuleKey = Literal["listening", "reading", "writing", "speaking"]
ModuleState = Literal[
    "not_assessed",  # not part of this mock yet (Listening, Speaking)
    "not_started",  # no attempt yet
    "in_progress",  # created or sitting
    "marking",
    "graded",
    "partial",  # Writing, when only one of the two tasks could be marked: no band
    "ungraded",  # marking failed
]

# The order the mock is sat in. `sittable` is False for a module that has no test yet.
MODULES: tuple[tuple[ModuleKey, str, bool], ...] = (
    ("listening", "Listening", False),
    ("reading", "Reading", True),
    ("writing", "Writing", True),
    ("speaking", "Speaking", False),
)
SITTABLE_ORDER: tuple[ModuleKey, ...] = tuple(key for key, _, sittable in MODULES if sittable)
ATTEMPT_MODULE = {"reading": MODULE_READING, "writing": MODULE_WRITING}
LABELS: dict[str, str] = {key: label for key, label, _ in MODULES}
# An attempt in one of these has been sat: the next module may start.
PAST_SITTING = ("submitted", "grading", "graded", "ungraded")


class ModuleStatus(BaseModel):
    key: ModuleKey
    label: str
    state: ModuleState
    band: float | None = None
    attempt_id: str | None = None
    sittable: bool
    retryable: bool = False  # marking failed and the module's retry route will take it again
    detail: str = ""


class NextModule(BaseModel):
    key: ModuleKey
    label: str
    reason: str


class MockReport(BaseModel):
    id: str
    created_at: float
    modules: list[ModuleStatus]
    overall: float | None  # the mean of the assessed skill bands, rounded; None before any band
    partial: bool  # true while any skill is not assessed
    overall_note: str
    overall_source_url: str = SCORING_SOURCE_URL
    next_module: NextModule | None = None
    disclaimer: str = DISCLAIMER


@dataclass(frozen=True)
class LoadedMock:
    """A mock with its module attempts and their reports, read once."""

    mock: Mock
    attempts: dict[str, Attempt | None]
    reading: ReadingReport | None
    writing: WritingModuleReport | None


def load_mock(store: AttemptStore, mock: Mock) -> LoadedMock:
    attempts: dict[str, Attempt | None] = {}
    for key, attempt_id in mock.attempt_ids.items():
        try:
            attempts[key] = store.get(attempt_id) if attempt_id else None
        except AttemptNotFoundError:
            attempts[key] = None
    reading, writing = attempts.get("reading"), attempts.get("writing")
    return LoadedMock(
        mock=mock,
        attempts=attempts,
        reading=(
            ReadingReport.model_validate_json(reading.report_json)
            if reading and reading.report_json else None
        ),
        writing=(
            WritingModuleReport.model_validate_json(writing.report_json)
            if writing and writing.report_json else None
        ),
    )


def _state_of(attempt: Attempt | None, band: float | None, has_report: bool) -> ModuleState:
    if attempt is None:
        return "not_started"
    if attempt.status in ("created", "in_progress"):
        return "in_progress"
    if attempt.status in ("submitted", "grading"):
        return "marking"
    if attempt.status == "ungraded":
        return "ungraded"
    if band is not None:
        return "graded"
    return "partial" if has_report else "ungraded"


def _detail(key: str, state: ModuleState, loaded: LoadedMock, retryable: bool) -> str:
    if state == "not_assessed":
        return "Not assessed: this module is not part of the mock yet."
    if state == "graded" and key == "reading" and loaded.reading:
        return f"{loaded.reading.raw} of {loaded.reading.total} correct"
    if state == "graded" and key == "writing":
        return "Task 1 letter and Task 2 essay"
    if state == "partial":
        return "Only one task could be marked, so there is no Writing band."
    if state == "ungraded":
        return "Marking did not finish." + (" Use Retry marking to try again." if retryable else "")
    return ""


def module_statuses(loaded: LoadedMock) -> list[ModuleStatus]:
    out: list[ModuleStatus] = []
    for key, label, sittable in MODULES:
        if not sittable:
            out.append(ModuleStatus(
                key=key, label=label, state="not_assessed", sittable=False,
                detail=_detail(key, "not_assessed", loaded, False),
            ))
            continue
        attempt = loaded.attempts.get(key)
        band = (
            loaded.reading.band if key == "reading" and loaded.reading
            else loaded.writing.writing_band if key == "writing" and loaded.writing
            else None
        )
        has_report = loaded.reading is not None if key == "reading" else loaded.writing is not None
        state = _state_of(attempt, band, has_report)
        retryable = bool(key == "writing" and state == "ungraded" and attempt and attempt.retryable)
        out.append(ModuleStatus(
            key=key, label=label, state=state, band=band if state == "graded" else None,
            attempt_id=attempt.id if attempt else None, sittable=True, retryable=retryable,
            detail=_detail(key, state, loaded, retryable),
        ))
    return out


def overall_band(bands: list[float]) -> float | None:
    """The mean of the assessed skill bands, rounded the IELTS way; None when there are none."""
    return round_half_band(sum(bands) / len(bands)) if bands else None


def _names(labels: list[str]) -> str:
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


def _missing_clauses(modules: list[ModuleStatus]) -> list[str]:
    """Why each skill without a band has none, grouped: 'Listening and Speaking not assessed'."""
    reasons: tuple[tuple[tuple[ModuleState, ...], str], ...] = (
        (("not_assessed",), "not assessed"),
        (("not_started", "in_progress"), "not taken yet"),
        (("marking",), "being marked"),
        (("ungraded", "partial"), "without a band because marking did not finish"),
    )
    clauses = []
    for states, text in reasons:
        labels = [m.label for m in modules if m.band is None and m.state in states]
        if labels:
            clauses.append(f"{_names(labels)} {text}")
    return clauses


def _overall_note(modules: list[ModuleStatus], overall: float | None) -> str:
    missing = [m for m in modules if m.band is None]
    if overall is None:
        return "No skill has a band yet, so there is no overall band."
    base = (
        "Overall band = the mean of the assessed skill bands, rounded to the nearest half band "
        "(IELTS rounds a quarter up). It is a mock estimate, not an official result."
    )
    if not missing:
        return base
    return (
        f"Partial overall: {'; '.join(_missing_clauses(modules))}, so this is the mean of "
        f"{len(modules) - len(missing)} of {len(modules)} skills. " + base
    )


def next_module(modules: list[ModuleStatus]) -> NextModule | None:
    """
    What to do next, decided in code so the coach and the page agree: a module not finished yet
    (in the mock's order), then a module whose marking failed, then the lowest assessed skill.
    Nothing while a module is being marked.
    """
    by_key = {m.key: m for m in modules}
    for key in SITTABLE_ORDER:
        if by_key[key].state in ("not_started", "in_progress"):
            m = by_key[key]
            verb = "Finish" if m.state == "in_progress" else "Sit"
            return NextModule(
                key=key, label=m.label,
                reason=f"{verb} {m.label} next: it is the next module in the mock.",
            )
    if any(m.state == "marking" for m in modules):
        return None
    for key in SITTABLE_ORDER:
        m = by_key[key]
        if m.state in ("ungraded", "partial"):
            return NextModule(
                key=key, label=m.label,
                reason=(
                    f"{m.label} has no band yet because its marking did not finish"
                    + (": use Retry marking." if m.retryable else ".")
                ),
            )
    graded = [m for m in modules if m.band is not None]
    if not graded:
        return None
    lowest = min(graded, key=lambda m: m.band if m.band is not None else 0.0)
    return NextModule(
        key=lowest.key, label=lowest.label,
        reason=(
            f"{lowest.label} is your lowest assessed skill (band {lowest.band}): "
            "practise it next."
        ),
    )


def build_report(loaded: LoadedMock) -> MockReport:
    modules = module_statuses(loaded)
    bands = [m.band for m in modules if m.band is not None]
    overall = overall_band(bands)
    return MockReport(
        id=loaded.mock.id, created_at=loaded.mock.created_at, modules=modules, overall=overall,
        partial=len(bands) < len(modules), overall_note=_overall_note(modules, overall),
        next_module=next_module(modules),
    )


def can_start(loaded: LoadedMock, key: str) -> str | None:
    """None when the module may start; otherwise why not. Modules are sat in order."""
    if key not in SITTABLE_ORDER:
        return f"{LABELS.get(key, key)} is not part of the mock yet."
    if loaded.attempts.get(key) is not None:
        return f"{LABELS[key]} already has an attempt in this mock."
    for earlier in SITTABLE_ORDER[: SITTABLE_ORDER.index(key)]:
        attempt = loaded.attempts.get(earlier)
        if attempt is None or attempt.status not in PAST_SITTING:
            return f"Finish {LABELS[earlier]} first: the mock is sat in order."
    return None
