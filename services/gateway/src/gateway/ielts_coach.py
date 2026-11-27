"""
What the IELTS coaches in /chat need from the attempt store: the learner's graded report, as the
digest the agent service cannot build itself (it has no access to the store).

A `/chat` turn for ("ielts", "writing") or ("ielts", "reading") names an attempt (`attempt_id`,
e.g. from the report page's "Discuss this report" link) or leaves it out for the latest graded one
of that section. Only the digest leaves the gateway: for Writing the bands, the report's own
words and its verified quotes, never the whole essay; for Reading the score, the missed questions
and the evidence sentence for each key, never a passage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from services.gateway.src.gateway.ielts.digest import build_digest, build_module_digest
from services.gateway.src.gateway.ielts.mock import build_report, load_mock
from services.gateway.src.gateway.ielts.mock_digest import build_digest as build_mock_digest
from services.gateway.src.gateway.ielts.models import ReadingReport, Report, WritingModuleReport
from services.gateway.src.gateway.ielts.reading_digest import (
    build_digest as build_reading_digest,
)
from services.gateway.src.gateway.ielts_store import (
    MODULE_READING,
    MODULE_WRITING,
    MODULE_WRITING_T2,
    Attempt,
    AttemptNotFoundError,
    AttemptStore,
    MockNotFoundError,
)

# The chat pairs this module coaches, and the attempt module each reads its report from. The
# full mock is not an attempt: its "module" is the mocks table, so `find_report` handles it apart.
MOCK_COACH = "ielts_mock"
COACH_MODULES: dict[tuple[str, str], str] = {
    ("ielts", "writing"): MODULE_WRITING_T2,
    ("ielts", "reading"): MODULE_READING,
    ("ielts", "mock"): MOCK_COACH,
}
# The Writing coach also reads a graded Writing module (letter and essay), alongside the
# standalone Task 2 mock.
EXTRA_COACH_MODULES: dict[str, tuple[str, ...]] = {"writing": (MODULE_WRITING,)}

NO_REPORT_NOTICE = (
    "I coach from your graded IELTS Writing mock report, and I can't find one yet. Take a "
    "Writing mock at /ielts-mock; once it is marked, come back and ask me about it."
)
UNKNOWN_ATTEMPT_NOTICE = (
    "I can't find a graded report for that attempt. Open /ielts-mock, pick a marked attempt, "
    "and use its \"Discuss this report\" link."
)
SOLVE_MODE_NOTICE = (
    "The IELTS Writing coach has one mode: it coaches you from your report and never writes the "
    "essay for you. Ask it about a criterion, or what to fix first."
)
READING_NO_REPORT_NOTICE = (
    "I coach from your marked IELTS Reading mock, and I can't find one yet. Sit a Reading mock "
    "at /ielts-mock; once you have submitted it, come back and ask me about it."
)
READING_UNKNOWN_ATTEMPT_NOTICE = (
    "I can't find a marked Reading mock for that attempt. Open /ielts-mock, pick a finished "
    "Reading attempt, and use its \"Discuss this result\" link."
)
READING_SOLVE_MODE_NOTICE = (
    "The IELTS Reading coach has one mode: it coaches you from your marked result and your "
    "evidence sentences. Ask it about a question type you missed."
)
MOCK_NO_REPORT_NOTICE = (
    "I coach from your full IELTS mock, and none has a band yet. Start a full mock at "
    "/ielts-mock?module=mock, sit Reading, and once it is marked come back and ask me what to "
    "practise next."
)
MOCK_UNKNOWN_ATTEMPT_NOTICE = (
    "I can't find a full mock with a band for that link. Open /ielts-mock?module=mock, pick a "
    "mock, and use its \"Discuss with the coach\" link."
)
MOCK_SOLVE_MODE_NOTICE = (
    "The full-mock coach has one mode: it reads your mock result and says which skill to "
    "practise next. Ask it why, or how to practise."
)


@dataclass(frozen=True)
class Notices:
    no_report: str
    unknown_attempt: str
    solve_mode: str


def notices_for(section: str) -> Notices:
    if section == "mock":
        return Notices(MOCK_NO_REPORT_NOTICE, MOCK_UNKNOWN_ATTEMPT_NOTICE, MOCK_SOLVE_MODE_NOTICE)
    if section == "reading":
        return Notices(
            READING_NO_REPORT_NOTICE, READING_UNKNOWN_ATTEMPT_NOTICE, READING_SOLVE_MODE_NOTICE
        )
    return Notices(NO_REPORT_NOTICE, UNKNOWN_ATTEMPT_NOTICE, SOLVE_MODE_NOTICE)


@dataclass(frozen=True)
class CoachReport:
    attempt_id: str
    digest: str


def _graded(attempt: Attempt, modules: tuple[str, ...]) -> bool:
    return (
        attempt.module in modules
        and attempt.status == "graded"
        and attempt.report_json is not None
    )


def _json_object(raw: str) -> dict[str, object]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _module_digest(attempt: Attempt, report_json: str) -> str:
    """The Writing module: its texts and prompts are JSON objects keyed t1 and t2."""
    texts = {k: str(v) for k, v in _json_object(attempt.essay).items()}
    prompts = {
        k: str(v.get("text", "")) for k, v in _json_object(attempt.prompt_text).items()
        if isinstance(v, dict)
    }
    return build_module_digest(WritingModuleReport.model_validate_json(report_json), texts, prompts)


def _digest(attempt: Attempt, report_json: str) -> str:
    if attempt.module == MODULE_READING:
        return build_reading_digest(ReadingReport.model_validate_json(report_json))
    if attempt.module == MODULE_WRITING:
        return _module_digest(attempt, report_json)
    report = Report.model_validate_json(report_json)
    return build_digest(report, attempt.essay, attempt.prompt_text)


def find_report(
    store: AttemptStore, attempt_id: str | None, section: str = "writing"
) -> CoachReport | None:
    """
    The digest of the attempt named, or of the latest graded one of this section; None when there
    is none. An attempt of the other section is "not found", not a digest of the wrong module.
    """
    if section == "mock":
        return _find_mock(store, attempt_id)
    modules = (COACH_MODULES[("ielts", section)], *EXTRA_COACH_MODULES.get(section, ()))
    found: Attempt | None
    if attempt_id:
        try:
            found = store.get(attempt_id)
        except AttemptNotFoundError:
            return None
    else:
        newest_first = sorted(
            (a for m in modules for a in store.recent(module=m)),
            key=lambda a: a.submitted_at or 0.0, reverse=True,
        )
        found = next((a for a in newest_first if _graded(a, modules)), None)
    if found is None or not _graded(found, modules) or found.report_json is None:
        return None
    digest = _digest(found, found.report_json)
    return CoachReport(found.id, digest) if digest else None


def _find_mock(store: AttemptStore, mock_id: str | None) -> CoachReport | None:
    """The digest of the full mock named, or of the newest one that has a band."""
    if mock_id:
        try:
            candidates = [store.get_mock(mock_id)]
        except MockNotFoundError:
            return None
    else:
        candidates = store.recent_mocks()
    for mock in candidates:
        loaded = load_mock(store, mock)
        digest = build_mock_digest(loaded, build_report(loaded))
        if digest:
            return CoachReport(mock.id, digest)
    return None
