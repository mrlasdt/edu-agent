"""
What the IELTS Writing coach in /chat needs from the attempt store: the learner's graded report,
as the digest the agent service cannot build itself (it has no access to the store).

A `/chat` turn for ("ielts", "writing") names an attempt (`attempt_id`, e.g. from the report
page's "Discuss this report" link) or leaves it out for the latest graded one. Only the digest
leaves the gateway: bands, the report's own words and its verified quotes, never the whole essay.
"""
from __future__ import annotations

from dataclasses import dataclass

from services.gateway.src.gateway.ielts.digest import build_digest
from services.gateway.src.gateway.ielts.models import Report
from services.gateway.src.gateway.ielts_store import (
    MODULE_WRITING_T2,
    Attempt,
    AttemptNotFoundError,
    AttemptStore,
)

COACH_PAIR = ("ielts", "writing")

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


@dataclass(frozen=True)
class CoachReport:
    attempt_id: str
    digest: str


def _graded(attempt: Attempt) -> bool:
    return (
        attempt.module == MODULE_WRITING_T2
        and attempt.status == "graded"
        and attempt.report_json is not None
    )


def find_report(store: AttemptStore, attempt_id: str | None) -> CoachReport | None:
    """The digest of the attempt named, or of the latest graded one; None when there is none."""
    found: Attempt | None
    if attempt_id:
        try:
            found = store.get(attempt_id)
        except AttemptNotFoundError:
            return None
    else:
        found = next((a for a in store.recent() if _graded(a)), None)
    if found is None or not _graded(found) or found.report_json is None:
        return None
    report = Report.model_validate_json(found.report_json)
    digest = build_digest(report, found.essay, found.prompt_text)
    return CoachReport(found.id, digest) if digest else None
