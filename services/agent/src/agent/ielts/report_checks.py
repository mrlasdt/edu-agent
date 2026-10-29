"""Mechanical checks on a finished report: no model needed, so they run in CI and on live output."""
from __future__ import annotations

from services.agent.src.agent.ielts.checks import paragraph_spans
from services.agent.src.agent.ielts.feedback import (
    MAX_QUOTE_WORDS,
    is_bare_number,
    within_paragraph,
)
from services.agent.src.agent.ielts.models import DISCLAIMER, Report
from services.agent.src.agent.ielts.scoring import CRITERIA, round_overall


def _in_paragraph(paragraphs: list[tuple[int, int]], number: int, start: int, end: int) -> bool:
    return 1 <= number <= len(paragraphs) and within_paragraph(
        [paragraphs[number - 1]], start, end
    )


def check_report(report: Report, essay: str) -> list[str]:
    """Violations of the report contract; an empty list means the report is sound."""
    if report.status == "ungraded":
        missing = [] if report.ungraded_reason else ["ungraded report has no reason"]
        if report.estimated_band or report.criteria:
            missing.append("ungraded report carries a band")
        return missing

    problems: list[str] = []
    if report.disclaimer != DISCLAIMER:
        problems.append("disclaimer altered")
    band = report.estimated_band
    if band is None:
        return [*problems, "graded report has no mock score"]
    bands = {c.id: c.band for c in report.criteria}
    if set(bands) != set(CRITERIA):
        return [*problems, f"criteria are {sorted(bands)}"]
    if band.overall != round_overall(bands):
        problems.append("overall is not the rounded mean of the four criteria")
    if not band.range[0] <= band.overall <= band.range[1]:
        problems.append("range does not bracket the overall band")
    if band.overall >= 9.0:
        problems.append("estimate of band 9")
    if band.calibration.status == "uncalibrated" and "ncalibrated" not in band.calibration.label:
        problems.append("uncalibrated estimate is not labelled as such")

    paragraphs = paragraph_spans(essay)
    for crit in report.criteria:
        if crit.verdict and is_bare_number(crit.verdict):
            problems.append(f"{crit.id} verdict is a bare number")
        for ev in crit.evidence:
            if not (0 <= ev.start < ev.end <= len(essay)) or essay[ev.start:ev.end] != ev.quote:
                problems.append(f"{crit.id} evidence quote is not in the essay: {ev.quote[:30]!r}")
            elif len(ev.quote.split()) > MAX_QUOTE_WORDS:
                problems.append(f"{crit.id} evidence quote is longer than {MAX_QUOTE_WORDS} words")
            elif not _in_paragraph(paragraphs, ev.para, ev.start, ev.end):
                problems.append(f"{crit.id} evidence quote is not within its paragraph")
        if report.feedback_status == "complete" and not crit.evidence:
            problems.append(f"{crit.id} has no evidence")
    for span in report.error_spans:
        if not 0 <= span.start < span.end <= len(essay) or essay[span.start:span.end] != span.quote:
            problems.append(f"error span is not in the essay: {span.quote[:30]!r}")
        elif not within_paragraph(paragraphs, span.start, span.end):
            problems.append(f"error span crosses a paragraph break: {span.quote[:30]!r}")

    if report.top_priorities:
        lowest = sorted(bands.values())[:3]
        picked = sorted(bands[p.criterion] for p in report.top_priorities)
        if picked != lowest[: len(picked)]:
            problems.append("priorities do not map to the lowest-banded criteria")
    return problems
