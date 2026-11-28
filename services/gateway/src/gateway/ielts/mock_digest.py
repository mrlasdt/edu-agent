"""
The mock coach's view of a full mock: bounded text, built in code.

Like the other digests it is untrusted data between content-derived markers (the Writing task
prompts and the model's priorities can carry instructions), and it never carries a passage or an
essay. It gives the per-skill bands, the partial overall, the module the code recommends next, and
for each sat module the few facts a coach needs to say why: Listening's and Reading's score by
question type and Writing's criterion bands and top priorities.
"""
from __future__ import annotations

from services.gateway.src.gateway.ielts.digest import line, wrap
from services.gateway.src.gateway.ielts.mock import LoadedMock, MockReport, build_report
from services.gateway.src.gateway.ielts.models import (
    ListeningReport,
    ReadingReport,
    Report,
    SpeakingReport,
    WritingModuleReport,
)

DIGEST_TITLE = "IELTS full mock result"
MAX_BODY_CHARS = 5_000


def _listening_lines(report: ListeningReport) -> list[str]:
    weakest = sorted(report.by_type, key=lambda t: t.correct / max(t.total, 1))
    return [
        f"Listening detail: {report.raw} of {report.total} correct, band {report.band}.",
        "Listening by question type, weakest first: "
        + line("; ".join(f"{t.label} {t.correct}/{t.total}" for t in weakest), 600),
    ]


def _reading_lines(report: ReadingReport) -> list[str]:
    weakest = sorted(report.by_type, key=lambda t: t.correct / max(t.total, 1))
    lines = [
        f"Reading detail: {report.raw} of {report.total} correct, band {report.band}.",
        "Reading by question type, weakest first: "
        + line("; ".join(f"{t.label} {t.correct}/{t.total}" for t in weakest), 600),
    ]
    return lines


def _task_lines(label: str, task: Report) -> list[str]:
    if task.status != "graded" or task.estimated_band is None:
        return [f"Writing {label}: not marked."]
    criteria = ", ".join(f"{c.id} {c.band:.1f}" for c in task.criteria)
    lines = [f"Writing {label}: band {task.estimated_band.overall:.1f} ({criteria})."]
    for p in task.top_priorities[:2]:
        lines.append(f"  priority {p.priority} [{p.criterion}]: {line(p.do, 220)}")
    return lines


def _writing_lines(report: WritingModuleReport) -> list[str]:
    return [*_task_lines("Task 1 letter", report.t1), *_task_lines("Task 2 essay", report.t2)]


def _speaking_lines(report: SpeakingReport) -> list[str]:
    if report.band is None:
        return ["Speaking: too little speech was heard to mark."]
    criteria = ", ".join(f"{c.id} {c.band:.1f}" for c in report.criteria if c.band is not None)
    lines = [
        f"Speaking detail: partial band {report.band} ({criteria}); "
        + ", ".join(f"{c} not assessed" for c in report.not_assessed) + ".",
    ]
    for p in report.priorities[:2]:
        lines.append(f"  priority {p.priority} [{p.criterion}]: {line(p.do, 220)}")
    return lines


def build_digest(loaded: LoadedMock, report: MockReport | None = None) -> str:
    """The digest of a mock; "" when no module has a band (there is nothing to coach from)."""
    report = report or build_report(loaded)
    if report.overall is None:
        return ""
    lines = ["Skill bands (mock estimates, not official results):"]
    for m in report.modules:
        lines.append(
            f"- {m.label}: band {m.band}"
            + (" (quick sample, not a real mock score)" if m.sample else "")
            if m.band is not None
            else f"- {m.label}: not assessed" if m.state == "not_assessed"
            else f"- {m.label}: no band yet (too little speech was heard)"
            if m.key == "speaking" and m.state == "partial"
            else f"- {m.label}: no band yet ({m.state.replace('_', ' ')})"
        )
    partial = " (PARTIAL: a skill above has no band, or its band leaves out a criterion)"
    lines.append(f"Overall band: {report.overall}{partial if report.partial else ''}.")
    lines.append(line(report.overall_note, 500))
    if report.next_module:
        lines.append(
            f"Recommended next module, worked out by the program: {report.next_module.label}. "
            f"{line(report.next_module.reason, 300)}"
        )
    if loaded.listening:
        lines += ["", *_listening_lines(loaded.listening)]
    if loaded.reading:
        lines += ["", *_reading_lines(loaded.reading)]
    if loaded.writing:
        lines += ["", *_writing_lines(loaded.writing)]
    if loaded.speaking:
        lines += ["", *_speaking_lines(loaded.speaking)]
    return wrap(DIGEST_TITLE, "\n".join(lines)[:MAX_BODY_CHARS])
