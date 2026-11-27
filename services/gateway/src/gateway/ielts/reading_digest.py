"""
The IELTS Reading coach's view of a marked mock: bounded text, built in code.

Same two rules as the Writing digest (see `digest.py`): it is untrusted data between
content-derived markers (the passage-derived evidence and the learner's own answers may contain
instructions), and it is bounded. It carries no passage, only each missed question with the
learner's answer, the key and the sentence the key rests on, so the coach can answer "why did I
miss the matching headings?" from the evidence and cannot reproduce the paper.
"""
from __future__ import annotations

from services.gateway.src.gateway.ielts.digest import line, wrap
from services.gateway.src.gateway.ielts.models import QuestionResult, ReadingReport
from shared.src.shared.wire import MAX_REPORT_DIGEST_CHARS

MAX_DIGEST_BODY_CHARS = 6_000
assert MAX_DIGEST_BODY_CHARS + 200 <= MAX_REPORT_DIGEST_CHARS

DIGEST_TITLE = "Marked IELTS Reading mock result"

# (prompt limit, evidence limit): from fullest to leanest; the first that fits is used.
_TIERS: tuple[tuple[int, int], ...] = ((200, 300), (120, 200), (60, 120), (0, 80))


def _missed_lines(q: QuestionResult, prompt_limit: int, evidence_limit: int) -> list[str]:
    where = f"{q.passage}" + (f", paragraph {q.paragraph}" if q.paragraph else "")
    head = f"- Q{q.number} [{q.task_label}] ({where})"
    if prompt_limit:
        head += f": {line(q.prompt, prompt_limit)}"
    return [
        head,
        f"    the learner answered: {line(q.answer_label, 120)}",
        f"    the key: {line(q.key_label, 120)}",
        f'    evidence sentence: "{line(q.evidence, evidence_limit)}"',
    ]


def _body(report: ReadingReport, prompt_limit: int, evidence_limit: int) -> str:
    t = report.timing
    lines = [
        f"Paper: a partial IELTS General Training Reading mock, {report.total} questions on "
        f"{len({q.passage for q in report.questions})} short passages (not a full 40-question "
        "paper). The answers were marked by program code against author-written keys that no "
        "human has checked, so a key can be arguable.",
        f"Score: {report.raw} of {report.total} correct, scaled to {report.scaled_40} of 40. "
        f"Mock band (an estimate from a conversion table, not an official result): "
        f"{report.band:.1f}.",
        f"Time used: {t.used_s // 60} of {t.allowed_s // 60} minutes"
        + (", submitted late." if t.late else "."),
        "",
        "By question type (correct of total):",
    ]
    lines += [f"- {s.label}: {s.correct} of {s.total}" for s in report.by_type]
    missed = [q for q in report.questions if not q.correct]
    if not missed:
        lines += ["", "Every question was answered correctly."]
        return "\n".join(lines)
    lines += ["", f"Missed questions ({len(missed)}), with the evidence for each key:"]
    for q in missed:
        lines += _missed_lines(q, prompt_limit, evidence_limit)
    return "\n".join(lines)


def build_digest(report: ReadingReport) -> str:
    """The coach's digest of a marked Reading mock, between content-derived markers."""
    body = ""
    for prompt_limit, evidence_limit in _TIERS:
        body = _body(report, prompt_limit, evidence_limit)
        if len(body) <= MAX_DIGEST_BODY_CHARS:
            break
    else:
        body = body[:MAX_DIGEST_BODY_CHARS]
    return wrap(DIGEST_TITLE, body)
