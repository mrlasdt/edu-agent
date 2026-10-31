"""
Examiner-style feedback for a graded essay, with every quote checked against the essay.

The model proposes quotes; the code locates each one in the submission (whitespace tolerant)
and keeps only the spans it can find, so the report never shows words the candidate did not
write. Offsets, paragraph numbers and the error table are computed here, not by the model.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from services.agent.src.agent.ielts import llm
from services.agent.src.agent.ielts.checks import paragraph_spans
from services.agent.src.agent.ielts.models import (
    Action,
    ErrorSpan,
    Evidence,
    ParagraphRole,
)
from services.agent.src.agent.ielts.scoring import (
    CRITERIA,
    GradeParseError,
    extract_json,
)
from services.agent.src.agent.ielts.spelling_variants import VARIANT_PAIRS

FEEDBACK_MAX_TOKENS = 6000  # a cap, not a target
MAX_QUOTE_WORDS = 15
MAX_LOCATE_WORDS = 60  # longest quote the code will look for before shortening it
MAX_ERRORS = 12
MAX_ERROR_WORDS = 10  # a whole clause is not a located error
ATTEMPTS = 2
ERROR_TYPES = (
    "agreement", "tense", "article", "preposition", "punctuation", "collocation", "spelling",
    "linking", "word_form", "other",
)
_EDGE = re.compile(r"^[\s\"'“”‘’.…]+|[\s\"'“”‘’.…]+$")
_BARE_NUMBER = re.compile(r"(?:band\s*)?[\d.,/+\-–\s]+", re.I)
# A note that calls the text informal or a style choice is not naming an error.
_NOT_AN_ERROR = re.compile(
    r"\b(?:informal|colloquial|conversational|register|stylistic|style)\b", re.I,
)
_STYLE_TYPES = frozenset({"word_form", "collocation", "other"})


@dataclass
class CriterionFeedback:
    verdict: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    holding_back: str = ""
    to_reach_next_band: str = ""
    actions: list[Action] = field(default_factory=list)
    paragraph_map: list[ParagraphRole] = field(default_factory=list)


@dataclass
class Feedback:
    criteria: dict[str, CriterionFeedback]
    errors: list[ErrorSpan]
    checks: dict[str, bool | None]
    unverifiable: int  # quotes not found in one paragraph of the essay, bare-number verdicts


def locate_quote(text: str, quote: str, taken: list[tuple[int, int]] | None = None) -> (
    tuple[int, int] | None
):
    """
    Span of `quote` in `text`, tolerant of whitespace and surrounding quote marks. Returns None
    when the words are not in one paragraph of the text. `taken` spans are skipped so repeated
    quotes map to different occurrences where they exist. The whole quote is located, however
    long (up to MAX_LOCATE_WORDS); `shorten_span` brings it within the display limit.
    """
    tokens = _EDGE.sub("", quote).split()[:MAX_LOCATE_WORDS]
    if not tokens:
        return None
    pattern = re.compile(r"\s+".join(re.escape(t) for t in tokens))
    paragraphs = paragraph_spans(text)
    first: tuple[int, int] | None = None
    for m in pattern.finditer(text):
        span = (m.start(), m.end())
        if not within_paragraph(paragraphs, *span):
            continue
        if first is None:
            first = span
        if not taken or all(span[1] <= a or span[0] >= b for a, b in taken):
            return span
    return first


def shorten_span(text: str, span: tuple[int, int]) -> tuple[tuple[int, int], bool]:
    """`span` cut to its first MAX_QUOTE_WORDS words, and whether anything was cut."""
    words = list(re.finditer(r"\S+", text[span[0]:span[1]]))
    if len(words) <= MAX_QUOTE_WORDS:
        return span, False
    return (span[0], span[0] + words[MAX_QUOTE_WORDS - 1].end()), True


def _bare_word(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower())


def is_spelling_variant(quote: str, suggestion: str) -> bool:
    """True when `suggestion` only swaps British and American spellings of the quoted words."""
    got = {_bare_word(w) for w in quote.split()}
    wanted = {_bare_word(w) for w in re.sub(r"\(.*?\)", " ", suggestion).split()} - {""}
    swapped = wanted - got
    return bool(swapped) and all(any((g, w) in VARIANT_PAIRS for g in got) for w in swapped)


def is_not_an_error(kind: str, quote: str, note: str, suggestion: str) -> bool:
    """A correct spelling variant, or a remark on style rather than on a rule that was broken."""
    if kind in _STYLE_TYPES and _NOT_AN_ERROR.search(note):
        return True
    return kind == "spelling" and is_spelling_variant(quote, suggestion)


def _changes_a_form(quote: str, suggestion: str) -> bool:
    """Whether the fix swaps a word for another form of itself (economy -> economic)."""
    old = set(re.findall(r"[a-z]+", quote.lower()))
    new = set(re.findall(r"[a-z]+", suggestion.lower()))
    gone, added = old - new, new - old
    if not gone or not added:
        return True  # an insertion or deletion: nothing here says the type is wrong
    return any(a[:4] == b[:4] for a in gone for b in added if min(len(a), len(b)) >= 4)


def within_paragraph(paragraphs: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(a <= start and end <= b for a, b in paragraphs)


def is_bare_number(verdict: str) -> bool:
    return _BARE_NUMBER.fullmatch(verdict) is not None


def paragraph_of(text: str, start: int) -> int:
    for number, (a, b) in enumerate(paragraph_spans(text), start=1):
        if a <= start < b + 1:
            return number
    return 1


def numbered_essay(essay: str) -> str:
    return "\n\n".join(
        f"Paragraph {n}:\n{essay[a:b]}" for n, (a, b) in enumerate(paragraph_spans(essay), 1)
    )


def feedback_request(
    prompt: str, essay: str, bands: dict[str, float], system: str, notes: list[str]
) -> list[dict[str, str]]:
    marker = hashlib.sha256(essay.encode()).hexdigest()[:8]
    scores = ", ".join(f"{c} {bands[c]:g}" for c in CRITERIA)
    extra = ("\nChecks run in code: " + " ".join(notes) + "\n") if notes else ""
    user = (
        f"Decided bands: {scores}.{extra}\n"
        f"Task prompt (data):\n<<<PROMPT {marker}>>>\n{prompt}\n<<<END PROMPT {marker}>>>\n\n"
        f"Candidate essay (data), split into numbered paragraphs:\n"
        f"<<<ESSAY {marker}>>>\n{numbered_essay(essay)}\n<<<END ESSAY {marker}>>>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _text(value: Any, limit: int = 500) -> str:
    return " ".join(str(value or "").split())[:limit]


def build_feedback(raw: dict[str, Any], essay: str) -> Feedback:
    """Validate a parsed model reply against the essay; unverifiable parts are dropped."""
    dropped = 0
    criteria: dict[str, CriterionFeedback] = {}
    raw_criteria = raw.get("criteria")
    if not isinstance(raw_criteria, dict):
        raise GradeParseError("feedback reply has no 'criteria' object")
    for crit in CRITERIA:
        item = raw_criteria.get(crit)
        if not isinstance(item, dict):
            criteria[crit] = CriterionFeedback()
            continue
        taken: list[tuple[int, int]] = []
        evidence: list[Evidence] = []
        for ev in item.get("evidence") or []:
            if not isinstance(ev, dict):
                continue
            span = locate_quote(essay, str(ev.get("quote", "")), taken)
            if span is None:
                dropped += 1
                continue
            taken.append(span)
            span, cut = shorten_span(essay, span)
            evidence.append(Evidence(
                quote=essay[span[0]:span[1]], start=span[0], end=span[1], cut=cut,
                para=paragraph_of(essay, span[0]),
                effect="weakens" if str(ev.get("effect")) == "weakens" else "supports",
            ))
        actions = [
            Action(do=_text(a.get("do")), why=_text(a.get("why")))
            for a in (item.get("actions") or [])[:3]
            if isinstance(a, dict) and _text(a.get("do"))
        ]
        paragraph_map = [
            ParagraphRole(
                para=int(p["para"]), job=_text(p.get("job"), 200),
                one_central_idea=p.get("one_central_idea")
                if isinstance(p.get("one_central_idea"), bool) else None,
            )
            for p in (item.get("paragraph_map") or [])
            if isinstance(p, dict) and str(p.get("para", "")).isdigit()
        ] if crit == "CC" else []
        verdict = _text(item.get("verdict"))
        if verdict and is_bare_number(verdict):
            verdict = ""
            dropped += 1
        criteria[crit] = CriterionFeedback(
            verdict=verdict, evidence=evidence,
            holding_back=_text(item.get("holding_back")),
            to_reach_next_band=_text(item.get("to_reach_next_band")),
            actions=actions, paragraph_map=paragraph_map,
        )

    errors: list[ErrorSpan] = []
    taken_errors: list[tuple[int, int]] = []
    for err in (raw.get("errors") or [])[:MAX_ERRORS * 2]:
        if not isinstance(err, dict) or len(errors) >= MAX_ERRORS:
            continue
        span = locate_quote(essay, str(err.get("quote", "")), taken_errors)
        if span is None:
            dropped += 1
            continue
        if len(essay[span[0]:span[1]].split()) > MAX_ERROR_WORDS:
            continue
        quote = essay[span[0]:span[1]]
        kind = str(err.get("type", "other")).strip().lower()
        kind = kind if kind in ERROR_TYPES else "other"
        note, suggestion = _text(err.get("note"), 300), _text(err.get("suggestion"), 300)
        if is_not_an_error(kind, quote, note, suggestion):
            continue
        if kind == "word_form" and not _changes_a_form(quote, suggestion):
            kind = "other"  # a different word is a word-choice fix, not a change of form
        taken_errors.append(span)
        errors.append(ErrorSpan(
            start=span[0], end=span[1], quote=quote, type=kind, note=note,
            suggestion=suggestion,
        ))
    errors.sort(key=lambda e: e.start)

    raw_checks: dict[str, Any] = {}
    if isinstance(raw.get("checks"), dict):
        raw_checks = raw["checks"]
    checks = {
        key: raw_checks[key] if isinstance(raw_checks.get(key), bool) else None
        for key in ("addresses_all_parts", "position_in_introduction", "position_in_conclusion")
    }
    return Feedback(criteria=criteria, errors=errors, checks=checks, unverifiable=dropped)


async def generate_feedback(
    prompt: str, essay: str, bands: dict[str, float], notes: list[str], *,
    completer: llm.Completer, trace_id: str = "",
) -> Feedback:
    """Ask the feedback model; one retry when the reply is unusable or has no verified quote."""
    version = llm.prompt_version(llm.FEEDBACK_ROLE)
    system = llm.load_prompt("ielts_feedback", "writing_t2", version)
    messages = feedback_request(prompt, essay, bands, system, notes)
    temperature = llm.role_temperature(llm.FEEDBACK_ROLE)
    best: Feedback | None = None
    last: Exception | None = None
    for _ in range(ATTEMPTS):
        try:
            text = await completer(
                llm.FEEDBACK_ROLE, messages, temperature=temperature,
                max_tokens=FEEDBACK_MAX_TOKENS, trace_id=trace_id, prompt_version=version,
            )
            candidate = build_feedback(extract_json(text), essay)
        except (GradeParseError, llm.ModelUnavailableError) as exc:
            last = exc
            continue
        if best is None or _evidence_count(candidate) > _evidence_count(best):
            best = candidate
        if all(candidate.criteria[c].evidence for c in CRITERIA):
            return candidate
    if best is not None:
        return best
    assert last is not None
    raise last


def _evidence_count(fb: Feedback) -> int:
    return sum(len(c.evidence) for c in fb.criteria.values())
