"""
Deterministic checks on an essay, run before and after the model grades it.

They exist because an LLM grader rewards fluent text, can be talked into a score by the essay
itself, and cannot count words. Each check raises a flag; some also cap a criterion. The caps
are a documented policy, not a calibrated measurement (ADR-0014).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from services.agent.src.agent.ielts.models import Flag

MIN_WORDS_TO_MARK = 50
TARGET_WORDS = 250
NEAR_TARGET_WORDS = 230
SHORT_WORDS = 150
MIN_PARAGRAPHS = 4
OFF_TOPIC_OVERLAP = 0.2
PROMPT_COPY_SHARE = 0.3
REPETITION_RATIO = 0.25  # distinct words / words, over a response of at least 60 words
ENGLISH_STOPWORD_SHARE = 0.12

_STOPWORDS = frozenset(
    "a an and are as at be been but by can do does for from had has have how i in is it its "
    "of on or that the their them they this to was were what when which who why will with "
    "would you your all any more most other some such than then there these those very not no "
    "so if about into over also many much one two should could may might each both others "
    "people say said think believe give reasons answer example examples".split()
)
_ENGLISH_FUNCTION_WORDS = frozenset(
    "the of to and a in is that it for as with was be are this on by at or have not from but "
    "an they which their can has more been would there should".split()
)
_INJECTION = re.compile(
    r"ignore\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier)"
    r"|disregard\s+(?:all\s+|the\s+)?(?:previous|prior|above|instructions|rules)"
    r"|(?:give|award|assign|grant|rate)\s+(?:\w+\s+){0,5}?band\b"
    r"|(?:band|score)\s*(?:of\s*)?(?:9|nine)\b.{0,40}(?:all|every|each)"
    r"|you\s+are\s+(?:now\s+)?(?:the\s+|an?\s+)?(?:grader|examiner|marker|assistant)"
    r"|as\s+an?\s+(?:ai|llm|language\s+model)\b"
    r"|system\s+prompt|new\s+instructions?\b"
    r"|</?\s*(?:candidate_essay|essay|system|assistant)\b"
    r"|\"\s*(?:TR|CC|LR|GRA)\s*\"\s*:",
    re.I | re.S,
)


TASK_FAILURE_BAND = 5.5  # ceiling for the other criteria when the response is not an answer
TOO_SHORT_BAND = 5.5  # a response this short shows too little range and control for more
ONE_PARAGRAPH_CC_BAND = 5.0  # inadequate paragraphing is a band 5 Coherence descriptor


def cap_task_failure(caps: dict[str, float]) -> None:
    """An off-topic, copied or repetitive response: TR at most 4 and no criterion above 5.5."""
    caps["TR"] = min(caps.get("TR", 9.0), 4.0)
    for crit in ("CC", "LR", "GRA"):
        caps[crit] = min(caps.get(crit, 9.0), TASK_FAILURE_BAND)


def _flag(code: str, severity: Literal["info", "warn", "block"], message: str) -> Flag:
    return Flag(code=code, severity=severity, message=message)


@dataclass
class EssayChecks:
    word_count: int
    paragraph_count: int
    prompt_overlap: float
    flags: list[Flag] = field(default_factory=list)
    caps: dict[str, float] = field(default_factory=dict)

    @property
    def blocked(self) -> Flag | None:
        return next((f for f in self.flags if f.severity == "block"), None)

    def has(self, code: str) -> bool:
        return any(f.code == code for f in self.flags)


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", text)


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each paragraph: blank-line separated, else line separated."""
    pattern = r"\n\s*\n" if re.search(r"\n\s*\n", text.strip()) else r"\n"
    spans: list[tuple[int, int]] = []
    pos = 0
    for sep in [*re.finditer(pattern, text), None]:
        end = sep.start() if sep else len(text)
        chunk = text[pos:end]
        if chunk.strip():
            lead = len(chunk) - len(chunk.lstrip())
            spans.append((pos + lead, pos + len(chunk.rstrip())))
        if sep:
            pos = sep.end()
    return spans


def _stem(word: str) -> str:
    w = word.lower().strip("'’-")
    for suffix in ("ing", "ed", "es", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def _content_terms(text: str) -> set[str]:
    return {_stem(w) for w in words(text) if w.lower() not in _STOPWORDS and len(w) > 2}


def prompt_overlap(prompt: str, essay: str) -> float:
    """Share of the prompt's content words (lightly stemmed) that the essay also uses."""
    terms = _content_terms(prompt)
    return len(terms & _content_terms(essay)) / len(terms) if terms else 1.0


def prompt_copy_share(prompt: str, essay: str, n: int = 6) -> float:
    """Share of essay words inside a run of `n` or more words copied from the prompt."""
    p = [w.lower() for w in words(prompt)]
    e = [w.lower() for w in words(essay)]
    if len(e) < n or len(p) < n:
        return 0.0
    grams = {tuple(p[i : i + n]) for i in range(len(p) - n + 1)}
    covered = [False] * len(e)
    for i in range(len(e) - n + 1):
        if tuple(e[i : i + n]) in grams:
            covered[i : i + n] = [True] * n
    return sum(covered) / len(e)


def check_essay(prompt: str, essay: str) -> EssayChecks:
    ws = words(essay)
    n = len(ws)
    checks = EssayChecks(
        word_count=n,
        paragraph_count=len(paragraph_spans(essay)),
        prompt_overlap=prompt_overlap(prompt, essay),
    )
    flags = checks.flags

    if n < MIN_WORDS_TO_MARK:
        flags.append(_flag(
            "too_short_to_mark", "block",
            f"{n} words is too short to estimate a band (the minimum is {MIN_WORDS_TO_MARK}).",
        ))
        return checks

    lowered = [w.lower() for w in ws]
    if sum(w in _ENGLISH_FUNCTION_WORDS for w in lowered) / n < ENGLISH_STOPWORD_SHARE:
        flags.append(_flag(
            "not_english_prose", "block",
            "The response does not read as English prose, so it cannot be marked.",
        ))
        return checks

    if n < SHORT_WORDS:
        flags.append(_flag(
            "under_length", "warn",
            f"{n} words is far under the 250-word minimum; Task Response is capped at 4 and"
            f" the other criteria at {TOO_SHORT_BAND:g}.",
        ))
        checks.caps["TR"] = 4.0
        for crit in ("CC", "LR", "GRA"):
            checks.caps[crit] = min(checks.caps.get(crit, 9.0), TOO_SHORT_BAND)
    elif n < NEAR_TARGET_WORDS:
        flags.append(_flag(
            "under_length", "warn",
            f"{n} words is under the 250-word minimum; Task Response is capped at 5.5.",
        ))
        checks.caps["TR"] = 5.5
    elif n < TARGET_WORDS:
        flags.append(_flag(
            "slightly_under_length", "info",
            f"{n} words is just under the 250-word minimum; examiners penalise this.",
        ))

    if checks.paragraph_count == 1:
        flags.append(_flag(
            "one_paragraph", "warn",
            "The whole response is one paragraph; Coherence and Cohesion is capped at"
            f" {ONE_PARAGRAPH_CC_BAND:g}.",
        ))
        checks.caps["CC"] = min(checks.caps.get("CC", 9.0), ONE_PARAGRAPH_CC_BAND)
    elif checks.paragraph_count < MIN_PARAGRAPHS:
        flags.append(_flag(
            "few_paragraphs", "info",
            f"{checks.paragraph_count} paragraph(s); a Task 2 essay normally has four or more.",
        ))

    if checks.prompt_overlap < OFF_TOPIC_OVERLAP:
        flags.append(_flag(
            "possibly_off_topic", "warn",
            "The essay shares almost no key words with the question; Task Response is capped at"
            " 4 and the other criteria at 5.5.",
        ))
        cap_task_failure(checks.caps)

    if prompt_copy_share(prompt, essay) > PROMPT_COPY_SHARE:
        flags.append(_flag(
            "copies_prompt", "warn",
            "Much of the response is copied from the question; Task Response is capped at 4 and"
            " the other criteria at 5.5.",
        ))
        cap_task_failure(checks.caps)

    if len(set(lowered)) / n < REPETITION_RATIO and n >= 60:
        flags.append(_flag(
            "repetitive_text", "warn",
            "The response repeats the same words; Task Response and Lexical Resource are capped"
            " at 4 and the other criteria at 5.5.",
        ))
        cap_task_failure(checks.caps)
        checks.caps["LR"] = 4.0

    if _INJECTION.search(essay):
        flags.append(_flag(
            "addressed_to_marker", "warn",
            "The response contains text aimed at the marker rather than the question. It was"
            " not followed; Task Response is capped at 5.",
        ))
        checks.caps["TR"] = min(checks.caps.get("TR", 9.0), 5.0)
    return checks
