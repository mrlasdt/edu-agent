"""
Streaming reply guard for /chat: answer-leak, citation and PII layers applied to the
agent's token stream.

Tokens are buffered to line boundaries so a marker split across chunks is still caught,
and every decision that changes what the Candidate sees is announced in the stream with a
bracketed notice, never dropped silently.
"""
from __future__ import annotations

import re

from services.agent.src.agent.critics.citation_checker import (
    check_citations,
    strip_uncited_claims,
)
from services.gateway.src.gateway.guardrails.events import emit_guardrail_event
from services.gateway.src.gateway.guardrails.output import redact_pii
from shared.src.shared.models import Mode

# A tutor reply that declares a final answer: an `Answer:` tag or line, or "the answer is <value>"
# where a digit follows shortly after (e.g. "x = 4") or the value is a single choice letter.
_ANSWER_TAG = re.compile(r"^[\s*#>_-]*(?:final\s+)?answer\s*:\s*\S", re.IGNORECASE)
_BOLD_ANSWER = re.compile(r"\*\*\s*(?:final\s+)?answer\s*:", re.IGNORECASE)
_ANSWER_IS = re.compile(
    r"\b(?:final\s+)?answer\s+(?:is|=)\s*[:*\s]*(?:[^\n]{0,40}?\d|\(?[A-Ea-e]\)?(?:[\s.,;)]|$))",
    re.IGNORECASE,
)

LEAK_NOTICE = (
    "\n\n[Answer withheld: in tutor mode I don't state the final answer. "
    "Ask for a hint, or use \"Show me the answer\".]"
)
CITATION_NOTICE = (
    "\n\n[Citation check: removed {n} sentence(s) that cited sources I did not retrieve.]"
)
PII_NOTICE = "\n\n[Privacy: removed an email address or phone number from this reply.]"
INPUT_PII_NOTICE = (
    "[Privacy: removed an email address or phone number from your message "
    "before processing it.]\n\n"
)


def detect_answer_leak(text: str) -> bool:
    """True when `text` states a final answer. Deterministic, so it runs offline."""
    return any(
        _ANSWER_TAG.search(line) or _BOLD_ANSWER.search(line) or _ANSWER_IS.search(line)
        for line in text.splitlines()
    )


class ReplyGuard:
    """
    Feed agent text with `feed()`, then call `finish()`; both return text safe to emit.
    After `stopped` turns true (a leak was blocked) the caller should stop reading upstream.
    """

    def __init__(self, mode: Mode, trace_id: str, num_sources: int | None = None) -> None:
        self._mode = mode
        self._trace_id = trace_id
        self._num_sources = num_sources
        self._buffer = ""
        self._pii_redacted = False
        self._citations_removed = 0
        self.stopped = False

    def feed(self, text: str) -> str:
        if self.stopped:
            return ""
        self._buffer += text
        lines = self._buffer.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._buffer = lines.pop()
        else:
            self._buffer = ""
        return self._process(lines)

    def finish(self) -> str:
        out = ""
        if not self.stopped:
            out = self._process([self._buffer] if self._buffer else [])
            self._buffer = ""
        if not self.stopped:
            if self._citations_removed:
                out += CITATION_NOTICE.format(n=self._citations_removed)
            if self._pii_redacted:
                out += PII_NOTICE
        return out

    def _process(self, lines: list[str]) -> str:
        out: list[str] = []
        for line in lines:
            if self._mode == Mode.tutor and detect_answer_leak(line):
                emit_guardrail_event("generation", "tutor_mode_answer_leak", "fail", self._trace_id)
                self.stopped = True
                out.append(LEAK_NOTICE)
                break
            if self._num_sources is not None:
                valid, invalid = check_citations(line, self._num_sources)
                if not valid:
                    emit_guardrail_event(
                        "generation", "citation", "fail", self._trace_id, invalid=len(invalid)
                    )
                    self._citations_removed += 1
                    trailing = line[len(line.rstrip("\r\n")):]
                    line = strip_uncited_claims(line.rstrip("\r\n"), invalid)
                    line = line + trailing if line else ""
            redacted = redact_pii(line)
            if redacted != line:
                emit_guardrail_event("output", "pii", "redacted", self._trace_id)
                self._pii_redacted = True
            out.append(redacted)
        return "".join(out)


def redact_input(text: str, trace_id: str) -> tuple[str, bool]:
    """Redact PII from a Candidate message. Returns (clean text, whether anything changed)."""
    clean = redact_pii(text)
    changed = clean != text
    if changed:
        emit_guardrail_event("input", "pii", "redacted", trace_id)
    return clean, changed
