"""
Streaming reply guard for /chat: answer-leak, citation and PII layers applied to the
agent's token stream.

Text is held back to the last sentence or line boundary so a marker split across chunks is
still caught while single-paragraph replies keep streaming,
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
from services.gateway.src.gateway.guardrails.output import detect_answer_leak, redact_pii
from shared.src.shared.models import Mode

LEAK_NOTICE = (
    "\n\n[Answer withheld: in tutor mode I don't state the final answer. "
    "Ask for a hint, or use \"Show me the answer\".]"
)
LLM_LEAK_NOTICE = (
    "\n\n[Answer check: this reply may state the final answer. "
    "Ask for a hint if you want to work it out yourself.]"
)
CITATION_NOTICE = (
    "\n\n[Citation check: removed {n} sentence(s) that cited sources I did not retrieve.]"
)
PII_NOTICE = "\n\n[Privacy: removed an email address or phone number from this reply.]"
INPUT_PII_NOTICE = (
    "[Privacy: removed an email address or phone number from your message "
    "before processing it.]\n\n"
)


# A flush point: a sentence end (terminator then whitespace) or a newline. Citation sentences
# and multi-word PII never straddle one, so each flushed segment is checked whole.
_BOUNDARY = re.compile(r"(?<=[.!?])\s|\n")


class ReplyGuard:
    """
    Feed agent text with `feed()`, then call `finish()`; both return text safe to emit.
    After `stopped` turns true (a leak was blocked) the caller should stop reading upstream.
    """

    def __init__(self, mode: Mode, trace_id: str, num_sources: int | None = None) -> None:
        self._mode = mode
        self._trace_id = trace_id
        self._num_sources = num_sources
        self._pending = ""
        self._pii_redacted = False
        self._citations_removed = 0
        self.text = ""
        self.stopped = False

    def feed(self, text: str) -> str:
        if self.stopped:
            return ""
        self.text += text
        self._pending += text
        if self._leaks(self.text):
            return self._block_leak()
        cut = 0
        for match in _BOUNDARY.finditer(self._pending):
            cut = match.end()
        segment, self._pending = self._pending[:cut], self._pending[cut:]
        return self._process(segment)

    def finish(self) -> str:
        if self.stopped:
            return ""
        if self._leaks(self.text + "\n"):
            return self._block_leak()
        out = self._process(self._pending)
        self._pending = ""
        if self._citations_removed:
            out += CITATION_NOTICE.format(n=self._citations_removed)
        if self._pii_redacted:
            out += PII_NOTICE
        return out

    def _leaks(self, text: str) -> bool:
        return self._mode == Mode.tutor and detect_answer_leak(text)

    def _block_leak(self) -> str:
        emit_guardrail_event("generation", "tutor_mode_answer_leak", "fail", self._trace_id)
        self.stopped = True
        self._pending = ""
        return LEAK_NOTICE

    def _process(self, segment: str) -> str:
        if not segment:
            return ""
        if self._num_sources is not None:
            valid, invalid = check_citations(segment, self._num_sources)
            if not valid:
                emit_guardrail_event(
                    "generation", "citation", "fail", self._trace_id, invalid=len(invalid)
                )
                self._citations_removed += 1
                body = segment.rstrip()
                trailing = segment[len(body):]
                body = strip_uncited_claims(body, invalid)
                segment = body + trailing if body else ""
        redacted = redact_pii(segment)
        if redacted != segment:
            emit_guardrail_event("output", "pii", "redacted", self._trace_id)
            self._pii_redacted = True
        return redacted


def redact_input(text: str, trace_id: str) -> tuple[str, bool]:
    """Redact PII from a Candidate message. Returns (clean text, whether anything changed)."""
    clean = redact_pii(text)
    changed = clean != text
    if changed:
        emit_guardrail_event("input", "pii", "redacted", trace_id)
    return clean, changed
