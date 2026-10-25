"""Output guardrails: PII redaction and tutor-mode answer-leak detection."""
from __future__ import annotations

import logging
import re

import litellm

from services.gateway.src.gateway.guardrails.events import emit_guardrail_event
from shared.src.shared.config import get_settings, load_model_config
from shared.src.shared.models import Mode

# PII patterns
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(?<![\w.+\-*/^])"
    r"(?:\+\d{1,3}[-.\s]?\d{10}"
    r"|(?:\+\d{1,3}[-.\s]?)?(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]?\d{4}"
    r"|0\d{9,10})"
    r"(?![\w*/^]|[-.+]\d)"
)

_log = logging.getLogger(__name__)

# Regex stage of the answer-leak check. A value is a number, `var = value`, a parenthesised
# choice letter, or an upper-case choice letter that ends its clause.
_NUM_WORD = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)\b"
_NUM = rf"(?:[-+\u2212]?\$?\d[\d,./]*|{_NUM_WORD})"
_END = r"(?=[ \t]*(?:[.,;!?)]|\n))"
_ANSWER_TAG = re.compile(r"^[\s*#>_-]*(?:final\s+)?answer\s*:\s*\S", re.IGNORECASE | re.MULTILINE)
_BOLD_ANSWER = re.compile(r"\*\*\s*(?:final\s+)?answer\s*:", re.IGNORECASE)
_ANSWER_IS = re.compile(
    r"\b(?:final\s+)?answer\s+(?:is|=)\s*[:*\s]*"
    rf"(?:[-+\u2212]?\$?\d|[A-Za-z]\s*=\s*\S|\([A-Ea-e]\)|(?-i:[A-E]){_END})",
    re.IGNORECASE,
)
_CONCLUSION = re.compile(
    r"\b(?:therefore|thus|hence|so)\b[,:]?\s+[A-Za-z]\w*\s*(?:=|equals|is equal to)\s*"
    rf"{_NUM}{_END}",
    re.IGNORECASE,
)
_BARE_ASSIGNMENT = re.compile(
    r"(?:^|[.!?:;]|\b(?:so|then|thus|hence|therefore|gives|get|is)\b)"
    rf"[\s*_$]*[A-Za-z]\s*=\s*{_NUM}{_END}",
    re.IGNORECASE | re.MULTILINE,
)
_LEAK_PATTERNS = (_ANSWER_TAG, _BOLD_ANSWER, _ANSWER_IS, _CONCLUSION, _BARE_ASSIGNMENT)

_LEAK_SYSTEM = (
    "You are checking if a GRE tutor response reveals the final answer. "
    "The tutor must NEVER directly state the numeric or symbolic final answer. "
    "Reply with only: LEAK or SAFE"
)


def redact_pii(text: str) -> str:
    """Replace email addresses and phone numbers with [REDACTED]."""
    text = _EMAIL_RE.sub("[REDACTED]", text)
    text = _PHONE_RE.sub("[REDACTED]", text)
    return text


def detect_answer_leak(text: str) -> bool:
    """Regex stage of the answer-leak check: True when `text` states a final answer."""
    return any(p.search(text) for p in _LEAK_PATTERNS)


async def check_answer_leak(text: str, mode: Mode, trace_id: str) -> bool:
    """
    In Tutor mode: regex stage, then a Haiku-tier check for answer leakage.
    In Solve mode: always returns False (leaking the answer is the point).
    """
    if mode != Mode.tutor:
        return False

    leaked = detect_answer_leak(text) or await _call_leak_checker(text)
    result = "fail" if leaked else "pass"
    emit_guardrail_event("generation", "tutor_mode_answer_leak", result, trace_id)
    return leaked


async def _call_leak_checker(text: str) -> bool:
    """
    Haiku-tier check for answer leakage. Seam for unit test patching.
    Skipped (not a leak) when the configured provider has no API key or the call fails.
    """
    cfg = load_model_config()
    settings = get_settings()
    keys = {"openai": settings.openai_api_key, "anthropic": settings.anthropic_api_key}
    provider = cfg.role("orchestrator").provider
    if provider in keys and not keys[provider]:
        return False
    try:
        response = await litellm.acompletion(
            model=cfg.litellm_model("orchestrator"),
            messages=[
                {"role": "system", "content": _LEAK_SYSTEM},
                {"role": "user", "content": text},
            ],
            temperature=0,
            stream=False,
        )
    except Exception:
        _log.warning("answer-leak LLM check failed; skipping", exc_info=True)
        return False
    content = response.choices[0].message.content.strip().upper()
    return bool(content.startswith("LEAK"))
