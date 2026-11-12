"""Output guardrails: PII redaction and tutor-mode answer-leak detection."""
from __future__ import annotations

import logging
import re

from services.gateway.src.gateway.guardrails.events import emit_guardrail_event
from services.gateway.src.gateway.guardrails.verdict import Verdict
from services.gateway.src.gateway.settings import get_settings
from shared.src.shared.config import load_model_config
from shared.src.shared.models import Mode
from shared.src.shared.prompts import read_prompt
from shared.src.shared.turn_meter import TurnMeter

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
_END = r"(?=[ \t]*(?:[.,;!?)]|\n|\Z))"
_ANSWER_TAG = re.compile(
    r"(?:^|[.!?;])[\s*#>_-]*(?:final\s+)?answer\s*:\s*\S", re.IGNORECASE | re.MULTILINE
)
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

# The model role of the leak check: the cheap guard model, set apart from the `orchestrator` role
# so a change to one does not move the other.
LEAK_CHECK_ROLE = "guard"

# The verdict is one word (LEAK or SAFE). A reasoning model also spends hidden thinking tokens
# against this cap, so it is sized well above the visible answer: low enough to stop a runaway
# reply from stalling the turn's last step, high enough that a reasoning model still finishes.
LEAK_CHECK_MAX_TOKENS = 512


def redact_pii(text: str) -> str:
    """Replace email addresses and phone numbers with [REDACTED]."""
    text = _EMAIL_RE.sub("[REDACTED]", text)
    text = _PHONE_RE.sub("[REDACTED]", text)
    return text


def detect_answer_leak(text: str) -> bool:
    """Regex stage of the answer-leak check: True when `text` states a final answer."""
    return any(p.search(text) for p in _LEAK_PATTERNS)


async def check_answer_leak(
    text: str, mode: Mode, trace_id: str, meter: TurnMeter | None = None
) -> Verdict:
    """
    In Tutor mode: regex stage, then a Haiku-tier check for answer leakage.
    In Solve mode: always allows (leaking the answer is the point).
    A leak is `Verdict(allowed=False)` with the turn's outcome unchanged: the reply has already
    streamed, so the caller can only flag it.
    With a `meter`, the LLM stage's model call is measured.
    """
    if mode != Mode.tutor:
        return Verdict.allow()

    leaked = detect_answer_leak(text) or await _call_leak_checker(text, meter)
    result = "fail" if leaked else "pass"
    emit_guardrail_event("generation", "tutor_mode_answer_leak", result, trace_id)
    return Verdict(allowed=False) if leaked else Verdict.allow()


async def _call_leak_checker(text: str, meter: TurnMeter | None = None) -> bool:
    """
    Haiku-tier check for answer leakage. Seam for unit test patching.
    Skipped (not a leak) when the configured provider has no API key or the call fails.
    """
    cfg = load_model_config()
    settings = get_settings()
    keys = {"openai": settings.openai_api_key, "anthropic": settings.anthropic_api_key}
    provider = cfg.role(LEAK_CHECK_ROLE).provider
    if provider in keys and not keys[provider]:
        return False
    version = cfg.prompt_version("orchestrator", "leak_check")
    system = read_prompt("orchestrator", "leak_check", version)
    try:
        response = await (meter or TurnMeter.detached()).complete(
            model=cfg.litellm_model(LEAK_CHECK_ROLE),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=LEAK_CHECK_MAX_TOKENS,
        )
    except Exception:
        _log.warning("answer-leak LLM check failed; skipping", exc_info=True)
        return False
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        # Cut off by the token cap before any verdict (a reasoning model can spend all of it
        # thinking): inconclusive, so skipped like a failed call, not a leak and not a crash.
        _log.warning("answer-leak LLM check returned no verdict; skipping")
        return False
    return content.strip().upper().startswith("LEAK")
