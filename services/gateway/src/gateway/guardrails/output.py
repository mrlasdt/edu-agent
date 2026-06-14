"""Output guardrails: PII redaction and tutor-mode answer-leak detection."""
from __future__ import annotations

import re

import litellm

from services.gateway.src.gateway.guardrails.events import emit_guardrail_event
from shared.src.shared.config import load_model_config
from shared.src.shared.models import Mode

# PII patterns
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")

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


async def check_answer_leak(text: str, mode: Mode, trace_id: str) -> bool:
    """
    In Tutor mode: run a Haiku-tier check for answer leakage.
    In Solve mode: always returns False (leaking the answer is the point).
    """
    if mode != Mode.tutor:
        return False

    leaked = await _call_leak_checker(text)
    result = "fail" if leaked else "pass"
    emit_guardrail_event("generation", "tutor_mode_answer_leak", result, trace_id)
    return leaked


async def _call_leak_checker(text: str) -> bool:
    """Haiku-tier check for answer leakage. Seam for unit test patching."""
    cfg = load_model_config()
    model = cfg.litellm_model("orchestrator")
    response = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": _LEAK_SYSTEM},
            {"role": "user", "content": text},
        ],
        temperature=0,
        stream=False,
    )
    content = response.choices[0].message.content.strip().upper()
    return content.startswith("LEAK")
