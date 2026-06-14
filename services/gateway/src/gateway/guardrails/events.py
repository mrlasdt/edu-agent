"""Structured guardrail event logging."""
from __future__ import annotations

import logging

_log = logging.getLogger("guardrails")


def emit_guardrail_event(
    layer: str, name: str, result: str, trace_id: str, **extra
) -> None:
    """
    Emit a structured guardrail event.
    Format: guardrail.{layer}.{name}.{result} trace_id={trace_id}
    """
    msg = f"guardrail.{layer}.{name}.{result} trace_id={trace_id}"
    for k, v in extra.items():
        msg += f" {k}={v}"
    _log.info(msg)
