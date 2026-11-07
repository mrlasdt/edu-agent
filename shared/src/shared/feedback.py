"""
User feedback on a reply, and the opt-in local capture of the turn's text.

The gateway sends each reply's trace id to the browser (`X-Trace-Id`). A thumbs up or down comes
back as `POST /feedback` and is appended, one JSON line, to the feedback log. A line holds the
trace id, the vote, an optional comment and a timestamp: never the learner's question or the
reply, which the turn log does not keep either. The turn log says which agent and prompt version
produced the reply; `python -m evals.turn_report --feedback` joins the two.

To turn a thumbs-down into a complete draft eval item (`python -m evals.feedback_to_item`) the
question and the reply must be kept somewhere. That is a separate decision: with
`TURN_CAPTURE_TEXT=1` the gateway also writes them to a local capture file keyed by trace id.
It is off by default, gitignored (`.turn-log/`) and never sent anywhere.

Both writes are best effort for the turn: a failure is logged and never breaks a reply.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Literal

from shared.src.shared.config import get_feedback_settings
from shared.src.shared.turn_meter import TurnLog

_log = logging.getLogger(__name__)

# Request size caps (a thumbs click carries a trace id, a vote and a short comment).
MAX_TRACE_ID_CHARS = 64
MAX_COMMENT_CHARS = 1_000

Vote = Literal["up", "down"]


def default_feedback_log() -> TurnLog | None:
    """The log at `FEEDBACK_LOG_PATH`; an empty path switches feedback off."""
    path = get_feedback_settings().feedback_log_path.strip()
    return TurnLog(path) if path else None


def default_capture_log() -> TurnLog | None:
    """The capture file, or None while `TURN_CAPTURE_TEXT` is off (the default)."""
    settings = get_feedback_settings()
    path = settings.turn_capture_path.strip()
    return TurnLog(path) if settings.turn_capture_text and path else None


def feedback_record(trace_id: str, vote: str, comment: str | None = None) -> dict[str, Any]:
    """One feedback line. There is deliberately no field for message text."""
    record: dict[str, Any] = {
        "ts": time.time(),
        "trace_id": trace_id,
        "vote": vote,
    }
    if comment:
        record["comment"] = comment
    return record


def capture_turn(log: TurnLog | None, trace_id: str, question: str, reply: str) -> None:
    """Keep one turn's text locally; a no-op without a log, never raises."""
    if log is None or not trace_id:
        return
    try:
        log.append(
            {
                "ts": time.time(),
                "trace_id": trace_id,
                "question": question,
                "reply": reply,
            }
        )
    except Exception:
        _log.warning("turn text not captured", exc_info=True)


def latest_votes(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The newest feedback record per trace id: a learner may change their mind."""
    latest: dict[str, dict[str, Any]] = {}
    for rec in sorted(records, key=lambda r: r.get("ts", 0.0)):
        trace_id, vote = rec.get("trace_id"), rec.get("vote")
        if isinstance(trace_id, str) and trace_id and vote in ("up", "down"):
            latest[trace_id] = rec
    return latest
