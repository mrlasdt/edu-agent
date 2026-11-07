"""How a gateway turn ended: the `outcome` field of its turn record."""
from __future__ import annotations

from enum import StrEnum


class Outcome(StrEnum):
    """
    The values are the strings the gateway has always written to the turn log, so
    `evals.turn_report` groups old and new records together. Never rename one: that splits
    history. A new way for a turn to end is a new member.
    """

    ok = "ok"  # the agent answered (possibly with notices; those are `flags`)
    rate_limited = "rate_limited"
    moderation_unavailable = "moderation_unavailable"  # the safety check failed closed
    supportive_reply = "supportive_reply"  # self-harm hit: the fixed supportive text
    content_blocked = "content_blocked"
    clarification_requested = "clarification_requested"  # incomplete or unroutable question
    answer_leak_blocked = "answer_leak_blocked"  # a tutor reply was cut off at a stated answer
    error = "error"  # the stream raised
    client_cancelled = "client_cancelled"  # the client left before the turn finished
    # Never written by the gateway: the report's label for a turn with no gateway record.
    no_gateway_record = "no gateway record"
