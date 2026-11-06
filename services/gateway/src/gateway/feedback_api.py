"""
POST /feedback: a learner's thumbs up or down on one reply.

The reply's trace id reaches the browser in the `X-Trace-Id` header of `/chat`. The vote is
appended to the feedback log (`shared.feedback`) and, when Langfuse is on, also posted as a
`user-feedback` score on that trace. A trace id the turn log has never seen is rejected, so the
log only ever holds votes on real turns. Nothing here sees or stores message text.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from services.gateway.src.gateway.guardrails.rate_limit import RateLimiter
from shared.src.shared import feedback as feedback_log
from shared.src.shared import turn_meter
from shared.src.shared.observability import post_user_feedback

_log = logging.getLogger(__name__)

router = APIRouter()

# Votes are keyed by client address (the request carries no candidate id). Behind the web app's
# nginx every browser shares one address, so this is one global bound, not a per-user limit.
_rate_limiter = RateLimiter(max_per_minute=30, max_per_day=500)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(min_length=1, max_length=feedback_log.MAX_TRACE_ID_CHARS)
    vote: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=feedback_log.MAX_COMMENT_CHARS)


@router.post("/feedback")
def post_feedback(
    req: FeedbackRequest, request: Request, background: BackgroundTasks
) -> JSONResponse:
    client = request.client.host if request.client else "unknown"
    allowed, message = _rate_limiter.check(client)
    if not allowed:
        return JSONResponse(status_code=429, content={"detail": message})

    votes = feedback_log.default_feedback_log()
    turns = turn_meter.default_turn_log()
    if votes is None or turns is None:
        return JSONResponse(status_code=503, content={"detail": "feedback is switched off"})
    if turns.last_with_trace(req.trace_id) is None:
        return JSONResponse(status_code=404, content={"detail": "unknown trace id"})

    votes.append(feedback_log.feedback_record(req.trace_id, req.vote, req.comment))
    background.add_task(post_user_feedback, req.trace_id, req.vote, req.comment)
    return JSONResponse(status_code=200, content={"status": "ok"})
