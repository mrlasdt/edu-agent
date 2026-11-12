"""A turn's bytes, run through the gates and the answer, with its turn record written."""
from __future__ import annotations

import time
from collections.abc import AsyncGenerator

from services.gateway.src.gateway.chat.answer import answer
from services.gateway.src.gateway.chat.gates import GATES
from services.gateway.src.gateway.chat.turn import ChatTurn
from shared.src.shared.feedback import capture_turn
from shared.src.shared.trailer import MARK
from shared.src.shared.turn_meter import TurnLog
from shared.src.shared.turn_outcome import Outcome


async def guarded_stream(turn: ChatTurn) -> AsyncGenerator[bytes, None]:
    for gate in GATES:
        verdict = await gate(turn)
        if not verdict.allowed:
            turn.meter.annotate(outcome=verdict.outcome)
        if verdict.message:
            yield verdict.message.encode("utf-8")
        if not verdict.allowed:
            return
    async for piece in answer(turn):
        yield piece


async def recorded_stream(
    turn: ChatTurn, capture_log: TurnLog | None
) -> AsyncGenerator[bytes, None]:
    """The turn's bytes, with its first-byte time, flags, final outcome and optional text capture
    written to the turn record however the stream ends."""
    meter = turn.meter
    started = time.perf_counter()
    first_byte_seen = False
    error: str | None = None
    completed = False
    sent: list[bytes] = []
    try:
        async for piece in guarded_stream(turn):
            if not first_byte_seen:
                first_byte_seen = True
                meter.annotate(first_byte_ms=(time.perf_counter() - started) * 1000)
            if capture_log is not None:
                sent.append(piece)
            yield piece
        completed = True
    except Exception as exc:
        error = type(exc).__name__
        meter.annotate(outcome=Outcome.error)
        raise
    finally:
        # The server abandons this generator when the client disconnects (Stop, a closed
        # tab): by cancellation, which also closes the agent request, or by closing it.
        if not completed and error is None:
            meter.annotate(outcome=Outcome.client_cancelled)
        meter.annotate(flags=turn.flags)
        meter.finish(error=error)
        if capture_log is not None:
            reply = b"".join(sent).decode("utf-8", errors="replace").partition(MARK)[0]
            capture_turn(capture_log, turn.trace_id, turn.message, reply)
