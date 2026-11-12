"""The math and notes tools of one Quant turn, with the bookkeeping the turn record needs."""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from services.agent.src.agent.tool_loop import ANSWERED, ToolCallRecord, run_tool_loop
from services.agent.src.agent.tools.notes_tools import NotesTools
from services.agent.src.agent.tools.toolbox import ToolBox
from shared.src.shared import events
from shared.src.shared.events import ToolTrace
from shared.src.shared.turn_meter import TurnMeter

_log = logging.getLogger(__name__)

# What a turn says when the model's tool loop ended without any text.
NO_REPLY = "I could not finish working that out just now. Could you send the problem again?"


class ToolTurn:
    """
    The math tools for one turn: runs the bounded tool loop for each model attempt and keeps
    the calls, so the turn can report them. One instance per turn (solve mode may attempt
    several times; the calls of all attempts are kept).
    """

    def __init__(
        self, box: ToolBox, *, math: bool = True, notes: NotesTools | None = None
    ) -> None:
        self._box = box
        self._math = math
        self.notes = notes
        self.calls: list[ToolCallRecord] = []
        self.rounds = 0
        self.stop_reason = ANSWERED

    async def reply(
        self,
        meter: TurnMeter,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        obs_meta: dict[str, Any] | None = None,
    ) -> str:
        outcome = await run_tool_loop(
            meter,
            model=model,
            messages=messages,
            tools=self._box.specs,
            execute=self._box.call,
            temperature=temperature,
            **(obs_meta or {}),
        )
        self.calls += [replace(c, round=c.round + self.rounds) for c in outcome.calls]
        self.rounds += outcome.rounds
        if self.stop_reason == ANSWERED:  # a turn keeps the first reason it was stopped for
            self.stop_reason = outcome.stop_reason
        for call in outcome.calls:
            _log.info(
                "tool %s ok=%s ms=%s args=%s result=%s",
                call.name, call.ok, call.ms, call.arguments, call.result,
            )
        return outcome.text.strip() or NO_REPLY

    def annotate(self, meter: TurnMeter) -> None:
        """Put the calls on the turn record (the operator's log; the learner never sees it)."""
        if self._math:
            meter.annotate(math_tools=True)
        if self.notes is not None:
            meter.annotate(
                retrieval="just_in_time",
                notes_retrieved=len(self.notes.chunks),
                notes_searches=len(self.notes.concepts),
            )
        meter.annotate(
            tool_rounds=self.rounds,
            tool_calls=[c.public() for c in self.calls],
            tool_errors=sum(1 for c in self.calls if not c.ok),
            tool_budget_exhausted=self.stop_reason != ANSWERED,
            tool_loop_stop=self.stop_reason,
        )

    def trace(self) -> ToolTrace:
        return ToolTrace(calls=tuple(events.ToolCall(**c.public()) for c in self.calls))
