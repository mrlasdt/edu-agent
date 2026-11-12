"""The two checks on a Quant solve turn's answers (well-formed, then independent)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shared.src.shared import events
from shared.src.shared.models import Session
from shared.src.shared.turn_meter import TurnMeter

if TYPE_CHECKING:
    from services.agent.src.agent.agents.quant import QuantAgent


class SolveChecks:
    """
    The checks on one solve turn's answers. The well-formed check runs on every attempt; the
    independent check only on a well-formed answer, so a garbled answer does not cost the extra
    model call. The model writes its expression from the problem alone, so the same expression
    serves every attempt: it is asked for once per turn.
    """

    def __init__(
        self,
        agent: QuantAgent,
        session: Session,
        message: str,
        meter: TurnMeter,
    ) -> None:
        self._agent = agent
        self._session = session
        self._message = message
        self._meter = meter
        self._asked = False
        self._expression: str | None = None
        self.last = events.IndependentCheck(
            verdict=events.NOT_CHECKABLE, reason="no answer has been checked yet"
        )
        self._record()

    async def passes(self, full_text: str, claimed: str) -> bool:
        """Whether the answer survives both checks; `last` holds the independent verdict."""
        with self._meter.phase("verify"):
            well_formed = await self._agent.verify_well_formed(full_text, claimed)
        if not well_formed.verified:
            self.last = events.IndependentCheck(
                verdict=events.NOT_CHECKABLE, reason="the answer is not well-formed"
            )
            self._record()
            return False
        if not self._asked:
            self._asked = True
            self._expression = await self._agent.formalize(
                self._session, self._message, self._meter
            )
        with self._meter.phase("independent_check"):
            self.last = await self._agent.check_independently(self._expression, claimed)
        self._record()
        return self.last.verdict != events.DISAGREES

    def _record(self) -> None:
        fields: dict[str, Any] = {"verified_independent": self.last.verdict}
        if self.last.solver:
            fields["independent_solver"] = self.last.solver
        if self.last.reason:
            fields["independent_reason"] = self.last.reason
        self._meter.annotate(**fields)

    def retry_note(self, prev_answer: str) -> str:
        """What the model is told before its second attempt."""
        if self.last.verdict == events.DISAGREES:
            return (
                f"[My previous answer {prev_answer!r} disagreed with an independent calculation "
                f"of the problem, which gave {self.last.solver}. Let me check my work and "
                "try again.]"
            )
        return f"[My previous answer {prev_answer!r} could not be verified. Let me try again.]"

    def report(self) -> events.IndependentCheck:
        return self.last.model_copy(update={"reason": ""})
