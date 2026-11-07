"""What a guard decides about a turn: let it go on, or stop it with a reply and an outcome."""
from __future__ import annotations

from dataclasses import dataclass

from shared.src.shared.turn_outcome import Outcome


@dataclass(frozen=True, slots=True)
class Verdict:
    """
    `message` is always text for the candidate and is sent when not empty. A guard that does not
    allow the turn also says how the turn ends (`outcome`). `Verdict.allow()` is the pass.

    The answer-leak check runs after the reply has streamed, so it can only flag: it returns
    `Verdict(allowed=False)` with the default `Outcome.ok`, and the turn keeps that outcome.
    """

    allowed: bool
    outcome: Outcome = Outcome.ok
    message: str = ""

    @classmethod
    def allow(cls, message: str = "") -> Verdict:
        return cls(True, Outcome.ok, message)

    @classmethod
    def stop(cls, outcome: Outcome, message: str) -> Verdict:
        return cls(False, outcome, message)
