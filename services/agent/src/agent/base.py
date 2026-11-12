from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

from services.agent.src.agent.capabilities.retrieve_corpus import CorpusChunk, RetrieveCorpus
from services.agent.src.agent.prompting import collect_stream
from shared.src.shared.config import load_model_config
from shared.src.shared.events import Event, Sources
from shared.src.shared.models import Mode, Session
from shared.src.shared.observability import build_llm_metadata
from shared.src.shared.turn_meter import TurnMeter

_log = logging.getLogger(__name__)


@dataclass
class Turn:
    """What the steps of one turn share. An agent that needs more state subclasses it."""

    session: Session
    message: str
    trace_id: str
    meter: TurnMeter
    mode: Mode
    version: str  # the prompt version the model config pins for this agent and mode
    obs_meta: dict[str, Any]  # the observability metadata every model call of the turn carries
    chunks: list[CorpusChunk] = field(default_factory=list)  # notes retrieved up front
    chunk_context: str = ""  # those notes as the block appended to the learner's message


class BaseAgent[T: Turn](ABC):
    """
    The turn every agent runs, registered in AgentRegistry by (test_type, section).
    Adding a new test type means subclassing BaseAgent and registering: no changes to the
    registry or orchestrator.

    `run` is the template: open the turn's meter, retrieve notes, report how many there are
    (`Sources`), play the tutor or the solve turn, and always close the meter, recording the
    error type when the turn raised. A subclass sets `ROLE` (its model role, and the name of its
    prompts) and `TOP_K`, writes `tutor_turn` and `solve_turn`, and overrides a hook only where it
    differs (`open_turn`, `wants_notes`, `format_chunks`, `play`, `close_turn`).
    """

    ROLE: ClassVar[str]
    # Notes fetched from the corpus before the model is called; 0 means the agent fetches none.
    TOP_K: ClassVar[int] = 0
    # An agent with one mode (the IELTS coach) names it here; the session's mode is ignored.
    SINGLE_MODE: ClassVar[Mode | None] = None

    def __init__(
        self,
        model_config_path: str | None = None,
        retrieve_corpus: RetrieveCorpus | None = None,
    ) -> None:
        self._model_config = load_model_config(model_config_path)
        self._model_config_path = model_config_path
        self._retrieve_corpus = retrieve_corpus

    async def run(
        self, session: Session, message: str, trace_id: str = ""
    ) -> AsyncGenerator[Event, None]:
        """
        Process a turn and yield its events (see `shared.events`): `Token`s of the reply,
        interleaved with what the agent reports about it (sources, verdicts, ...).
        Session carries mode (tutor/solve) and history.
        """
        turn = self.open_turn(session, message, trace_id)
        error: str | None = None
        try:
            async for event in self.play(turn):
                yield event
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            self.close_turn(turn)
            turn.meter.finish(error=error)

    # ── the turn ──────────────────────────────────────────────────────────────

    def open_turn(self, session: Session, message: str, trace_id: str) -> T:
        """
        The turn's state and its meter, named by the prompt version the agent runs. An agent whose
        turn keeps more than a `Turn` overrides this and builds its own kind.
        """
        mode = session.mode if self.SINGLE_MODE is None else self.SINGLE_MODE
        version = self._model_config.prompt_version(self.ROLE, mode.value)
        meter = TurnMeter(trace_id, "agent", self.ROLE, mode.value, version)
        obs_meta = build_llm_metadata(trace_id=trace_id, prompt_version=version)
        return cast(T, Turn(session, message, trace_id, meter, mode, version, obs_meta))

    async def play(self, turn: T) -> AsyncGenerator[Event, None]:
        """Retrieve notes, say how many there are, then play the tutor or the solve turn."""
        if self.wants_notes(turn):
            with turn.meter.phase("retrieval"):
                turn.chunks = await self._retrieve(turn.session, turn.message)
            turn.meter.annotate(notes_retrieved=len(turn.chunks))
        if turn.chunks:
            yield Sources(count=len(turn.chunks))
        turn.chunk_context = self.format_chunks(turn.chunks)
        play_mode = self.tutor_turn if turn.mode == Mode.tutor else self.solve_turn
        async for event in play_mode(turn):
            yield event

    def wants_notes(self, turn: T) -> bool:
        """Whether this turn fetches its notes up front."""
        return self.TOP_K > 0

    @staticmethod
    def format_chunks(chunks: list[CorpusChunk]) -> str:
        """The notes as the block appended to the learner's message ("" when there are none)."""
        return ""

    def close_turn(self, turn: T) -> None:
        """Hook: record what the agent kept for the turn record. Runs before the meter closes."""

    @abstractmethod
    def tutor_turn(self, turn: T) -> AsyncGenerator[Event, None]:
        """Yield the tutor-mode reply and what the agent reports about it."""
        ...  # pragma: no cover

    @abstractmethod
    def solve_turn(self, turn: T) -> AsyncGenerator[Event, None]:
        """Yield the solve-mode reply and what the agent reports about it."""
        ...  # pragma: no cover

    # ── seams shared by the agents (patched in unit tests) ────────────────────

    async def _retrieve(self, session: Session, message: str) -> list[CorpusChunk]:
        """
        Fetch grounding notes from the Corpus Service. Fails open: an unreachable corpus costs the
        turn its grounding, not the answer.
        """
        if self._retrieve_corpus is None:
            return []
        try:
            return await self._retrieve_corpus.retrieve(
                query=message,
                candidate_id=session.candidate_id,
                school_id="",
                test_type=session.test_type,
                top_k=self.TOP_K,
            )
        except Exception:
            _log.warning("corpus retrieval failed; answering without notes", exc_info=True)
            return []

    def stream(
        self, turn: T, messages: list[dict[str, Any]], *, role: str | None = None,
        with_meta: bool = True,
    ) -> AsyncIterator[Any]:
        """One metered streaming call to the model of `role` (default: the agent's own)."""
        role = role or self.ROLE
        return turn.meter.stream(
            model=self._model_config.litellm_model(role),
            messages=messages,
            temperature=self._model_config.role(role).temperature,
            **(turn.obs_meta if with_meta else {}),
        )

    async def collect(
        self, turn: T, messages: list[dict[str, Any]], *, role: str | None = None,
        with_meta: bool = True,
    ) -> tuple[list[str], str]:
        """`stream`, drained: the tokens and the full text."""
        return await collect_stream(self.stream(turn, messages, role=role, with_meta=with_meta))


def with_note(messages: list[dict[str, Any]], note: str) -> list[dict[str, Any]]:
    """The messages plus a note the model reads as its own previous turn (a retry cue)."""
    return [*messages, {"role": "assistant", "content": note}]
