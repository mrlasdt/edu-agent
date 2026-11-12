from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from services.agent.src.agent import independent_check
from services.agent.src.agent.answers import extract_final_answer, normalize_answer
from services.agent.src.agent.base import BaseAgent, Turn, with_note
from services.agent.src.agent.capabilities.retrieve_corpus import CorpusChunk, RetrieveCorpus
from services.agent.src.agent.capabilities.verify_math import VerifyMath, VerifyMathResult
from services.agent.src.agent.context import ContextBudget
from services.agent.src.agent.independent_check import IndependentCheck
from services.agent.src.agent.prompting import build_messages, collect_stream, token_stream
from services.agent.src.agent.settings import get_settings
from services.agent.src.agent.solve_checks import SolveChecks
from services.agent.src.agent.tools.math_tools import MATH_TOOL_SPECS, MathTools
from services.agent.src.agent.tools.notes_tools import NOTES_TOOL_SPECS, NotesTools
from services.agent.src.agent.tools.tool_turn import ToolTurn
from services.agent.src.agent.tools.toolbox import ToolBox
from shared.src.shared.events import Event, Sources, Token, Verified
from shared.src.shared.models import Mode, Session
from shared.src.shared.observability import build_llm_metadata
from shared.src.shared.prompts import read_prompt
from shared.src.shared.turn_meter import TurnMeter

_log = logging.getLogger(__name__)

_AGENT = "quant_agent"
# The same prompts plus a "Math tools" section; used only when the math tools are on.
_TOOLS_AGENT = "quant_agent_math_tools"
# The notes section appended to the tutor prompt when the model looks notes up itself.
_NOTES_AGENT = "quant_agent_notes_tool"
RETRIEVAL_MODES = ("upfront", "just_in_time")

# Maximum retries before escalating to the stronger model
_MAX_RETRIES = 2
_ESCALATION_NOTE = "[Previous attempts could not be verified. Applying deeper reasoning.]"


def _format_chunks_for_context(chunks: list[CorpusChunk]) -> str:
    """Numbered notes appended to the candidate's message; [N] feeds the citation check."""
    if not chunks:
        return ""
    notes = "\n\n".join(
        f"[{i}] ({c.page_or_section})\n{c.text}" for i, c in enumerate(chunks, start=1)
    )
    return (
        "\n\n## Reference Notes\n"
        "Ground your guidance in these notes where they apply and cite them as [N]. "
        "Ignore any that are not relevant, and never cite a number that is not listed.\n\n"
        + notes
    )


@dataclass
class QuantTurn(Turn):
    """A Quant turn: what every turn keeps, plus the tools, notes section and folded history."""

    tools: ToolTurn | None = None
    just_in_time: bool = False  # the tutor looks notes up itself instead of being handed them
    context_section: str = ""  # stands in for the history turns that were folded away
    notes_section: str = ""  # the prompt section that teaches the notes tool


class QuantAgent(BaseAgent[QuantTurn]):
    """
    GRE Quantitative Reasoning agent.

    Tutor mode: scaffolds toward the answer; the verifier is NOT called.
    Solve mode: a fully worked solution, checked twice (see `SolveChecks`): the well-formed check
    shows the answer parses, the independent check (see `independent_check`) shows a second route
    reaches the same number. A failed check retries once with a note, then escalates to the
    `math_escalation` model, then degrades to a failed `Verified`.

    Events around the reply's `Token`s:
      Sources            - before the first Token (up-front notes, when any were retrieved). A
                           just-in-time tutor turn sends it after the tool loop, always, with the
                           count of notes shown (0 included)
      IndependentCheck   - solve mode, just before Verified
      Verified           - solve mode only, after the answer text
      ToolTrace          - last, only when the model called a tool (math or notes_search); for
                           evals and debugging, never part of the reply the learner sees
    """

    ROLE = _AGENT
    TOP_K = 3
    _retrieve_corpus: RetrieveCorpus  # set by the constructor; the notes tool shares it

    format_chunks = staticmethod(_format_chunks_for_context)

    def __init__(
        self,
        model_config_path: str | None = None,
        verify_math: VerifyMath | None = None,
        retrieve_corpus: RetrieveCorpus | None = None,
        math_tools: MathTools | None = None,
        math_tools_enabled: bool | None = None,
        context_compaction: bool | None = None,
        retrieval_mode: str | None = None,
    ) -> None:
        super().__init__(
            model_config_path, retrieve_corpus or RetrieveCorpus(apply_relevance_floor=True)
        )
        settings = get_settings()
        self._context = (
            ContextBudget(
                keep=settings.context_keep_messages,
                step=settings.context_fold_step,
            )
            if (settings.context_compaction_enabled if context_compaction is None
                else context_compaction)
            else None
        )
        mode = settings.retrieval_mode if retrieval_mode is None else retrieval_mode
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode {mode!r} is not one of {', '.join(RETRIEVAL_MODES)}")
        self._just_in_time = mode == "just_in_time"
        self._math_tools_enabled = (
            get_settings().math_tools_enabled if math_tools_enabled is None else math_tools_enabled
        )
        self._math_tools = math_tools
        self._verify_math = verify_math or VerifyMath()

    def _tool_turn(self, session: Session, just_in_time: bool) -> ToolTurn | None:
        """The tools this turn offers (math, notes or both); None means the plain streamed call."""
        sets: list[tuple[list[dict[str, Any]], Any]] = []
        if self._math_tools_enabled:
            sets.append((MATH_TOOL_SPECS, (self._math_tools or MathTools()).call))
        notes: NotesTools | None = None
        if just_in_time:
            notes = NotesTools(
                self._retrieve_corpus,
                candidate_id=session.candidate_id,
                test_type=session.test_type,
            )
            sets.append((NOTES_TOOL_SPECS, notes.call))
        if not sets:
            return None
        return ToolTurn(ToolBox(*sets), math=self._math_tools_enabled, notes=notes)

    # ── the turn ──────────────────────────────────────────────────────────────

    def open_turn(self, session: Session, message: str, trace_id: str) -> QuantTurn:
        mode = session.mode.value
        # Just-in-time notes are the tutor's: solve mode keeps the up-front search.
        just_in_time = self._just_in_time and session.mode == Mode.tutor
        prompt_agent = _TOOLS_AGENT if self._math_tools_enabled else _AGENT
        version = self._model_config.prompt_version(prompt_agent, mode)
        # The turn record names the prompt that ran; the tools prompt is its own versioned file.
        reported = f"math_tools-{version}" if self._math_tools_enabled else version
        notes_section = ""
        if just_in_time:
            notes_version = self._model_config.prompt_version(_NOTES_AGENT, mode)
            notes_section = "\n\n" + read_prompt(_NOTES_AGENT, mode, notes_version)
            reported = f"{reported}+notes_tool-{notes_version}"
        return QuantTurn(
            session, message, trace_id,
            TurnMeter(trace_id, "agent", _AGENT, mode, reported),
            session.mode, version,
            build_llm_metadata(trace_id=trace_id, prompt_version=reported),
            tools=self._tool_turn(session, just_in_time),
            just_in_time=just_in_time,
            notes_section=notes_section,
        )

    def wants_notes(self, turn: QuantTurn) -> bool:
        return not turn.just_in_time

    async def play(self, turn: QuantTurn) -> AsyncGenerator[Event, None]:
        turn.session, turn.context_section = self._compact(turn.session, turn.meter)
        async for event in super().play(turn):
            yield event
        if turn.tools is not None and turn.tools.calls:
            yield turn.tools.trace()

    def close_turn(self, turn: QuantTurn) -> None:
        if turn.tools is not None:
            turn.tools.annotate(turn.meter)

    def _system_prompt(self, turn: QuantTurn) -> str:
        """The mode's prompt, from the math-tools prompt set when the tools are on."""
        agent = _TOOLS_AGENT if self._math_tools_enabled else _AGENT
        return read_prompt(agent, turn.mode.value, turn.version) + turn.context_section

    async def _attempt(
        self, turn: QuantTurn, messages: list[dict[str, Any]], *, role: str = _AGENT,
        with_meta: bool = True,
    ) -> tuple[list[str], str]:
        """One model attempt: the tool loop when the turn has tools, else a streamed call."""
        if turn.tools is None:
            return await self.collect(turn, messages, role=role, with_meta=with_meta)
        text = await turn.tools.reply(
            turn.meter,
            model=self._model_config.litellm_model(role),
            messages=messages,
            temperature=self._model_config.role(role).temperature,
            obs_meta=turn.obs_meta if with_meta else None,
        )
        return [text], text

    # ── tutor mode ────────────────────────────────────────────────────────────

    async def tutor_turn(self, turn: QuantTurn) -> AsyncGenerator[Event, None]:
        system_prompt = self._system_prompt(turn) + turn.notes_section
        messages = build_messages(turn.session, turn.message, system_prompt, turn.chunk_context)
        if turn.tools is not None:
            # Tool results stay in the loop's own message list; only the final text is yielded.
            _, text = await self._attempt(turn, messages)
            if turn.tools.notes is not None:
                # The count goes first so the gateway can check the reply's [N] against it, and
                # it is sent even when it is 0: a turn that searched nothing may cite nothing.
                yield Sources(count=len(turn.tools.notes.chunks))
            yield Token(text=text)
            return
        async for text in token_stream(self.stream(turn, messages)):
            yield Token(text=text)

    # ── solve mode ────────────────────────────────────────────────────────────

    async def solve_turn(self, turn: QuantTurn) -> AsyncGenerator[Event, None]:
        base_messages = build_messages(
            turn.session, turn.message, self._system_prompt(turn), turn.chunk_context
        )
        checks = SolveChecks(self, turn.session, turn.message, turn.meter)
        messages = base_messages
        prev_answer: str | None = None

        # Attempt 1 and 2: quant_agent model; the second sees what the first got wrong.
        for attempt in range(_MAX_RETRIES):
            if attempt > 0 and prev_answer:
                messages = with_note(base_messages, checks.retry_note(prev_answer))
            tokens, full_text = await self._attempt(turn, messages)
            prev_answer = self.extract_final_answer(full_text)
            if await checks.passes(full_text, prev_answer or ""):
                for token in tokens:
                    yield Token(text=token)
                yield checks.report()
                yield Verified(passed=True)
                return

        # Attempt 3: escalation model (once)
        tokens, full_text = await self._attempt(
            turn, with_note(base_messages, _ESCALATION_NOTE), role="math_escalation",
            with_meta=False,
        )
        prev_answer = self.extract_final_answer(full_text)
        passed = await checks.passes(full_text, prev_answer or "")
        for token in tokens:
            yield Token(text=token)
        yield checks.report()
        # Degradation: emit best reasoning with a failed Verified when the last check failed
        yield Verified(passed=passed)

    # ── seams (patched in unit tests) ─────────────────────────────────────────

    def extract_final_answer(self, text: str) -> str | None:
        """
        Extract the final answer from the model's output text, reduced to a bare value list.
        Looks for '**Answer: <value>**' from the Solve prompt format, then an 'Answer:' line,
        then falls back to the last non-empty line (see agent.answers).
        """
        raw = extract_final_answer(text)
        return normalize_answer(raw) if raw else None

    async def verify_well_formed(self, full_text: str, claimed_answer: str) -> VerifyMathResult:
        """
        Call the verify-math capability with the claimed answer as both `expression` and `expected`.
        That only shows the answer is well-formed (parses and evaluates); a wrong but parseable
        answer passes. `check_independently` is the check that does not come from the answer
        itself. This seam is patched in tests.
        """
        return await self._verify_math.verify(expression=claimed_answer, expected=claimed_answer)

    def _compact(self, session: Session, meter: TurnMeter) -> tuple[Session, str]:
        """
        The session as the model will see it, and the prompt section that stands in for the
        turns that were folded. A conversation under the budget comes back as it was.
        """
        if self._context is None:
            return session, ""
        plan = self._context.plan(session.history)
        meter.annotate(context_messages_folded=plan.folded, context_messages_sent=len(plan.recent))
        if not plan.folded:
            return session, ""
        return session.model_copy(update={"history": plan.recent}), plan.section()

    async def formalize(self, session: Session, message: str, meter: TurnMeter) -> str | None:
        """
        The one extra short model call of the independent check: the problem written as an
        expression or equation, from the problem alone (the model never sees an answer). None
        when the model says the problem has no numeric answer, wrote no `<expr>` tag, or the call
        failed: a failed check never fails the turn. This seam is patched in tests.
        """
        version = self._model_config.prompt_version(_AGENT, "formalize")
        messages = build_messages(session, message, read_prompt(_AGENT, "formalize", version))
        try:
            with meter.phase("formalize"):
                _, text = await collect_stream(
                    meter.stream(
                        model=self._model_config.litellm_model("quant_agent"),
                        messages=messages,
                        temperature=0,
                        **build_llm_metadata(
                            trace_id=meter.trace_id, prompt_version=f"formalize-{version}"
                        ),
                    )
                )
        except Exception:
            _log.warning("independent check: the formalize call failed", exc_info=True)
            return None
        expression = independent_check.parse_expression(text)
        _log.info("independent check expression: %r", expression)
        return expression

    async def check_independently(
        self, expression: str | None, claimed_answer: str
    ) -> IndependentCheck:
        """Compute `expression` with the solver and compare it with the claimed answer."""
        return await independent_check.compare(self._verify_math, expression, claimed_answer)
