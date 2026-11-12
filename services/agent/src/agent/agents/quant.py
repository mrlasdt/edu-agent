from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import replace
from typing import Any

from services.agent.src.agent import independent_check
from services.agent.src.agent.answers import extract_final_answer, normalize_answer
from services.agent.src.agent.base import BaseAgent
from services.agent.src.agent.capabilities.retrieve_corpus import CorpusChunk, RetrieveCorpus
from services.agent.src.agent.capabilities.verify_math import VerifyMath, VerifyMathResult
from services.agent.src.agent.context import ContextBudget
from services.agent.src.agent.prompting import build_messages, collect_stream
from services.agent.src.agent.settings import get_settings
from services.agent.src.agent.tool_loop import ANSWERED, ToolCallRecord, run_tool_loop
from services.agent.src.agent.tools.math_tools import MATH_TOOL_SPECS, MathTools
from services.agent.src.agent.tools.notes_tools import NOTES_TOOL_SPECS, NotesTools
from services.agent.src.agent.tools.toolbox import ToolBox
from shared.src.shared import events
from shared.src.shared.config import load_model_config
from shared.src.shared.events import Event, Sources, Token, ToolTrace, Verified
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
_NO_REPLY = "I could not finish working that out just now. Could you send the problem again?"

# Maximum retries before escalating to the stronger model
_MAX_RETRIES = 2


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


class _ToolTurn:
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
        return outcome.text.strip() or _NO_REPLY

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


class _SolveChecks:
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
            self._expression = await self._agent._formalize(
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


class QuantAgent(BaseAgent):
    """
    GRE Quantitative Reasoning agent.

    Tutor mode: scaffolds toward the answer; verifier is NOT called.
    Solve mode: returns a fully worked solution and checks the final answer twice with the
                verify-math capability. The well-formed check (it parses and evaluates) compares the
                answer with itself, so it does not establish that the answer is right. The
                independent check does: one extra short model call writes the problem as an
                expression without seeing the answer, the solver computes it, and the result is
                compared with the claimed answer (see `independent_check`). The turn record's
                `verified_independent` says agrees, disagrees or not_checkable. A disagreement
                counts as a failed check.  Retry / escalation pipeline:
                  attempt 1 (quant_agent model)
                  → if a check fails:   attempt 2 (quant_agent model, mismatch context)
                  → if still failing:   attempt 3 (math_escalation model)
                  → if still failing:   degrade gracefully (emit a failed Verified)
    """

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
        self._model_config = load_model_config(model_config_path)
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
        self._retrieve_corpus = retrieve_corpus or RetrieveCorpus(apply_relevance_floor=True)
        self._verify_math = verify_math or VerifyMath()

    def _tool_turn(self, session: Session, just_in_time: bool) -> _ToolTurn | None:
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
        return _ToolTurn(ToolBox(*sets), math=self._math_tools_enabled, notes=notes)

    # ── public interface ──────────────────────────────────────────────────────

    async def run(
        self, session: Session, message: str, trace_id: str = ""
    ) -> AsyncGenerator[Event, None]:
        """
        Yield the reply as `Token`s as it is generated, with these events around it:
          Sources            — before the first Token. Up-front retrieval: only when corpus
                               notes were retrieved. Just-in-time tutor turn: after the tool
                               loop, always, with the count of notes shown (0 included)
          IndependentCheck   — Solve mode, just before Verified; `solver` (what the solver
                               computed) is set only when the verdict is "disagrees"
          Verified           — only emitted in Solve mode, after the answer text
          ToolTrace          — last, only when the model called any tool (math or
                               notes_search): each call with its arguments and result (for
                               evals and debugging; never part of the reply the learner sees)
        """
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
        obs_meta = build_llm_metadata(trace_id=trace_id, prompt_version=reported)
        meter = TurnMeter(trace_id, "agent", "quant_agent", mode, reported)
        tools = self._tool_turn(session, just_in_time)
        try:
            session, context_section = self._compact(session, meter)
            chunks: list[CorpusChunk] = []
            if not just_in_time:
                with meter.phase("retrieval"):
                    chunks = await self._retrieve(session, message)
                meter.annotate(notes_retrieved=len(chunks))
            if chunks:
                yield Sources(count=len(chunks))
            chunk_context = _format_chunks_for_context(chunks)
            if session.mode == Mode.tutor:
                async for event in self._tutor_turn(
                    session, message, meter, obs_meta, version, chunk_context, tools,
                    context_section, notes_section,
                ):
                    yield event
            else:
                async for item in self._solve_turn(
                    session, message, meter, obs_meta, version, chunk_context, tools,
                    context_section,
                ):
                    yield item
            if tools is not None and tools.calls:
                yield tools.trace()
        except Exception as exc:
            if tools is not None:
                tools.annotate(meter)
            meter.finish(error=type(exc).__name__)
            raise
        finally:
            if tools is not None:
                tools.annotate(meter)
            meter.finish()

    # ── tutor mode ────────────────────────────────────────────────────────────

    async def _tutor_turn(
        self,
        session: Session,
        message: str,
        meter: TurnMeter,
        obs_meta: dict[str, Any] | None = None,
        version: str = "v1",
        chunk_context: str = "",
        tools: _ToolTurn | None = None,
        context_section: str = "",
        notes_section: str = "",
    ) -> AsyncGenerator[Event, None]:
        agent = _TOOLS_AGENT if self._math_tools_enabled else _AGENT
        system_prompt = read_prompt(agent, Mode.tutor.value, version)
        system_prompt += context_section + notes_section
        messages = build_messages(session, message, system_prompt, chunk_context)
        model = self._model_config.litellm_model("quant_agent")
        temperature = self._model_config.role("quant_agent").temperature

        if tools is not None:
            # Tool results stay in the loop's own message list; only the final text is yielded.
            text = await tools.reply(
                meter, model=model, messages=messages, temperature=temperature,
                obs_meta=obs_meta,
            )
            if tools.notes is not None:
                # The count goes first so the gateway can check the reply's [N] against it, and
                # it is sent even when it is 0: a turn that searched nothing may cite nothing.
                yield Sources(count=len(tools.notes.chunks))
            yield Token(text=text)
            return
        response = meter.stream(
            model=model,
            messages=messages,
            temperature=temperature,
            **(obs_meta or {}),
        )
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield Token(text=content)

    # ── solve mode ────────────────────────────────────────────────────────────

    async def _solve_turn(
        self,
        session: Session,
        message: str,
        meter: TurnMeter,
        obs_meta: dict[str, Any] | None = None,
        version: str = "v1",
        chunk_context: str = "",
        tools: _ToolTurn | None = None,
        context_section: str = "",
    ) -> AsyncGenerator[Event, None]:
        agent = _TOOLS_AGENT if self._math_tools_enabled else _AGENT
        system_prompt = read_prompt(agent, Mode.solve.value, version) + context_section
        base_messages = build_messages(session, message, system_prompt, chunk_context)
        checks = _SolveChecks(self, session, message, meter)

        # Attempt 1 and 2: quant_agent model
        model = self._model_config.litellm_model("quant_agent")
        temperature = self._model_config.role("quant_agent").temperature
        messages = base_messages
        prev_answer: str | None = None

        for attempt in range(_MAX_RETRIES):
            if attempt > 0 and prev_answer:
                # Inject mismatch context so the model knows to reconsider
                messages = base_messages + [
                    {"role": "assistant", "content": checks.retry_note(prev_answer)}
                ]

            if tools is not None:
                full_text = await tools.reply(
                    meter, model=model, messages=messages, temperature=temperature,
                    obs_meta=obs_meta,
                )
                tokens = [full_text]
            else:
                response = meter.stream(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    **(obs_meta or {}),
                )
                tokens, full_text = await collect_stream(response)
            prev_answer = self.extract_final_answer(full_text)

            if await checks.passes(full_text, prev_answer or ""):
                for token in tokens:
                    yield Token(text=token)
                yield checks.report()
                yield Verified(passed=True)
                return

        # Attempt 3: escalation model (once)
        escalation_model = self._model_config.litellm_model("math_escalation")
        escalation_messages = base_messages + [
            {
                "role": "assistant",
                "content": (
                    "[Previous attempts could not be verified. Applying deeper reasoning.]"
                ),
            }
        ]
        escalation_temperature = self._model_config.role("math_escalation").temperature
        if tools is not None:
            full_text = await tools.reply(
                meter, model=escalation_model, messages=escalation_messages,
                temperature=escalation_temperature,
            )
            tokens = [full_text]
        else:
            response = meter.stream(
                model=escalation_model,
                messages=escalation_messages,
                temperature=escalation_temperature,
            )
            tokens, full_text = await collect_stream(response)
        prev_answer = self.extract_final_answer(full_text)
        passed = await checks.passes(full_text, prev_answer or "")
        for token in tokens:
            yield Token(text=token)
        yield checks.report()
        # Degradation: emit best reasoning with a failed Verified when the last check failed
        yield Verified(passed=passed)

    # ── helpers (designed as seams for unit test patching) ────────────────────

    async def _retrieve(self, session: Session, message: str) -> list[CorpusChunk]:
        """
        Fetch grounding notes from the Corpus Service. Fails open: an unreachable
        corpus costs the turn its grounding, not the answer.
        """
        try:
            return await self._retrieve_corpus.retrieve(
                query=message,
                candidate_id=session.candidate_id,
                school_id="",
                test_type=session.test_type,
                top_k=3,
            )
        except Exception:
            _log.warning("corpus retrieval failed; answering without notes", exc_info=True)
            return []

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
        return await self._verify_math.verify(
            expression=claimed_answer,
            expected=claimed_answer,
        )

    def _compact(self, session: Session, meter: TurnMeter) -> tuple[Session, str]:
        """
        The session as the model will see it, and the prompt section that stands in for the
        turns that were folded. A conversation under the budget comes back as it was.
        """
        if self._context is None:
            return session, ""
        plan = self._context.plan(session.history)
        meter.annotate(
            context_messages_folded=plan.folded,
            context_messages_sent=len(plan.recent),
        )
        if not plan.folded:
            return session, ""
        return session.model_copy(update={"history": plan.recent}), plan.section()

    async def _formalize(
        self,
        session: Session,
        message: str,
        meter: TurnMeter,
    ) -> str | None:
        """
        The one extra short model call of the independent check: the problem written as an
        expression or equation, from the problem alone (the model never sees an answer). None
        when the model says the problem has no numeric answer, wrote no `<expr>` tag, or the call
        failed: a failed check never fails the turn. This seam is patched in tests.
        """
        version = self._model_config.prompt_version(_AGENT, "formalize")
        messages = build_messages(
            session, message, read_prompt(_AGENT, "formalize", version)
        )
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
    ) -> events.IndependentCheck:
        """Compute `expression` with the solver and compare it with the claimed answer."""
        return await independent_check.compare(self._verify_math, expression, claimed_answer)
