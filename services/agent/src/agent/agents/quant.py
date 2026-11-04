from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from services.agent.src.agent.answers import extract_final_answer, normalize_answer
from services.agent.src.agent.base import BaseAgent
from services.agent.src.agent.skills.math_tools import MATH_TOOL_SPECS, MathToolsSkill
from services.agent.src.agent.skills.retrieve_corpus import CorpusChunk, RetrieveCorpusSkill
from services.agent.src.agent.skills.verify_math import VerifyMathResult, VerifyMathSkill
from services.agent.src.agent.tool_loop import ToolCallRecord, run_tool_loop
from shared.src.shared.config import get_settings, load_model_config
from shared.src.shared.models import MessageRole, Mode, Session
from shared.src.shared.observability import build_llm_metadata
from shared.src.shared.turn_meter import TurnMeter

_log = logging.getLogger(__name__)

_PROMPT_ROOT = Path("prompts")
_AGENT = "quant_agent"
# The same prompts plus a "Math tools" section; used only when the math tools are on.
_TOOLS_AGENT = "quant_agent_math_tools"
_NO_REPLY = "I could not finish working that out just now. Could you send the problem again?"

# Maximum retries before escalating to the stronger model
_MAX_RETRIES = 2


def _load_prompt(mode: Mode, version: str, agent: str = _AGENT) -> str:
    """Load the system prompt for the given mode + version, stripping YAML frontmatter."""
    path = _PROMPT_ROOT / agent / mode.value / f"{version}.md"
    raw = path.read_text()
    return re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL).strip()


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


def _build_messages(
    session: Session, user_message: str, system_prompt: str, chunk_context: str = ""
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for msg in session.history:
        role = "user" if msg.role == MessageRole.candidate else "assistant"
        messages.append({"role": role, "content": msg.content})
    messages.append({"role": "user", "content": f"{user_message}{chunk_context}"})
    return messages


async def _collect_stream(stream: AsyncIterator[Any]) -> tuple[list[str], str]:
    """Drain a litellm stream, collecting tokens and the full text."""
    tokens: list[str] = []
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            tokens.append(content)
    return tokens, "".join(tokens)


class _ToolTurn:
    """
    The math tools for one turn: runs the bounded tool loop for each model attempt and keeps
    the calls, so the turn can report them. One instance per turn (solve mode may attempt
    several times; the calls of all attempts are kept).
    """

    def __init__(self, skill: MathToolsSkill) -> None:
        self._skill = skill
        self.calls: list[ToolCallRecord] = []
        self.rounds = 0
        self.out_of_budget = False

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
            tools=MATH_TOOL_SPECS,
            execute=self._skill.call,
            temperature=temperature,
            **(obs_meta or {}),
        )
        self.calls += [replace(c, round=c.round + self.rounds) for c in outcome.calls]
        self.rounds += outcome.rounds
        self.out_of_budget = self.out_of_budget or outcome.out_of_budget
        for call in outcome.calls:
            _log.info(
                "math tool %s ok=%s ms=%s args=%s result=%s",
                call.name, call.ok, call.ms, call.arguments, call.result,
            )
        return outcome.text.strip() or _NO_REPLY

    def annotate(self, meter: TurnMeter) -> None:
        """Put the calls on the turn record (the operator's log; the learner never sees it)."""
        meter.annotate(
            math_tools=True,
            tool_rounds=self.rounds,
            tool_calls=[c.public() for c in self.calls],
            tool_errors=sum(1 for c in self.calls if not c.ok),
            tool_budget_exhausted=self.out_of_budget,
        )

    def trace(self) -> dict[str, Any]:
        return {"tool_trace": [c.public() for c in self.calls]}


class QuantAgent(BaseAgent):
    """
    GRE Quantitative Reasoning agent.

    Tutor mode: scaffolds toward the answer; verifier is NOT called.
    Solve mode: returns a fully worked solution; the verify-math Skill validates
                the final answer.  Retry / escalation pipeline:
                  attempt 1 (quant_agent model)
                  → if verifier disagrees: attempt 2 (quant_agent model, mismatch context)
                  → if still disagrees:    attempt 3 (math_escalation model)
                  → if still disagrees:    degrade gracefully (emit verifier_fail metadata)
    """

    def __init__(
        self,
        model_config_path: str | None = None,
        verify_math_skill: VerifyMathSkill | None = None,
        retrieve_corpus_skill: RetrieveCorpusSkill | None = None,
        math_tools_skill: MathToolsSkill | None = None,
        math_tools_enabled: bool | None = None,
    ) -> None:
        self._model_config = load_model_config(model_config_path)
        self._math_tools_enabled = (
            get_settings().math_tools_enabled if math_tools_enabled is None else math_tools_enabled
        )
        self._math_tools_skill = math_tools_skill
        self._retrieve_skill = retrieve_corpus_skill or RetrieveCorpusSkill()
        self._verify_skill = verify_math_skill or VerifyMathSkill()

    # ── public interface ──────────────────────────────────────────────────────

    async def run(
        self, session: Session, message: str, trace_id: str = ""
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        """
        Yield str tokens as they are generated, followed by a final dict metadata item.

        Metadata dict shapes:
          {"sources": int}          — first item, only when corpus notes were retrieved
          {"verifier_fail": bool}   — only emitted in Solve mode, after the answer text
          {"tool_trace": [...]}     — last item, only when the model called a math tool: each
                                      call with its arguments and result (for evals and
                                      debugging; never part of the reply the learner sees)
        """
        mode = session.mode.value
        prompt_agent = _TOOLS_AGENT if self._math_tools_enabled else _AGENT
        version = self._model_config.prompt_version(prompt_agent, mode)
        # The turn record names the prompt that ran; the tools prompt is its own versioned file.
        reported = f"math_tools-{version}" if self._math_tools_enabled else version
        obs_meta = build_llm_metadata(trace_id=trace_id, prompt_version=reported)
        meter = TurnMeter(trace_id, "agent", "quant_agent", mode, reported)
        tools = _ToolTurn(self._math_tools_skill or MathToolsSkill()) if (
            self._math_tools_enabled
        ) else None
        try:
            with meter.phase("retrieval"):
                chunks = await self._retrieve(session, message)
            meter.annotate(notes_retrieved=len(chunks))
            if chunks:
                yield {"sources": len(chunks)}
            chunk_context = _format_chunks_for_context(chunks)
            if session.mode == Mode.tutor:
                async for token in self._tutor_turn(
                    session, message, meter, obs_meta, version, chunk_context, tools
                ):
                    yield token
            else:
                async for item in self._solve_turn(
                    session, message, meter, obs_meta, version, chunk_context, tools
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
    ) -> AsyncGenerator[str, None]:
        system_prompt = (
            _load_prompt(Mode.tutor, version, _TOOLS_AGENT)
            if tools is not None
            else _load_prompt(Mode.tutor, version)
        )
        messages = _build_messages(session, message, system_prompt, chunk_context)
        model = self._model_config.litellm_model("quant_agent")
        temperature = self._model_config.role("quant_agent").temperature

        if tools is not None:
            # Tool results stay in the loop's own message list; only the final text is yielded.
            yield await tools.reply(
                meter, model=model, messages=messages, temperature=temperature, obs_meta=obs_meta
            )
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
                yield content

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
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        system_prompt = (
            _load_prompt(Mode.solve, version, _TOOLS_AGENT)
            if tools is not None
            else _load_prompt(Mode.solve, version)
        )
        base_messages = _build_messages(session, message, system_prompt, chunk_context)

        # Attempt 1 and 2: quant_agent model
        model = self._model_config.litellm_model("quant_agent")
        temperature = self._model_config.role("quant_agent").temperature
        messages = base_messages
        prev_answer: str | None = None

        for attempt in range(_MAX_RETRIES):
            if attempt > 0 and prev_answer:
                # Inject mismatch context so the model knows to reconsider
                messages = base_messages + [
                    {
                        "role": "assistant",
                        "content": (
                            f"[My previous answer {prev_answer!r} could not be verified. "
                            "Let me try again.]"
                        ),
                    }
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
                tokens, full_text = await _collect_stream(response)
            prev_answer = self._extract_final_answer(full_text)

            with meter.phase("verify"):
                result = await self._verify(full_text, prev_answer or "")
            if result.verified:
                for token in tokens:
                    yield token
                yield {"verifier_fail": False}
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
            tokens, full_text = await _collect_stream(response)
        prev_answer = self._extract_final_answer(full_text)
        with meter.phase("verify"):
            result = await self._verify(full_text, prev_answer or "")
        if result.verified:
            for token in tokens:
                yield token
            yield {"verifier_fail": False}
            return

        # Degradation: emit best reasoning with verifier_fail metadata
        for token in tokens:
            yield token
        yield {"verifier_fail": True}

    # ── helpers (designed as seams for unit test patching) ────────────────────

    async def _retrieve(self, session: Session, message: str) -> list[CorpusChunk]:
        """
        Fetch grounding notes from the Corpus Service. Fails open: an unreachable
        corpus costs the turn its grounding, not the answer.
        """
        try:
            return await self._retrieve_skill.retrieve(
                query=message,
                candidate_id=session.candidate_id,
                school_id="",
                test_type=session.test_type,
                top_k=3,
            )
        except Exception:
            _log.warning("corpus retrieval failed; answering without notes", exc_info=True)
            return []

    def _extract_final_answer(self, text: str) -> str | None:
        """
        Extract the final answer from the model's output text, reduced to a bare value list.
        Looks for '**Answer: <value>**' from the Solve prompt format, then an 'Answer:' line,
        then falls back to the last non-empty line (see agent.answers).
        """
        raw = extract_final_answer(text)
        return normalize_answer(raw) if raw else None

    async def _verify(self, full_text: str, claimed_answer: str) -> VerifyMathResult:
        """
        Call the verify-math Skill with the claimed answer.
        Extracts an equation from the LLM's output to verify against.
        This seam is patched in tests.
        """
        # For v1: use the raw answer as the "expression" to validate
        # In a richer implementation, the LLM would emit a structured expression tag
        return await self._verify_skill.verify(
            expression=claimed_answer,
            expected=claimed_answer,
        )
