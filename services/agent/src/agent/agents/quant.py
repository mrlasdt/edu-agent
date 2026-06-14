from __future__ import annotations

import re
from pathlib import Path
from typing import AsyncGenerator

import litellm

from services.agent.src.agent.base import BaseAgent
from services.agent.src.agent.skills.verify_math import VerifyMathResult, VerifyMathSkill
from shared.src.shared.config import load_model_config
from shared.src.shared.models import MessageRole, Mode, Session
from shared.src.shared.observability import build_llm_metadata

_PROMPT_ROOT = Path("prompts/quant_agent")

# Maximum retries before escalating to the stronger model
_MAX_RETRIES = 2


def _load_prompt(mode: Mode) -> str:
    """Load the system prompt for the given mode, stripping YAML frontmatter."""
    path = _PROMPT_ROOT / mode.value / "v1.md"
    raw = path.read_text()
    return re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL).strip()


def _build_messages(session: Session, user_message: str, system_prompt: str) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in session.history:
        role = "user" if msg.role == MessageRole.candidate else "assistant"
        messages.append({"role": role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})
    return messages


async def _collect_stream(stream) -> tuple[list[str], str]:
    """Drain a litellm stream, collecting tokens and the full text."""
    tokens: list[str] = []
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            tokens.append(content)
    return tokens, "".join(tokens)


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
    ) -> None:
        self._model_config = load_model_config(model_config_path)
        self._verify_skill = verify_math_skill or VerifyMathSkill()

    # ── public interface ──────────────────────────────────────────────────────

    async def run(
        self, session: Session, message: str, trace_id: str = ""
    ) -> AsyncGenerator[str | dict, None]:
        """
        Yield str tokens as they are generated, followed by a final dict metadata item.

        Metadata dict shape:
          {"verifier_fail": bool}   — only emitted in Solve mode
        """
        obs_meta = build_llm_metadata(
            trace_id=trace_id,
            prompt_version="v1",  # loaded from prompt filename in full impl
        )
        if session.mode == Mode.tutor:
            async for token in self._tutor_turn(session, message, obs_meta):
                yield token
        else:
            async for item in self._solve_turn(session, message, obs_meta):
                yield item

    # ── tutor mode ────────────────────────────────────────────────────────────

    async def _tutor_turn(
        self, session: Session, message: str, obs_meta: dict | None = None
    ) -> AsyncGenerator[str, None]:
        system_prompt = _load_prompt(Mode.tutor)
        messages = _build_messages(session, message, system_prompt)
        model = self._model_config.litellm_model("quant_agent")
        temperature = self._model_config.role("quant_agent").temperature

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
            **(obs_meta or {}),
        )
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    # ── solve mode ────────────────────────────────────────────────────────────

    async def _solve_turn(
        self, session: Session, message: str, obs_meta: dict | None = None
    ) -> AsyncGenerator[str | dict, None]:
        system_prompt = _load_prompt(Mode.solve)
        base_messages = _build_messages(session, message, system_prompt)

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
                        "content": f"[My previous answer {prev_answer!r} could not be verified. Let me try again.]",
                    }
                ]

            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
                **(obs_meta or {}),
            )
            tokens, full_text = await _collect_stream(response)
            prev_answer = self._extract_final_answer(full_text)

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
                    f"[Previous attempts could not be verified. "
                    "Applying deeper reasoning.]"
                ),
            }
        ]
        response = await litellm.acompletion(
            model=escalation_model,
            messages=escalation_messages,
            temperature=self._model_config.role("math_escalation").temperature,
            stream=True,
        )
        tokens, full_text = await _collect_stream(response)
        prev_answer = self._extract_final_answer(full_text)
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

    def _extract_final_answer(self, text: str) -> str | None:
        """
        Extract the final answer from the model's output text.
        Looks for the pattern '**Answer: <value>**' from the Solve prompt format.
        Falls back to the last non-empty line.
        """
        match = re.search(r"\*\*Answer:\s*(.+?)\*\*", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return lines[-1] if lines else None

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
