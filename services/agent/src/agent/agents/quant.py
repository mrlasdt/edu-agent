from __future__ import annotations

import re
from pathlib import Path
from typing import AsyncGenerator

import litellm

from services.agent.src.agent.base import BaseAgent
from shared.src.shared.config import load_model_config
from shared.src.shared.models import MessageRole, Mode, Session

_PROMPT_ROOT = Path("prompts/quant_agent")


def _load_prompt(mode: Mode) -> str:
    """Load the system prompt for the given mode, stripping YAML frontmatter."""
    path = _PROMPT_ROOT / mode.value / "v1.md"
    raw = path.read_text()
    # Strip YAML frontmatter (--- ... ---)
    stripped = re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL)
    return stripped.strip()


def _build_messages(session: Session, user_message: str, system_prompt: str) -> list[dict]:
    """Assemble the full message list for LiteLLM from session history + new message."""
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in session.history:
        role = "user" if msg.role == MessageRole.candidate else "assistant"
        messages.append({"role": role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})
    return messages


class QuantAgent(BaseAgent):
    """
    GRE Quantitative Reasoning agent.
    Registered with the AgentRegistry as ("gre", "quant").

    Tutor mode: scaffolds toward the answer via Socratic hints (never reveals the answer).
    Solve mode: returns a fully worked step-by-step solution, subject to verify-math
                Skill validation (wired in issue 05).
    """

    def __init__(self, model_config_path: str | None = None) -> None:
        self._model_config = load_model_config(model_config_path)

    async def run(self, session: Session, message: str) -> AsyncGenerator[str, None]:
        system_prompt = _load_prompt(session.mode)
        messages = _build_messages(session, message, system_prompt)
        model = self._model_config.litellm_model("quant_agent")
        temperature = self._model_config.role("quant_agent").temperature

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )

        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
