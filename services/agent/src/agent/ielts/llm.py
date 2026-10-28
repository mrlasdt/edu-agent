"""
Model access for the IELTS mock, behind one seam.

`Completer` is what the grader and the feedback generator call. The production completer walks
a model chain through litellm; the eval harness passes its own that calls `litellm.acompletion`
directly so the metered transport (budget ledger, fallbacks, cache) sits underneath unchanged.
"""
from __future__ import annotations

import re
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, Protocol

import litellm

from shared.src.shared.config import get_settings, load_model_config
from shared.src.shared.observability import build_llm_metadata

PROMPT_ROOT = Path("prompts")
GRADER_ROLE = "ielts_grader"
FEEDBACK_ROLE = "ielts_feedback"
_SECRET = re.compile(r"(sk-[\w-]{8,}|Bearer\s+\S+)", re.I)


class ModelUnavailableError(RuntimeError):
    """Every model in the chain failed for a call."""


class Completer(Protocol):
    def __call__(
        self, role: str, messages: list[dict[str, str]], *, temperature: float, max_tokens: int,
        trace_id: str = "", prompt_version: str = "",
    ) -> Awaitable[str]: ...


def load_prompt(agent: str, mode: str, version: str) -> str:
    raw = (PROMPT_ROOT / agent / mode / f"{version}.md").read_text()
    return re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL).strip()


def prompt_version(role: str) -> str:
    return load_model_config(get_settings().model_config_path).prompt_version(role, "writing_t2")


def role_temperature(role: str) -> float:
    return load_model_config(get_settings().model_config_path).role(role).temperature


def model_chain(role: str) -> list[str]:
    """`IELTS_MODEL_CHAIN` (comma separated litellm strings) overrides the role's model."""
    override = [m.strip() for m in get_settings().ielts_model_chain.split(",") if m.strip()]
    if override:
        return override
    return [load_model_config(get_settings().model_config_path).litellm_model(role)]


class LiteLLMCompleter:
    """Try each model of the role's chain in order; the first non-empty reply wins."""

    async def __call__(
        self, role: str, messages: list[dict[str, str]], *, temperature: float, max_tokens: int,
        trace_id: str = "", prompt_version: str = "",
    ) -> str:
        failures: list[str] = []
        extra: dict[str, Any] = (
            build_llm_metadata(trace_id, prompt_version or "v1") if trace_id else {}
        )
        for model in model_chain(role):
            try:
                response = await litellm.acompletion(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, stream=False, **extra,
                )
                text = str(response.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001 - any provider failure moves to the next model
                reason = _SECRET.sub("[redacted]", " ".join(str(exc).split()))[:120]
                failures.append(f"{model}: {type(exc).__name__}: {reason}")
                continue
            if text.strip():
                return text
            failures.append(f"{model}: empty reply")
        raise ModelUnavailableError("; ".join(failures))
