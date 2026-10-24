"""
Provider layer for the pipeline: role -> model via the shared model config, called through
litellm so the eval harness's metered transport (budget ledger, free-first fallback chain)
sits underneath it unchanged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

import litellm

from shared.src.shared.config import ModelConfig, load_model_config
from shared.src.shared.observability import build_llm_metadata

_PROMPT_ROOT = Path("prompts/question_pipeline")
PROMPT_AGENT = "question_pipeline"
# Generous: free routed models spend part of the budget on hidden reasoning, and a reply cut off
# mid-JSON is an unparseable item.
MAX_TOKENS = 2500


class Completer(Protocol):
    """Anything that can answer a chat request for a model role (real or scripted)."""

    def model_for(self, role: str) -> str: ...

    async def complete(self, role: str, messages: list[dict[str, Any]], trace_id: str) -> str: ...


class LiteLLMCompleter:
    def __init__(self, model_config_path: str | None = None) -> None:
        self._config: ModelConfig = load_model_config(model_config_path)

    def model_for(self, role: str) -> str:
        return self._config.litellm_model(role)

    def prompt_version(self, name: str) -> str:
        return self._config.prompt_version(PROMPT_AGENT, name)

    async def complete(self, role: str, messages: list[dict[str, Any]], trace_id: str) -> str:
        response = await litellm.acompletion(
            model=self._config.litellm_model(role),
            messages=messages,
            temperature=self._config.role(role).temperature,
            max_tokens=MAX_TOKENS,
            stream=False,
            **build_llm_metadata(trace_id=trace_id, prompt_version=PROMPT_AGENT),
        )
        return str(response.choices[0].message.content or "")


def load_prompt(name: str, version: str = "v1") -> str:
    """System prompt for a pipeline role, with the YAML frontmatter stripped."""
    raw = (_PROMPT_ROOT / name / f"{version}.md").read_text()
    return re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict[str, Any] | None:
    """First JSON object in `text` (tolerates prose and code fences around it), else None."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None
