"""The agent service's settings: only what the agent process (Quant and AW agents) reads."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from shared.src.shared.config import LangfuseSettings, LlmSettings, TurnLogSettings


class AgentSettings(LlmSettings, TurnLogSettings, LangfuseSettings):
    corpus_service_url: str = "http://localhost:8001"
    math_verifier_url: str = "http://localhost:8090"
    # Math tools for the Quant agent (math_evaluate, math_solve, math_compare): the model
    # computes with sympy instead of in its head. Off by default: a tool turn is not streamed and
    # costs more tokens (README, "Math tools"). true = the tool loop.
    math_tools_enabled: bool = False
    # Context control for the Quant agent (services/agent/src/agent/context.py): in a long session
    # the first message (the problem) is always sent, the last `context_keep_messages` messages are
    # sent verbatim, and older turns are dropped, `context_fold_step` at a time. Shorter
    # conversations are not touched.
    context_compaction_enabled: bool = True
    context_keep_messages: int = 8
    context_fold_step: int = 4
    # How the Quant tutor gets its reference notes. "upfront": every turn searches the corpus
    # with the student's message and puts the top notes in front of the model. "just_in_time":
    # the model calls `notes_search` itself when a hint needs a fact (tutor mode only; solve
    # mode and the AW agent keep the up-front search). A just-in-time turn is not streamed.
    retrieval_mode: Literal["upfront", "just_in_time"] = "upfront"


@lru_cache
def get_settings() -> AgentSettings:
    return AgentSettings()
