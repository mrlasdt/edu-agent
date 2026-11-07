"""The agent service's settings: only what the agent process (Quant and AW agents) reads."""
from __future__ import annotations

from functools import lru_cache

from shared.src.shared.config import LangfuseSettings, LlmSettings, TurnLogSettings


class AgentSettings(LlmSettings, TurnLogSettings, LangfuseSettings):
    corpus_service_url: str = "http://localhost:8001"
    math_verifier_url: str = "http://localhost:8090"
    # Math tools for the Quant agent (math_evaluate, math_solve, math_compare): the model
    # computes with sympy instead of in its head. Off by default: a tool turn is not streamed and
    # costs more tokens (README, "Math tools"). true = the tool loop.
    math_tools_enabled: bool = False


@lru_cache
def get_settings() -> AgentSettings:
    return AgentSettings()
