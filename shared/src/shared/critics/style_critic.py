"""
Style critic — checks AW essay for canonical structural rubric (PEEL/MEAL).
Runs post-generation using a Haiku-tier model call.
Returns (passed: bool, feedback: str).
"""

from __future__ import annotations

from typing import Any

from shared.src.shared.config import load_model_config
from shared.src.shared.observability import build_llm_metadata
from shared.src.shared.prompts import read_prompt
from shared.src.shared.turn_meter import TurnMeter

_PROMPT_AGENT = "style_critic"
_PROMPT_MODE = "rubric"



async def run_style_critic(
    essay_text: str,
    model_config_path: str | None = None,
    trace_id: str = "",
    meter: TurnMeter | None = None,
) -> tuple[bool, str]:
    """
    Run the style critic on an essay.
    Returns (passed, feedback) where feedback is empty when passed=True.
    With a `meter`, the call is counted in the turn's measurement.
    """
    cfg = load_model_config(model_config_path)
    model = cfg.litellm_model("style_critic")
    temperature = cfg.role("style_critic").temperature
    version = cfg.prompt_version(_PROMPT_AGENT, _PROMPT_MODE)

    call_args: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": read_prompt(_PROMPT_AGENT, _PROMPT_MODE, version)},
            {"role": "user", "content": essay_text},
        ],
        "temperature": temperature,
        **build_llm_metadata(trace_id=trace_id, prompt_version=version),
    }
    response = await (meter or TurnMeter.detached()).complete(**call_args)
    result = response.choices[0].message.content.strip()
    if result.upper().startswith("PASS"):
        return True, ""
    return False, result
