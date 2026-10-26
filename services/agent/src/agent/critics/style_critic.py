"""
Style critic — checks AW essay for canonical structural rubric (PEEL/MEAL).
Runs post-generation using a Haiku-tier model call.
Returns (passed: bool, feedback: str).
"""

from __future__ import annotations

import litellm

from shared.src.shared.config import load_model_config
from shared.src.shared.models import Mode
from shared.src.shared.observability import build_llm_metadata

_STYLE_RUBRIC = """You are a strict GRE essay structure reviewer.
Check whether this essay follows the PEEL/MEAL paragraph structure:
  Point   → clear topic sentence
  Evidence/Example → specific supporting detail
  Explanation → analysis of the evidence
  Link    → connection back to thesis

Respond with exactly:
  PASS  — if every body paragraph follows this structure
  FAIL: <brief reason>  — if any body paragraph violates it

Do not suggest rewrites. Only assess structure."""


async def run_style_critic(
    essay_text: str, model_config_path: str | None = None, trace_id: str = ""
) -> tuple[bool, str]:
    """
    Run the style critic on an essay.
    Returns (passed, feedback) where feedback is empty when passed=True.
    """
    cfg = load_model_config(model_config_path)
    model = cfg.litellm_model("style_critic")
    temperature = cfg.role("style_critic").temperature
    version = cfg.prompt_version("aw_agent", Mode.solve.value)

    response = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": _STYLE_RUBRIC},
            {"role": "user", "content": essay_text},
        ],
        temperature=temperature,
        stream=False,
        **build_llm_metadata(trace_id=trace_id, prompt_version=version),
    )
    result = response.choices[0].message.content.strip()
    if result.upper().startswith("PASS"):
        return True, ""
    return False, result
