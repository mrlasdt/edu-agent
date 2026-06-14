from __future__ import annotations

import json
from dataclasses import dataclass

import litellm

from shared.src.shared.config import load_model_config

_RUBRIC = """You are an expert GRE Analytical Writing scorer calibrated to the ETS 6-point rubric.

Score the candidate essay on the following scale:
  6 – Outstanding: insightful, well-developed, precise language, varied syntax, no notable errors
  5 – Strong: well-developed, generally precise, minor errors
  4 – Adequate: competent but analysis is limited or language uneven
  3 – Limited: flawed analysis, limited development, frequent errors
  2 – Seriously flawed: little relevant analysis, serious language problems
  1 – Fundamentally deficient: little understanding of the task, pervasive errors

You have been provided anchor essays at specific score levels to calibrate your judgment.

Respond with valid JSON only:
{"score": <float 0.0-6.0 in 0.5 increments>, "rationale": "<2-3 sentences>"}
"""


@dataclass
class AWResult:
    item_id: str
    score: float
    rationale: str


async def grade_aw_item(item: dict, model_config_path: str | None = None) -> AWResult:
    """Grade an AW golden-set item using an LLM judge with anchor essays."""
    item_id = item.get("id", "unknown")
    prompt = item.get("prompt", "")
    essay = item.get("candidate_essay", "")
    anchor_essays: dict = item.get("anchor_essays", {})

    raw = await _call_aw_judge_llm(
        prompt=prompt,
        essay=essay,
        anchor_essays=anchor_essays,
        model_config_path=model_config_path,
    )
    return AWResult(
        item_id=item_id,
        score=float(raw.get("score", 0.0)),
        rationale=str(raw.get("rationale", "")),
    )


async def _call_aw_judge_llm(
    prompt: str,
    essay: str,
    anchor_essays: dict,
    model_config_path: str | None = None,
) -> dict:
    """LLM judge call with rubric + anchor essays in context. Seam for test patching."""
    cfg = load_model_config(model_config_path)
    model = cfg.litellm_model("aw_judge_nightly")

    anchor_text = "\n\n".join(
        f"--- Anchor essay (score {score}) ---\n{text}"
        for score, text in sorted(anchor_essays.items())
    )

    user_content = (
        f"## Task prompt\n{prompt}\n\n"
        f"## Anchor essays for calibration\n{anchor_text}\n\n"
        f"## Candidate essay to score\n{essay}"
    )

    response = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": _RUBRIC},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        stream=False,
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        return {"score": 0.0, "rationale": "parse error"}
