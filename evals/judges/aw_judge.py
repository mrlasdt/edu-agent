from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, Field

from shared.src.shared.config import load_model_config
from shared.src.shared.llm_router import get_router
from shared.src.shared.prompts import read_prompt
from shared.src.shared.structured import (
    StructuredOutputError,
    parse_structured,
    response_format_for,
)

_log = logging.getLogger(__name__)


class AWVerdict(BaseModel):
    """The judge's reply. `score` must be a JSON number on the 0-6 scale, not a string."""

    score: Annotated[float, Field(strict=True, ge=0.0, le=6.0)]
    rationale: str


@dataclass
class AWResult:
    """
    One graded item. A judge reply that failed validation is an *errored* item: `score` is None
    and `error` says why. It is never a 0.0, so it cannot pull an average down; `summarize_aw`
    leaves it out of the average and counts it.
    """

    item_id: str
    score: float | None
    rationale: str
    error: str | None = None


@dataclass
class AWSummary:
    average: float | None  # mean score of the graded items; None when none were graded
    graded: int
    errored: int


def summarize_aw(results: list[AWResult]) -> AWSummary:
    """The average over graded items only, with the number of errored items reported beside it."""
    scores = [r.score for r in results if r.error is None and r.score is not None]
    return AWSummary(
        average=sum(scores) / len(scores) if scores else None,
        graded=len(scores),
        errored=len(results) - len(scores),
    )


async def grade_aw_item(item: dict[str, Any], model_config_path: str | None = None) -> AWResult:
    """Grade an AW golden-set item using an LLM judge with anchor essays."""
    item_id = item.get("id", "unknown")
    prompt = item.get("prompt", "")
    essay = item.get("candidate_essay", "")
    anchor_essays: dict[str, Any] = item.get("anchor_essays", {})

    reply = await _call_aw_judge_llm(
        prompt=prompt,
        essay=essay,
        anchor_essays=anchor_essays,
        model_config_path=model_config_path,
    )
    try:
        verdict = parse_structured(reply, AWVerdict)
    except StructuredOutputError as exc:
        _log.warning("AW judge reply for %s is invalid: %s", item_id, exc)
        return AWResult(item_id=item_id, score=None, rationale="", error=str(exc))
    return AWResult(item_id=item_id, score=verdict.score, rationale=verdict.rationale)


async def _call_aw_judge_llm(
    prompt: str,
    essay: str,
    anchor_essays: dict[str, Any],
    model_config_path: str | None = None,
) -> str | None:
    """
    LLM judge call with rubric + anchor essays in context. Returns the model's raw reply, which
    grade_aw_item validates as an AWVerdict. Seam for test patching.
    """
    cfg = load_model_config(model_config_path)
    model = cfg.litellm_model("aw_judge_nightly")
    rubric = read_prompt("aw_judge", "rubric", cfg.prompt_version("aw_judge", "rubric"))

    anchor_text = "\n\n".join(
        f"--- Anchor essay (score {score}) ---\n{text}"
        for score, text in sorted(anchor_essays.items())
    )

    user_content = (
        f"## Task prompt\n{prompt}\n\n"
        f"## Anchor essays for calibration\n{anchor_text}\n\n"
        f"## Candidate essay to score\n{essay}"
    )

    response = await get_router().acompletion(
        model=model,
        messages=[
            {"role": "system", "content": rubric},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        response_format=response_format_for(model, AWVerdict),
        stream=False,
    )
    content: str | None = response.choices[0].message.content
    return content
