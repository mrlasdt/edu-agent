"""
The production Writing Task 2 rubric grader.

Several samples of one model score the essay on the four criteria; the per-criterion median is
taken, deterministic caps from `checks.py` are applied, and the overall band is computed in code.
The essay reaches the model as delimited data, and a failure is an explicit error, never a
fabricated band.
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field

from services.agent.src.agent.ielts import llm
from services.agent.src.agent.ielts.checks import EssayChecks, cap_task_failure, check_essay
from services.agent.src.agent.ielts.models import Flag
from services.agent.src.agent.ielts.scoring import (
    CRITERIA,
    MAX_ESTIMATED_BAND,
    GradeParseError,
    GradeSample,
    median_band,
    parse_sample,
    round_overall,
)

GRADER_MAX_TOKENS = 2000  # a cap: leaves room for models that reason before they answer
PARSE_ATTEMPTS = 2


class GradingError(RuntimeError):
    """The essay could not be given a band; `reason` is safe to show the candidate."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class EssayNotMarkableError(GradingError):
    pass


@dataclass
class GradeOutcome:
    criteria: dict[str, float]  # final bands: sample median, then caps
    median: dict[str, float]  # before caps
    samples: list[dict[str, float]]
    spread: float  # max - min of the sample overall bands
    n_failed: int
    checks: EssayChecks
    caps: dict[str, float] = field(default_factory=dict)
    flags: list[Flag] = field(default_factory=list)
    rationale: str = ""

    @property
    def overall(self) -> float:
        return round_overall(self.criteria)

    @property
    def sample_overalls(self) -> list[float]:
        """Each sample's overall band under the same caps as the final bands."""
        ceiling = {c: min(self.caps.get(c, 9.0), MAX_ESTIMATED_BAND) for c in CRITERIA}
        return [round_overall({c: min(s[c], ceiling[c]) for c in CRITERIA}) for s in self.samples]


def grade_request(prompt: str, essay: str, system: str) -> list[dict[str, str]]:
    """Prompt and essay are data between markers; the id is derived from the essay itself."""
    marker = hashlib.sha256(essay.encode()).hexdigest()[:8]
    user = (
        f"Task prompt (data):\n<<<PROMPT {marker}>>>\n{prompt}\n<<<END PROMPT {marker}>>>\n\n"
        f"Candidate essay (data):\n<<<ESSAY {marker}>>>\n{essay}\n<<<END ESSAY {marker}>>>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def _one_sample(
    completer: llm.Completer, messages: list[dict[str, str]], temperature: float,
    trace_id: str, version: str,
) -> GradeSample:
    last: GradeParseError | None = None
    for _ in range(PARSE_ATTEMPTS):
        text = await completer(
            llm.GRADER_ROLE, messages, temperature=temperature, max_tokens=GRADER_MAX_TOKENS,
            trace_id=trace_id, prompt_version=version,
        )
        try:
            return parse_sample(text)
        except GradeParseError as exc:
            last = exc
    assert last is not None
    raise last


async def grade_essay(
    prompt: str, essay: str, *, completer: llm.Completer, samples: int = 3, trace_id: str = "",
) -> GradeOutcome:
    checks = check_essay(prompt, essay)
    if checks.blocked:
        raise EssayNotMarkableError(checks.blocked.message)

    version = llm.prompt_version(llm.GRADER_ROLE)
    system = llm.load_prompt("ielts_grader", "writing_t2", version)
    messages = grade_request(prompt, essay, system)
    temperature = llm.role_temperature(llm.GRADER_ROLE)
    results = await asyncio.gather(
        *[_one_sample(completer, messages, temperature, trace_id, version) for _ in range(samples)],
        return_exceptions=True,
    )
    for r in results:  # budget guards and programming errors are not a failed sample
        if isinstance(r, BaseException) and not isinstance(
            r, GradeParseError | llm.ModelUnavailableError
        ):
            raise r
    good = [r for r in results if isinstance(r, GradeSample)]
    if not good:
        unavailable = any(isinstance(r, llm.ModelUnavailableError) for r in results)
        raise GradingError(
            "The marking model is unavailable right now. Your essay is saved; try marking again."
            if unavailable
            else "The marking model returned unusable replies. Your essay is saved; try again."
        )

    median = {c: median_band([s.bands[c] for s in good]) for c in CRITERIA}
    flags = list(checks.flags)
    caps = dict(checks.caps)
    if sum(s.off_topic for s in good) * 2 > len(good) and not checks.has("possibly_off_topic"):
        flags.append(Flag(
            code="possibly_off_topic", severity="warn",
            message="The marker judged the essay not to answer the question; Task Response is"
            " capped at 4 and the other criteria at 5.5.",
        ))
        cap_task_failure(caps)
    if sum(s.addressed_to_marker for s in good) * 2 > len(good) and not checks.has(
        "addressed_to_marker"
    ):
        flags.append(Flag(
            code="addressed_to_marker", severity="warn",
            message="The response contains text aimed at the marker rather than the question."
            " It was not followed; Task Response is capped at 5.",
        ))
        caps["TR"] = min(caps.get("TR", 9.0), 5.0)

    final = {c: min(median[c], caps.get(c, 9.0), MAX_ESTIMATED_BAND) for c in CRITERIA}
    if any(median[c] > MAX_ESTIMATED_BAND for c in CRITERIA):
        flags.append(Flag(
            code="top_of_scale_capped", severity="info",
            message=f"Estimates above band {MAX_ESTIMATED_BAND:g} are not supported; capped.",
        ))
    outcome = GradeOutcome(
        criteria=final, median=median, samples=[s.bands for s in good], spread=0.0,
        n_failed=len(results) - len(good), checks=checks, caps=caps, flags=flags,
        rationale=good[0].rationale,
    )
    overalls = outcome.sample_overalls
    outcome.spread = max(overalls) - min(overalls)
    return outcome
