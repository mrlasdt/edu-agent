"""
The production Writing rubric grader (Task 1 letter and Task 2 essay).

Several samples of one model score the essay on the four criteria; the per-criterion median is
taken, deterministic caps from `checks.py` are applied, and the overall band is computed in code.
The essay reaches the model as delimited data, and a failure is an explicit error, never a
fabricated band.
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field

from services.gateway.src.gateway.ielts import llm
from services.gateway.src.gateway.ielts.checks import (
    TASK_RULES,
    EssayChecks,
    cap_task_failure,
    check_essay,
)
from services.gateway.src.gateway.ielts.models import Flag
from services.gateway.src.gateway.ielts.scoring import (
    CRITERIA_BY_TASK,
    MAX_ESTIMATED_BAND,
    GradeParseError,
    GradeSample,
    median_band,
    parse_sample,
    round_overall,
)
from shared.src.shared.prompts import read_prompt

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
    task: str = "T2"

    @property
    def overall(self) -> float:
        return round_overall(self.criteria)

    @property
    def sample_overalls(self) -> list[float]:
        """Each sample's overall band under the same caps as the final bands."""
        criteria = CRITERIA_BY_TASK[self.task]
        ceiling = {c: min(self.caps.get(c, 9.0), MAX_ESTIMATED_BAND) for c in criteria}
        return [round_overall({c: min(s[c], ceiling[c]) for c in criteria}) for s in self.samples]


def grade_request(
    prompt: str, essay: str, system: str, task: str = "T2"
) -> list[dict[str, str]]:
    """Prompt and essay are data between markers; the id is derived from the essay itself."""
    marker = hashlib.sha256(essay.encode()).hexdigest()[:8]
    label = TASK_RULES[task].label
    user = (
        f"Task prompt (data):\n<<<PROMPT {marker}>>>\n{prompt}\n<<<END PROMPT {marker}>>>\n\n"
        f"Candidate {label} (data):\n<<<ESSAY {marker}>>>\n{essay}\n<<<END ESSAY {marker}>>>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def _one_sample(
    completer: llm.Completer, messages: list[dict[str, str]], temperature: float,
    trace_id: str, version: str, task: str = "T2",
) -> GradeSample:
    last: GradeParseError | None = None
    for _ in range(PARSE_ATTEMPTS):
        text = await completer(
            llm.GRADER_ROLE, messages, temperature=temperature, max_tokens=GRADER_MAX_TOKENS,
            trace_id=trace_id, prompt_version=version,
        )
        try:
            return parse_sample(text, task)
        except GradeParseError as exc:
            last = exc
    assert last is not None
    raise last


async def grade_essay(
    prompt: str, essay: str, *, completer: llm.Completer, samples: int = 3, trace_id: str = "",
    task: str = "T2",
) -> GradeOutcome:
    rules = TASK_RULES[task]
    criteria = CRITERIA_BY_TASK[task]
    checks = check_essay(prompt, essay, task)
    if checks.blocked:
        raise EssayNotMarkableError(checks.blocked.message)

    version = llm.prompt_version(llm.GRADER_ROLE, task)
    system = read_prompt("ielts_grader", f"writing_{task.lower()}", version)
    messages = grade_request(prompt, essay, system, task)
    temperature = llm.role_temperature(llm.GRADER_ROLE)
    results = await asyncio.gather(
        *[
            _one_sample(completer, messages, temperature, trace_id, version, task)
            for _ in range(samples)
        ],
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
            f"The marking model is unavailable right now. Your {rules.label} is saved; try marking"
            " again."
            if unavailable
            else f"The marking model returned unusable replies. Your {rules.label} is saved; try"
            " again."
        )

    median = {c: median_band([s.bands[c] for s in good]) for c in criteria}
    flags = list(checks.flags)
    caps = dict(checks.caps)
    if sum(s.off_topic for s in good) * 2 > len(good) and not checks.has("possibly_off_topic"):
        flags.append(Flag(
            code="possibly_off_topic", severity="warn",
            message=f"The marker judged the {rules.label} not to answer the question;"
            f" {rules.criterion_name} is capped at 4 and the other criteria at 5.5.",
        ))
        cap_task_failure(caps, task)
    if sum(s.addressed_to_marker for s in good) * 2 > len(good) and not checks.has(
        "addressed_to_marker"
    ):
        flags.append(Flag(
            code="addressed_to_marker", severity="warn",
            message="The response contains text aimed at the marker rather than the question."
            f" It was not followed; {rules.criterion_name} is capped at 5.",
        ))
        caps[rules.criterion] = min(caps.get(rules.criterion, 9.0), 5.0)

    final = {c: min(median[c], caps.get(c, 9.0), MAX_ESTIMATED_BAND) for c in criteria}
    if any(median[c] > MAX_ESTIMATED_BAND for c in criteria):
        flags.append(Flag(
            code="top_of_scale_capped", severity="info",
            message=f"Estimates above band {MAX_ESTIMATED_BAND:g} are not supported; capped.",
        ))
    outcome = GradeOutcome(
        criteria=final, median=median, samples=[s.bands for s in good], spread=0.0,
        n_failed=len(results) - len(good), checks=checks, caps=caps, flags=flags,
        rationale=good[0].rationale, task=task,
    )
    overalls = outcome.sample_overalls
    outcome.spread = max(overalls) - min(overalls)
    return outcome
