"""The report the IELTS Writing mock returns, shared by the API and the web app."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DISCLAIMER = "Mock score: an estimate, not an official IELTS result."


class Flag(BaseModel):
    code: str
    severity: Literal["info", "warn", "block"]
    message: str


class Evidence(BaseModel):
    quote: str
    start: int
    end: int
    para: int
    effect: Literal["supports", "weakens"]
    cut: bool = False  # the model's quote was longer than the limit, so this is its opening words


class Action(BaseModel):
    do: str
    why: str


class ParagraphRole(BaseModel):
    para: int
    job: str
    one_central_idea: bool | None = None


class CriterionReport(BaseModel):
    id: str
    name: str
    band: float
    sample_bands: list[float] = Field(default_factory=list)
    capped_from: float | None = None  # the model's median when a rule capped the band
    verdict: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    holding_back: str = ""
    to_reach_next_band: str = ""
    actions: list[Action] = Field(default_factory=list)
    paragraph_map: list[ParagraphRole] = Field(default_factory=list)  # CC only
    error_table: dict[str, int] = Field(default_factory=dict)  # GRA only: error type -> count


class ErrorSpan(BaseModel):
    start: int
    end: int
    quote: str
    type: str
    note: str
    suggestion: str


class TaskCheck(BaseModel):
    check: str
    passed: bool | None  # None = not assessed (the feedback step was unavailable)


class Priority(BaseModel):
    priority: int
    criterion: str
    do: str
    why: str


class Calibration(BaseModel):
    status: Literal["uncalibrated", "calibrated"]
    label: str
    n: int | None = None
    tier: str | None = None
    within_half_band: float | None = None
    note: str = ""


class EstimatedBand(BaseModel):
    overall: float
    range: tuple[float, float]
    confidence: Literal["low", "medium", "high"]
    samples: int
    sample_spread: float
    calibration: Calibration


class Timing(BaseModel):
    allowed_s: int
    used_s: int
    late: bool


class PreviousAttempt(BaseModel):
    overall_delta: float
    criteria_deltas: dict[str, float]


class ModelInfo(BaseModel):
    grader_models: list[str]
    grader_prompt: str
    feedback_prompt: str


class Report(BaseModel):
    status: Literal["graded", "ungraded"]
    task: Literal["T2"] = "T2"
    word_count: int
    timing: Timing | None = None
    estimated_band: EstimatedBand | None = None
    criteria: list[CriterionReport] = Field(default_factory=list)
    error_spans: list[ErrorSpan] = Field(default_factory=list)
    task_checks: list[TaskCheck] = Field(default_factory=list)
    top_priorities: list[Priority] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    feedback_status: Literal["complete", "partial", "unavailable"] = "unavailable"
    vs_previous_attempt: PreviousAttempt | None = None
    ungraded_reason: str | None = None
    prompt_versions: ModelInfo | None = None
    disclaimer: str = DISCLAIMER


class QuestionResult(BaseModel):
    """One Reading question as marked: what was asked, answered, and the sentence behind the key."""

    number: int
    id: str
    passage: str  # the passage title
    task_type: str
    task_label: str
    prompt: str
    paragraph: int | None = None
    answer: str  # as the learner gave it ("" when left blank)
    answer_label: str
    key: str
    key_label: str
    correct: bool
    evidence: str


class TypeScore(BaseModel):
    task_type: str
    label: str
    correct: int
    total: int


class ReadingReport(BaseModel):
    """Marked in code from the stored answers. Carries everything the coach digest needs."""

    status: Literal["graded"] = "graded"
    module: Literal["ielts_reading"] = "ielts_reading"
    set_id: str
    raw: int
    total: int
    scaled_40: int  # raw scaled to the 40-mark band table
    band: float
    timing: Timing
    by_type: list[TypeScore]
    questions: list[QuestionResult]
    band_note: str
    set_note: str
    keys_note: str
    band_source_url: str
    timing_source_url: str
    disclaimer: str = DISCLAIMER
