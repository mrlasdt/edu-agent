"""The report the IELTS Writing mock returns, shared by the API and the web app."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TaskId = Literal["T1", "T2"]
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
    task: TaskId = "T2"
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


class SectionTranscript(BaseModel):
    """What was said in one Listening recording, shown only after the paper is submitted."""

    id: str
    title: str
    lines: list[str]


class ListeningReport(BaseModel):
    """Listening marked in code. `questions[].passage` is the title of the recording."""

    status: Literal["graded"] = "graded"
    module: Literal["ielts_listening"] = "ielts_listening"
    set_id: str
    raw: int
    total: int
    scaled_40: int
    band: float
    timing: Timing
    by_type: list[TypeScore]
    questions: list[QuestionResult]
    transcripts: list[SectionTranscript]
    band_note: str
    set_note: str
    keys_note: str
    band_source_url: str
    timing_source_url: str
    disclaimer: str = DISCLAIMER


class WritingModuleReport(BaseModel):
    """
    The Writing module: a Task 1 letter and a Task 2 essay marked separately, then one band.
    `writing_band` is only set when both tasks were marked; otherwise the status is `partial`
    and the band is withheld (a missing task cannot be averaged in).
    """

    status: Literal["graded", "partial"]
    module: Literal["ielts_writing"] = "ielts_writing"
    t1: Report
    t2: Report
    writing_band: float | None = None
    weights_note: str
    weights_source_urls: list[str]
    timing: Timing
    disclaimer: str = DISCLAIMER


class SpeakingCriterion(BaseModel):
    """
    One of the four Speaking criteria. `assessed` is False for a criterion this mock cannot
    judge (Pronunciation): it then has no band and `reason` says why, and it is shown as "not
    assessed", never as a guessed number.
    """

    id: Literal["FC", "LR", "GRA", "P"]
    name: str
    assessed: bool
    band: float | None = None
    rationale: str = ""
    evidence: list[str] = []  # short quotes from the transcript that the band rests on
    reason: str = ""  # why it was not assessed


class SpeakingPart(BaseModel):
    """What was said in one part, with the fluency figures counted in code from the transcript."""

    part: Literal[1, 2, 3]
    title: str
    transcript: str
    duration_s: float  # length of the recording, as measured by the browser
    allowed_s: int | None = None  # Part 2's long turn limit
    words: int
    words_per_minute: float | None = None
    fillers: int  # "um", "uh", "er" and the like


class SpeakingPriority(BaseModel):
    criterion: Literal["FC", "LR", "GRA"]
    priority: int
    do: str


class SpeakingReport(BaseModel):
    """
    The Speaking module marked from audio: a transcript and the three criteria a model can judge
    from it. `band` is the mean of the assessed criteria and is always `partial` while
    Pronunciation is not assessed. `status` is `no_speech` (and `band` None) when too little
    was said to mark.
    """

    status: Literal["graded", "no_speech"]
    module: Literal["ielts_speaking"] = "ielts_speaking"
    set_id: str
    sample: bool = False  # the quick sample set: a few sentences, so the band means little
    band: float | None = None
    partial: bool = True
    partial_note: str
    criteria: list[SpeakingCriterion]
    not_assessed: list[str]  # the ids of the criteria without a band
    parts: list[SpeakingPart]
    strengths: list[str] = []
    priorities: list[SpeakingPriority] = []
    flags: list[str] = []
    model: str = ""
    timing: Timing
    descriptors_source_url: str
    format_source_url: str
    disclaimer: str = DISCLAIMER
