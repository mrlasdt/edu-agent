"""Typed data for the question-generation pipeline: requests, plans, items, traces."""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class QuestionType(StrEnum):
    numeric_entry = "numeric_entry"
    multiple_choice_single = "multiple_choice_single"
    multiple_choice_multiple = "multiple_choice_multiple"
    quantitative_comparison = "quantitative_comparison"


class Expert(StrEnum):
    """Generator specialists. Each owns one prompt and one family of question types."""

    numeric = "numeric_expert"
    choice = "choice_expert"
    comparison = "comparison_expert"


EXPERT_FOR_TYPE: dict[QuestionType, Expert] = {
    QuestionType.numeric_entry: Expert.numeric,
    QuestionType.multiple_choice_single: Expert.choice,
    QuestionType.multiple_choice_multiple: Expert.choice,
    QuestionType.quantitative_comparison: Expert.comparison,
}

Difficulty = Literal["easy", "medium", "hard"]


class GenerationRequest(BaseModel):
    request_id: str
    topic: str
    difficulty: Difficulty = "medium"
    question_type: QuestionType | None = None  # None: the planner chooses


class Plan(BaseModel):
    question_type: QuestionType
    expert: Expert
    topic: str
    difficulty: Difficulty
    learning_objective: str = ""
    constraints: list[str] = Field(default_factory=list)


class QuestionItem(BaseModel):
    """
    A generated GRE Quant item, shaped so its key can be recomputed by a solver.

    `expression` is a sympy expression or equation whose value (or solution set) is the
    key; for multiple choice `choices` holds the numeric value of each option and `answer`
    the value(s) of the correct option(s). A quantitative comparison carries
    `quantity_a` / `quantity_b` instead and `answer` is one of A, B, C, D.
    """

    question_type: QuestionType
    stem: str
    choices: list[str] = Field(default_factory=list)
    expression: str = ""
    quantity_a: str = ""
    quantity_b: str = ""
    answer: str
    explanation: str = ""

    @field_validator("choices", mode="before")
    @classmethod
    def _choices_as_text(cls, value: object) -> object:
        """Models often emit numeric options as JSON numbers (or null for none)."""
        if value is None:
            return []
        return [str(v) for v in value] if isinstance(value, list) else value

    @field_validator("expression", "quantity_a", "quantity_b", "explanation", mode="before")
    @classmethod
    def _null_as_empty(cls, value: object) -> object:
        return "" if value is None else value

    @field_validator("answer", mode="before")
    @classmethod
    def _answer_as_text(cls, value: object) -> object:
        is_number = isinstance(value, int | float) and not isinstance(value, bool)
        return str(value) if is_number else value

    @model_validator(mode="after")
    def _non_empty(self) -> QuestionItem:
        if not self.stem.strip() or not self.answer.strip():
            raise ValueError("stem and answer must be non-empty")
        return self


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class TraceEvent(BaseModel):
    """One auditable step. `data` carries the step's inputs/outputs (plan, item, checks)."""

    step: str
    attempt: int = 0
    ok: bool = True
    model: str | None = None
    latency_s: float = 0.0
    data: dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    request: GenerationRequest
    trace_id: str
    status: Literal["accepted", "needs_review", "unchecked", "unparseable"]
    item: QuestionItem | None = None  # accepted item (or the baseline's unchecked one)
    candidate: QuestionItem | None = None  # last attempt, kept for the human reviewer
    plan: Plan | None = None
    attempts: int = 0
    llm_calls: int = 0
    failures: list[str] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
