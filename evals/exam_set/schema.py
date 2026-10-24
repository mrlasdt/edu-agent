"""
Shared item schema for the IELTS and SAT golden sets (the GRE files keep their own shape).

Every item carries a *label tier* that says where its label came from, so a metric computed
against it can always be reported with that caveat. Tiers, strongest first for essay scoring:

  examiner            published band from the exam owner's own sample material
  teacher_graded      per-criterion bands from a named experienced instructor on a public page
                      (not an official examiner; a single rater, so noisier than `examiner`)
  captain_blind       scored by the project owner under the blind protocol (not an examiner)
  site_or_model_weak  score from a scraped site's users or a model: noisy, never examiner truth
  solver              answer computed by sympy with a second independent computation
  author_key          key written by the item's author (an agent unless a human checked it)

Third-party text is never committed: items whose `source.kind` is `third_party_local` may only
live in the gitignored local directory (see `evals/exam_set/local_store.py`).
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LabelTier(StrEnum):
    examiner = "examiner"
    teacher_graded = "teacher_graded"
    captain_blind = "captain_blind"
    site_or_model_weak = "site_or_model_weak"
    solver = "solver"
    author_key = "author_key"


# Order used to pick "the best available" label for essay scoring.
ESSAY_TIER_ORDER: tuple[LabelTier, ...] = (
    LabelTier.examiner,
    LabelTier.teacher_graded,
    LabelTier.captain_blind,
    LabelTier.site_or_model_weak,
)

TIER_CAVEATS: dict[LabelTier, str] = {
    LabelTier.examiner: (
        "published band from the exam owner's sample material; not a re-rating, and the "
        "publisher warns sample scripts are not definitive examples of a band"
    ),
    LabelTier.teacher_graded: (
        "criterion bands and comments from one experienced instructor on a public page; not an "
        "official examiner, unaudited, and the overall band is the one stated in the page "
        "heading, falling back to the mean of the four criterion bands when none is stated"
    ),
    LabelTier.captain_blind: "single-rater blind scores from the project owner, not an examiner",
    LabelTier.site_or_model_weak: (
        "scraped site or model scores (about 1.4 bands off an LLM grader on average per the "
        "dataset card; noisy at the extremes). Agreement with them is NOT agreement with an "
        "examiner"
    ),
    LabelTier.solver: "computed by sympy and re-computed independently; must agree",
    LabelTier.author_key: "author-written key; unchecked by a human unless noted",
}

CRITERIA: tuple[str, ...] = ("TR", "CC", "LR", "GRA")
CRITERION_NAMES = {
    "TR": "Task Response",
    "CC": "Coherence and Cohesion",
    "LR": "Lexical Resource",
    "GRA": "Grammatical Range and Accuracy",
}

Exam = Literal["ielts", "sat", "gre"]
Skill = Literal[
    "writing_t2", "writing_t1", "speaking_transcript", "reading", "listening_transcript",
    "sat_math", "sat_rw",
]
SourceKind = Literal["original", "cc_by", "third_party_local"]
COMMITTABLE_KINDS: frozenset[str] = frozenset({"original", "cc_by"})


def _check_band(value: float) -> float:
    if not 0.0 <= value <= 9.0 or (value * 2) % 1 != 0:
        raise ValueError(f"not a half band in 0..9: {value}")
    return value


class Labels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall: float | None = None
    criteria: dict[str, float] = Field(default_factory=dict)

    @field_validator("overall")
    @classmethod
    def _overall(cls, v: float | None) -> float | None:
        return None if v is None else _check_band(v)

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, v: dict[str, float]) -> dict[str, float]:
        for key, band in v.items():
            if key not in CRITERIA:
                raise ValueError(f"unknown criterion {key!r}")
            _check_band(band)
        return v


class LabelProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: LabelTier
    n_raters: int | None = None
    blind_to_target: bool | None = None
    date: str | None = None
    tolerance: float | None = None
    note: str = ""


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: SourceKind
    license: str
    source_id: str | None = None  # key into the local manifest for third-party data
    url: str | None = None
    attribution: str | None = None


class Design(BaseModel):
    model_config = ConfigDict(extra="forbid")

    band_target: float | None = None
    generator: str = "human"  # "human", "llm_edited" or "template:<id>"
    seed: int | None = None
    checks: list[str] = Field(default_factory=list)  # independent computations that agreed


class ExamItem(BaseModel):
    """One golden item. Optional fields are used by the skills that need them."""

    model_config = ConfigDict(extra="forbid")

    id: str
    exam: Exam
    skill: Skill
    task_type: str
    prompt: str
    response: str | None = None
    labels: Labels | None = None
    label_provenance: LabelProvenance
    source: Source
    design: Design = Field(default_factory=Design)
    split: Literal["dev", "calibration", "test", "train"] = "dev"
    # SAT Math / objective items (expression + ground_truth work with the sympy verifier)
    question_type: str | None = None
    expression: str | None = None
    ground_truth: str | None = None
    choices: dict[str, str] | None = None
    answer_choice: str | None = None
    # Reading items
    passage: str | None = None
    evidence: str | None = None
    # Writing items with published feedback (local only)
    examiner_comment: str | None = None
    extras: dict[str, Any] = Field(default_factory=dict)

    def committable(self) -> bool:
        return self.source.kind in COMMITTABLE_KINDS


def best_label_tier(items: list[ExamItem], order: tuple[LabelTier, ...] = ESSAY_TIER_ORDER) -> (
    LabelTier | None
):
    """The strongest tier in `order` that has at least one labelled item."""
    present = {i.label_provenance.tier for i in items if i.labels and i.labels.overall is not None}
    return next((t for t in order if t in present), None)
