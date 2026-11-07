"""
The shape of an eval run: one typed model per suite result, and the report that holds them.

A suite result used to be a `dict[str, Any]` assembled at seven write sites and read back by
hand (about thirty hard-coded key paths). A suite that forgot `units_total` then read as full
coverage and the coverage gate never fired for it. Here every suite result subclasses
`SuiteResult`, whose `units_total` and `units_dropped` are required, so a suite that forgets
either fails at construction.

The stored JSON keeps its old key names, so reports and baselines written before this module
(`evals/baselines/*.json`, `.eval-runs/*.json`) still load: `units_dropped` is stored as
`units_dropped_model_unavailable` and read from either spelling. A result stored before the
unit counts existed loads through `SuiteResult.from_stored` with those counts unknown (None): its
coverage is unknown and the coverage gate fails it. Only that path may hold an unknown count.

Only the figures a reader (the gate, the regression metrics, the baseline compare, the two
report printers) looks at are typed field by field. Blobs that are written for a human to read
and never read back by code stay `dict[str, Any]`.
"""
from __future__ import annotations

from typing import Any, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    ValidationInfo,
    field_validator,
    model_validator,
)

JSON = dict[str, Any]

STORED: JSON = {"stored": True}


class StrictModel(BaseModel):
    """No unknown keys (a typo at a write site fails), written under the stored key names."""

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True, protected_namespaces=())


# ── small shared figures ─────────────────────────────────────────────────────


class Rate(StrictModel):
    """A proportion with its n and 95% Wilson interval (`value` and `ci95` None when n is 0)."""

    value: float | None
    successes: int
    n: int
    ci95: list[float] | None


class GroupStats(StrictModel):
    """pass@k, pass^k and the mean pass rate of one group of items."""

    n_items: int
    samples: int
    attempts: int
    mean_pass_rate: float | None
    mean_pass_rate_ci95: list[float] | None
    pass_at_k: Rate
    pass_hat_k: Rate
    flaky_items: list[str]
    always_failing_items: list[str]


class AgreementReport(StrictModel):
    """Within-tolerance rate, exact match, MAE, signed bias and QWK over n (pred, gold) pairs."""

    n: int
    tolerance: float
    within_tolerance: float | None
    exact: float | None
    mae: float | None
    bias: float | None
    qwk: float | None


class LabelAgreement(StrictModel):
    """A yes/no judge against known labels."""

    n: int
    agreement: float | None
    false_positive: int
    false_negative: int


class CheckRate(StrictModel):
    passed: int
    total: int
    value: float


class TurnCost(StrictModel):
    """Cost and latency over a set of agent turns. Only `turns` is set when there were none."""

    turns: int
    usd_total: float | None = None
    usd_per_turn_mean: float | None = None
    usd_per_turn_p95: float | None = None
    usd_per_turn_if_gpt_4o_mini: float | None = None
    llm_calls_per_turn_mean: float | None = None
    tool_calls_per_turn_mean: float | None = None
    turns_with_tool_calls: int | None = None
    tool_error_rate: float | None = None
    turns_tool_budget_exhausted: int | None = None
    input_tokens_per_turn_mean: float | None = None
    output_tokens_per_turn_mean: float | None = None
    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None
    turns_with_estimated_tokens: int | None = None


class RunMeta(StrictModel):
    """Provenance of a run: which model produced the numbers, and whether it is the default."""

    model: str
    model_chain: list[str]
    samples: int
    max_tokens: int | None = None
    product_default_model: str
    is_product_default_model: bool
    math_tools: str | None = None
    prompt_versions: dict[str, Any] | None = None
    temperature: float | None = None
    git_commit: str | None
    date_utc: str


# ── suite results ────────────────────────────────────────────────────────────


class SuiteResult(StrictModel):
    """
    What every suite reports. `units_total` is how many units of work the suite started and
    `units_dropped` how many of them never finished because the model chain was unavailable;
    the coverage gate is `1 - units_dropped / units_total`. Both are required on purpose.
    """

    units_total: int | None
    units_dropped: int | None = Field(
        validation_alias=AliasChoices("units_dropped", "units_dropped_model_unavailable"),
        serialization_alias="units_dropped_model_unavailable",
    )

    @model_validator(mode="before")
    @classmethod
    def _unknown_counts_when_stored(cls, data: Any, info: ValidationInfo) -> Any:
        """A result read from disk may predate the counts; they load as unknown, never as 0."""
        if not (isinstance(data, dict) and info.context == STORED):
            return data
        filled = dict(data)
        filled.setdefault("units_total", None)
        if not {"units_dropped", "units_dropped_model_unavailable"} & data.keys():
            filled["units_dropped"] = None
        return filled

    @field_validator("units_total", "units_dropped")
    @classmethod
    def _counts_are_known_unless_stored(cls, value: int | None, info: ValidationInfo) -> int | None:
        if value is None and info.context != STORED:
            raise ValueError("a suite result must say how many units it started and dropped")
        return value

    # Subclasses that carry per-attempt agent rows say how long each one took.
    def agent_latencies(self) -> list[float]:
        return []

    @classmethod
    def from_stored(cls, raw: JSON) -> Self:
        """Load a result written to disk; counts it never recorded stay unknown (None)."""
        return cls.model_validate(raw, context=STORED)

    @property
    def coverage(self) -> float | None:
        """Share of units that finished: 1.0 when none started, None when the counts are unknown."""
        if self.units_total is None or self.units_dropped is None:
            return None
        return 1 - self.units_dropped / self.units_total if self.units_total else 1.0


class TurnUsage(StrictModel):
    """Cost, tokens and time of one agent turn, from the record its meter wrote."""

    usd: float
    input_tokens: int
    output_tokens: int
    llm_calls: int
    tool_calls: int
    tool_errors: int
    tool_budget_exhausted: bool
    latency_ms: float
    estimated: bool


class TranscriptMessage(StrictModel):
    role: str
    content: str


class QuantAttempt(StrictModel):
    """One attempt at one item of the failure-seeded suite."""

    item_id: str
    kind: str
    tag: str
    attempt: int
    passed: bool
    checks: dict[str, bool]
    transcript: list[TranscriptMessage]
    turns: list[TurnUsage]
    answer: str | None = None  # solve items only
    independent_check: str | None = None  # solve items only


class QuantRows(StrictModel):
    attempts: list[QuantAttempt]


class GroupSummary(GroupStats):
    checks: dict[str, CheckRate]
    cost_latency: TurnCost


class SecurityBlock(Rate):
    failed_items: list[str]


class VerdictTally(StrictModel):
    """How the independent check judged a set of answers."""

    agrees: int
    disagrees: int
    not_checkable: int


class SeededWrongAnswers(Rate, VerdictTally):
    count: int
    accepted_items: list[str]


class SelfCompare(Rate):
    count: int
    accepted_items: list[str]


class Control(VerdictTally):
    n: int


class ObservedWrong(Rate):
    count: int
    solve_attempts: int
    not_checkable: int


class WrongButVerified(StrictModel):
    """How often the product's independent check agrees with an answer the key says is wrong."""

    definition: str
    seeded: SeededWrongAnswers
    self_compare: SelfCompare
    control: Control
    observed: ObservedWrong


class QuantSuiteResult(SuiteResult):
    """The failure-seeded Quant suite (`--suite quant`). `rows` is absent from a baseline."""

    samples: int
    n_items: int
    groups: dict[str, GroupSummary]
    by_tag: dict[str, GroupStats]
    all_model_turns: TurnCost
    security: SecurityBlock | None = None
    caveats: list[str] = Field(default_factory=list)
    wrong_but_verified: WrongButVerified | None = None
    rows: QuantRows | None = None


class TutorSummary(StrictModel):
    leak_rate_rule: float | None
    leak_rate_judge: float | None
    leak_rate_rule_by_probe: dict[str, float | None]
    judge_vs_rule_on_agent_output: LabelAgreement


class SolveSummary(StrictModel):
    accuracy: float | None
    verifier_fail_rate: float | None
    judge_vs_sympy_on_agent_output: LabelAgreement


class JudgeCalibration(StrictModel):
    leak_labels: LabelAgreement
    correctness_labels: LabelAgreement
    errors: int


class TutorRow(StrictModel):
    item_id: str
    probe: str
    latency_s: float
    leak_rule: bool
    leak_judge: bool | None


class SolveRow(StrictModel):
    item_id: str
    latency_s: float
    answer: str
    correct: bool
    verifier_fail: bool | None
    judge_correct: bool | None


class QuantJudgeRows(StrictModel):
    tutor: list[TutorRow]
    solve: list[SolveRow]


class QuantJudgeResult(SuiteResult):
    """The LLM-judge calibration on the solve items (`--suite quant_judge`)."""

    n_items: int
    tutor: TutorSummary
    solve: SolveSummary
    judge_calibration: JudgeCalibration
    rows: QuantJudgeRows

    def agent_latencies(self) -> list[float]:
        return [r.latency_s for r in self.rows.solve] + [r.latency_s for r in self.rows.tutor]


class AwResult(SuiteResult):
    """Judge calibration against the published GRE Analytical Writing anchor essays."""

    n_judgements: int
    within_tolerance: float | None
    tolerance: float
    mae: float | None
    errors: int


class GenerationRow(StrictModel):
    request_id: str
    status: str
    attempts: int
    llm_calls: int
    defects: list[str]
    valid: bool
    judge_ok: bool | None
    failures: list[str]
    trace_id: str
    trace: list[JSON]


class GenerationCost(StrictModel):
    calls: int
    usd: float
    prompt_tokens: int
    completion_tokens: int
    usd_per_valid_item: float | None
    usd_per_accepted_item: float | None


class GenerationTarget(StrictModel):
    """One generator (baseline or pipeline) over the same requests."""

    n_requests: int
    n_completed: int
    emitted_rate: float | None
    valid_yield: float | None
    verifier_reject_rate: float | None
    shipped_defect_rate: float | None
    shipped_defects_by_check: dict[str, int]
    escalation_rate: float | None
    unparseable_rate: float | None
    mean_attempts: float | None
    mean_llm_calls: float | None
    judge_accept_rate: float | None
    judge_vs_audit_agreement: float | None
    cost: GenerationCost


class GenerationResult(SuiteResult):
    """Baseline vs LangGraph pipeline question generation (`--suite generation`)."""

    n_requests: int
    targets: dict[str, GenerationTarget]
    comparison: JSON  # written for a human; empty unless both targets ran
    review_queue: int
    judge_errors: int
    rows: dict[str, list[GenerationRow]]


class IeltsWritingResult(SuiteResult):
    """The IELTS Writing grader against labelled essays. Only the header when none exist."""

    label_tier: str | None
    label_tier_caveat: str | None
    tier_counts_available: dict[str, int]
    grader: str
    tolerance: float
    n_items: int
    note: str | None = None
    overall: AgreementReport | None = None
    per_criterion: dict[str, AgreementReport] | None = None
    calibration_gate: JSON | None = None
    errors: int | None = None
    rows: dict[str, list[JSON]] | None = None


class IeltsReportResult(SuiteResult):
    """Full reports (grade plus feedback) run through the mechanical checks."""

    label_tier: str | None
    n_items: int
    reports_with_problems: int
    problems: list[str]
    feedback_status: dict[str, int]
    rows: dict[str, list[JSON]]


class IeltsAdversarialResult(SuiteResult):
    """Hostile essays must raise their flags and stay under their band ceiling."""

    n_cases: int
    passed: int
    failed_cases: list[str]
    max_band_given: float | None
    rows: dict[str, list[JSON]]


SUITE_MODELS: dict[str, type[SuiteResult]] = {
    "quant": QuantSuiteResult,
    "quant_judge": QuantJudgeResult,
    "aw": AwResult,
    "generation": GenerationResult,
    "ielts_writing": IeltsWritingResult,
    "ielts_report": IeltsReportResult,
    "ielts_adversarial": IeltsAdversarialResult,
}


# ── the report ───────────────────────────────────────────────────────────────


class CoverageGate(StrictModel):
    passed: bool
    min: float
    coverage: dict[str, float | None]


class EvalReport(StrictModel):
    """One run of `python -m evals.agent_eval` (non dry-run): its suites and its provenance."""

    # `SerializeAsAny` writes each suite under its own subclass's fields, not the base's two.
    suites: dict[str, SerializeAsAny[SuiteResult]] = Field(default_factory=dict)

    run_id: str
    mode: str
    model_chain: list[str]
    judge: str
    seed: int
    run_meta: RunMeta | None = None  # reports written before provenance existed have none
    aborted: str | None = None
    cost_latency: JSON | None = None
    models_used: dict[str, int] = Field(default_factory=dict)
    ledger_spent_usd: float | None = None
    call_failures: JSON | None = None
    cache: dict[str, int] | None = None
    coverage_gate: CoverageGate | None = None
    regression_metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("suites", mode="before")
    @classmethod
    def _load_stored_suites(cls, value: Any) -> Any:
        """A suite read from disk is a dict; build it as the model its name says."""
        if not isinstance(value, dict):
            return value
        return {
            name: SUITE_MODELS[name].from_stored(raw) if isinstance(raw, dict) else raw
            for name, raw in value.items()
        }

    def suite[T: SuiteResult](self, name: str, kind: type[T]) -> T | None:
        """The named suite's result, or None when that suite was not run."""
        found = self.suites.get(name)
        if found is None:
            return None
        if not isinstance(found, kind):
            raise TypeError(f"suite {name!r} holds a {type(found).__name__}, not a {kind.__name__}")
        return found

    def agent_latencies(self) -> list[float]:
        return [t for s in self.suites.values() for t in s.agent_latencies()]

    def coverage_failures(self, min_coverage: float) -> dict[str, float | None]:
        """Suites whose share of completed units is below `min_coverage`, or unknown (None)."""
        coverages = {name: s.coverage for name, s in self.suites.items()}
        return {name: c for name, c in coverages.items() if c is None or c < min_coverage}


class BaselineMetric(StrictModel):
    value: float | None
    n: int
    ci95: list[float] | None
    samples: int


class QuantBaseline(StrictModel):
    """`evals/baselines/quant.json`: aggregates only (no model text) and the run's provenance."""

    schema_id: str = Field(alias="schema")
    suite: str
    meta: RunMeta
    note: str
    metrics: dict[str, BaselineMetric]
    summary: QuantSuiteResult

    model_config = ConfigDict(populate_by_name=True)
