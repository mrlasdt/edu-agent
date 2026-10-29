"""
IELTS Writing Task 2 rubric grader target: metrics, reply parsing, tier selection, the suite
runner and the CLI wiring. Model traffic goes through fakes; nothing here touches the network.
"""
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals.agent_eval import cli, ielts_writing
from evals.agent_eval.agreement import (
    agreement_report,
    quadratic_weighted_kappa,
    within_tolerance,
)
from evals.agent_eval.budget import Ledger
from evals.agent_eval.cache import ResponseCache
from evals.agent_eval.judges import JudgeError
from evals.agent_eval.transport import AllModelsFailedError, MeteredTransport
from evals.exam_set import ielts_writing_adversarial as adversarial
from evals.exam_set.local_store import register_source
from evals.exam_set.schema import (
    Design,
    ExamItem,
    LabelProvenance,
    Labels,
    LabelTier,
    Source,
)
from evals.loader import load_exam_items
from services.agent.src.agent.ielts.grading import grade_request
from services.agent.src.agent.ielts.scoring import GradeParseError
from tests.test_ielts_marking import ESSAY, PROMPT

# ── metrics ──────────────────────────────────────────────────────────────────


def test_within_tolerance_counts_half_band_misses_as_agreement():
    pairs = [(6.0, 6.0), (6.5, 6.0), (7.0, 6.0), (4.0, 6.5)]
    assert within_tolerance(pairs, 0.5) == 0.5
    assert within_tolerance([], 0.5) is None


def test_qwk_perfect_inverted_constant_and_undefined():
    perfect = [(4.0, 4.0), (6.0, 6.0), (8.0, 8.0)]
    assert quadratic_weighted_kappa(perfect) == pytest.approx(1.0)
    inverted = [(4.0, 8.0), (8.0, 4.0), (6.0, 6.0)]
    assert quadratic_weighted_kappa(inverted) < 0
    assert quadratic_weighted_kappa([(6.0, 5.0), (6.0, 7.0)]) == pytest.approx(0.0)  # constant
    assert quadratic_weighted_kappa([(6.0, 6.0), (6.0, 6.0)]) is None  # both raters constant
    assert quadratic_weighted_kappa([]) is None


def test_qwk_matches_a_hand_computed_value():
    # levels are half-band steps; two disagreements by one full band (2 levels) out of 4
    pairs = [(5.0, 5.0), (6.0, 6.0), (7.0, 6.0), (6.0, 7.0)]
    k = 19
    observed_cost = 2 * (2**2) / (k - 1) ** 2  # sum of weighted disagreements
    pred = {10: 1, 12: 2, 14: 1}
    gold = {10: 1, 12: 2, 14: 1}
    expected_cost = sum(
        ((i - j) ** 2 / (k - 1) ** 2) * pred.get(i, 0) * gold.get(j, 0) / 4
        for i in range(k) for j in range(k)
    )
    assert quadratic_weighted_kappa(pairs) == pytest.approx(1 - observed_cost / expected_cost)
    assert quadratic_weighted_kappa(pairs) == pytest.approx(0.5)


def test_agreement_report_fields():
    report = agreement_report([(6.0, 6.0), (7.0, 6.0)])
    assert report["n"] == 2 and report["exact"] == 0.5 and report["bias"] == 0.5
    assert report["mae"] == 0.5 and report["tolerance"] == 0.5
    assert report["within_tolerance"] == 0.5  # a full-band miss is outside 0.5
    assert agreement_report([])["n"] == 0


# ── parsing and scoring ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("criteria", "expected"),
    [
        ((6, 6, 6, 6), 6.0),
        ((6, 6, 6, 6.5), 6.0),  # mean 6.125 rounds down
        ((6, 6, 6.5, 6.5), 6.5),  # mean 6.25 rounds up to the half band
        ((6.5, 6.5, 6.5, 7), 6.5),  # mean 6.625 rounds down to the half band
        ((6.5, 6.5, 7, 7), 7.0),  # mean 6.75 rounds up to the whole band
        ((4, 5, 5, 5), 5.0),  # mean 4.75
    ],
)
def test_round_overall_uses_the_ielts_rounding_rule(
    criteria: tuple[float, float, float, float], expected: float
):
    scores = dict(zip(("TR", "CC", "LR", "GRA"), criteria, strict=True))
    assert ielts_writing.round_overall(scores) == expected


def test_parse_grade_snaps_to_half_bands_and_clamps():
    reply = 'Here you go: {"TR": 6.3, "CC": 5, "LR": "7.0", "GRA": 11, "rationale": "x"}'
    assert ielts_writing.parse_grade(reply) == {"TR": 6.5, "CC": 5.0, "LR": 7.0, "GRA": 9.0}


@pytest.mark.parametrize(
    "reply",
    ["no json here", '{"TR": 6, "CC": 6, "LR": 6}', '{"TR": "high", "CC": 1}', "{bad json}"],
)
def test_parse_grade_rejects_unusable_replies(reply: str):
    with pytest.raises(GradeParseError):
        ielts_writing.parse_grade(reply)


def test_long_essays_are_sent_whole_not_truncated():
    essay = " ".join(["word"] * 2000)
    messages = grade_request("Prompt?", essay, "system")
    assert messages[1]["content"].count("word") == 2000


async def test_llm_grader_goes_through_litellm(monkeypatch: pytest.MonkeyPatch):
    seen: list[dict[str, Any]] = []
    essay = adversarial.generate_all()[0].response
    assert essay is not None

    async def fake(**kw: Any) -> Any:
        seen.append(kw)
        content = '{"TR": 6, "CC": 6.5, "LR": 6, "GRA": 5.5, "rationale": "ok"}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr("litellm.acompletion", fake)
    grader = ielts_writing.LLMIeltsGrader("m", samples=3)
    graded = await grader.grade(adversarial.PROMPT, essay, essay_item("x", 6.0))
    assert graded.criteria["CC"] == 6.5 and graded.criteria["GRA"] == 5.5
    assert len(seen) == 3 and graded.spread == 0.0
    assert seen[0]["model"] == "m" and seen[0]["temperature"] == 0.3
    assert "Candidate essay" in seen[0]["messages"][1]["content"]


async def test_cached_reruns_replay_each_grader_sample_and_retry_separately(tmp_path: Path):
    replies = ["garbage", *(
        json.dumps({c: 4.0 + n for c in ("TR", "CC", "LR", "GRA")} | {"rationale": "r"})
        for n in (1, 2, 3)
    )]
    calls: list[int] = []

    async def real(**kw: Any) -> Any:
        calls.append(1)
        content = replies[len(calls) - 1]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )

    async def run() -> ielts_writing.EvalGrade:
        transport = MeteredTransport(
            Ledger(tmp_path / "l.json", 20.0), "run", cache=ResponseCache(tmp_path / "cache")
        )
        transport._real = real
        with transport.installed():
            grader = ielts_writing.LLMIeltsGrader("openai/gpt-4.1-mini", samples=3)
            return await grader.grade(PROMPT, ESSAY, essay_item("x", 6.0))

    first = await run()
    assert len(calls) == 4  # three samples and the retry of the unusable reply
    assert sorted(s["TR"] for s in first.samples) == [5.0, 6.0, 7.0] and first.spread == 2.0
    again = await run()
    assert len(calls) == 4 and again.samples == first.samples and again.spread == 2.0


# ── tier selection and sampling ──────────────────────────────────────────────


def essay_item(
    id: str, overall: float, tier: LabelTier = LabelTier.site_or_model_weak,
    criteria: dict[str, float] | None = None, skill: str = "writing_t2",
) -> ExamItem:
    return ExamItem(
        id=id, exam="ielts", skill=skill,  # type: ignore[arg-type]
        task_type="essay_score", prompt="Synthetic prompt?", response="Synthetic essay.",
        labels=Labels(
            overall=overall,
            criteria=(
                criteria if criteria is not None else {c: overall for c in ielts_writing.CRITERIA}
            ),
        ),
        label_provenance=LabelProvenance(tier=tier),
        source=Source(kind="third_party_local", license="synthetic"),
        design=Design(),
    )


def test_select_items_picks_the_best_available_tier():
    weak = [essay_item(f"w{i}", 6.0) for i in range(3)]
    captain = [essay_item("c1", 5.0, LabelTier.captain_blind)]
    tier, picked, counts = ielts_writing.select_items(weak + captain, "best", 10, 1)
    assert tier == LabelTier.captain_blind and [i.id for i in picked] == ["c1"]
    assert counts == {
        "examiner": 0, "teacher_graded": 0, "captain_blind": 1, "site_or_model_weak": 3,
    }
    forced, picked, _ = ielts_writing.select_items(weak + captain, "site_or_model_weak", 2, 1)
    assert forced == LabelTier.site_or_model_weak and len(picked) == 2


def test_select_items_ignores_task1_and_unlabelled_items_and_handles_empty():
    items = [essay_item("t1", 6.0, skill="writing_t1")]
    assert ielts_writing.select_items(items, "best", 5, 1) == (
        None, [], {"examiner": 0, "teacher_graded": 0, "captain_blind": 0, "site_or_model_weak": 0}
    )
    tier, picked, _ = ielts_writing.select_items(items, "examiner", 5, 1)
    assert tier == LabelTier.examiner and picked == []


def test_stratified_sample_spreads_bands_and_is_deterministic():
    items = [essay_item(f"a{i}", 4.0) for i in range(20)] + [essay_item("b", 8.0)]
    first = ielts_writing.stratified_sample(items, 4, seed=7)
    assert first == ielts_writing.stratified_sample(items, 4, seed=7)
    bands = [i.labels.overall for i in first if i.labels]
    assert 8.0 in bands and bands.count(4.0) == 3
    assert ielts_writing.stratified_sample(items, 99, seed=7) == items


# ── suite runner ─────────────────────────────────────────────────────────────


def offset_grader(offset: float) -> ielts_writing.HeuristicIeltsGrader:
    def score(item: ExamItem) -> dict[str, float]:
        assert item.labels is not None
        return {c: min(9.0, g + offset) for c, g in item.labels.criteria.items()}

    return ielts_writing.HeuristicIeltsGrader(score)


async def test_run_reports_tier_n_and_agreement_per_criterion():
    items = [essay_item(f"s{i}", b) for i, b in enumerate([4.0, 5.0, 6.0, 7.0, 8.0])]
    report = await ielts_writing.run_ielts_writing(items, offset_grader(0.5), max_items=5)
    assert report["label_tier"] == "site_or_model_weak" and report["n_items"] == 5
    assert "NOT agreement with an examiner" in report["label_tier_caveat"]
    assert report["overall"]["n"] == 5 and report["overall"]["within_tolerance"] == 1.0
    assert report["overall"]["exact"] == 0.0 and report["overall"]["bias"] == pytest.approx(0.5)
    assert report["overall"]["qwk"] is not None and report["overall"]["qwk"] > 0.9
    assert set(report["per_criterion"]) == {
        "Task Response", "Coherence and Cohesion", "Lexical Resource",
        "Grammatical Range and Accuracy",
    }
    assert all(v["n"] == 5 for v in report["per_criterion"].values())
    assert report["units_total"] == 5 and report["units_dropped_model_unavailable"] == 0


async def test_run_scores_overall_from_the_criteria_not_the_model():
    items = [essay_item("a", 6.0, criteria={"TR": 6, "CC": 6, "LR": 6, "GRA": 6})]
    grader = ielts_writing.HeuristicIeltsGrader(
        lambda item: {"TR": 6.0, "CC": 6.0, "LR": 6.0, "GRA": 6.5}
    )
    report = await ielts_writing.run_ielts_writing(items, grader)
    assert report["rows"]["grade"][0]["pred_overall"] == 6.0  # mean 6.125 rounds down


async def test_run_without_per_criterion_labels_only_scores_overall():
    items = [
        essay_item("a", 6.0, LabelTier.examiner, criteria={}),
        essay_item("b", 7.0, LabelTier.examiner, criteria={}),
    ]
    grader = ielts_writing.HeuristicIeltsGrader(
        lambda item: {c: 6.5 for c in ielts_writing.CRITERIA}
    )
    report = await ielts_writing.run_ielts_writing(items, grader)
    assert report["label_tier"] == "examiner" and report["overall"]["n"] == 2
    assert all(v["n"] == 0 for v in report["per_criterion"].values())


async def test_run_counts_judge_errors_and_unavailable_models():
    class Flaky:
        name = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        async def grade(self, prompt: str, essay: str, item: ExamItem) -> ielts_writing.EvalGrade:
            self.calls += 1
            if self.calls == 1:
                raise JudgeError("unparseable")
            if self.calls == 2:
                raise AllModelsFailedError("down")
            return ielts_writing.EvalGrade(criteria={c: 6.0 for c in ielts_writing.CRITERIA})

    items = [essay_item(f"s{i}", 6.0) for i in range(3)]
    report = await ielts_writing.run_ielts_writing(items, Flaky(), max_items=3)
    assert report["n_items"] == 1 and report["errors"] == 1
    assert report["units_total"] == 3 and report["units_dropped_model_unavailable"] == 1


async def test_run_with_no_labelled_items_says_so():
    report = await ielts_writing.run_ielts_writing([], offset_grader(0))
    assert report["n_items"] == 0 and report["label_tier"] is None and "note" in report


# ── CLI wiring ───────────────────────────────────────────────────────────────


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*a: Any, **k: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def _seed_local(tmp_path: Path) -> Path:
    local = tmp_path / "local"
    items = [essay_item(f"s{i}", b) for i, b in enumerate([4.0, 5.5, 6.5, 8.0])]
    register_source(
        local, source_id="syn", name="synthetic", url="n/a", licence_status="synthetic",
        items=items, file_name="syn.jsonl",
    )
    return local


def test_offline_cli_run_reports_ielts_tier_and_n(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _block_network(monkeypatch)
    local = _seed_local(tmp_path)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    code = cli.main([
        "--mode", "offline", "--suite", "ielts_writing", "--max-items", "4",
        "--local-dir", str(local), "--ledger", str(tmp_path / "l.json"),
        "--out-dir", str(tmp_path / "runs"),
    ])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    suite = out["suites"]["ielts_writing"]
    assert suite["label_tier"] == "site_or_model_weak" and suite["n_items"] == 4
    assert suite["overall"]["within_tolerance"] == 1.0  # offline grader echoes the labels
    assert out["models_used"] == {}  # the offline grader never calls a model
    assert out["regression_metrics"] == {
        "ielts_overall_within_half_band__site_or_model_weak": 1.0
    }
    assert "rows" not in suite  # the printed summary never carries per-item output


def test_dry_run_plan_counts_ielts_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    local = _seed_local(tmp_path)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    assert cli.main([
        "--suite", "ielts_writing", "--max-items", "3", "--local-dir", str(local),
        "--ledger", str(tmp_path / "l.json"),
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["items"]["ielts_writing"] == 3
    assert plan["max_judge_calls"] == 9  # 3 essays x 3 samples


def test_offline_cli_run_without_local_data_is_a_clean_empty_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    code = cli.main([
        "--mode", "offline", "--suite", "ielts_writing", "--local-dir", str(tmp_path / "none"),
        "--ledger", str(tmp_path / "l.json"), "--out-dir", str(tmp_path / "runs"),
    ])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["suites"]["ielts_writing"]["n_items"] == 0
    assert out["regression_metrics"] == {}


def test_ielts_writing_is_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    # It needs local third-party data, so the default suites leave it out.
    local = _seed_local(tmp_path)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    assert cli.main(["--local-dir", str(local), "--ledger", str(tmp_path / "l.json")]) == 0
    assert json.loads(capsys.readouterr().out)["items"]["ielts_writing"] == 0


# ── gate, report and adversarial suites ──────────────────────────────────────


async def test_run_includes_the_calibration_gate_with_spreads():
    items = [essay_item(f"s{i}", b) for i, b in enumerate([4.0, 5.0, 6.0, 7.0, 8.0])]
    report = await ielts_writing.run_ielts_writing(items, offset_grader(0.0), max_items=5)
    gate = report["calibration_gate"]
    assert gate["n"] == 5 and gate["tier"] == "site_or_model_weak"
    assert gate["development_gate"]["passed"] is True
    assert gate["decision_gate"]["passed"] is False  # n < 60 and a weak tier cannot certify
    assert "test-retest spread <= 0.5" in gate["decision_gate"]["unmeasured"]


async def test_adversarial_suite_holds_even_when_the_model_scores_everything_a_9():
    items = load_exam_items(exam="ielts", skill="writing_t2")
    report = await ielts_writing.run_ielts_adversarial(
        items, ielts_writing.OfflineCompleter(), samples=1
    )
    assert report["n_cases"] >= 10 and report["failed_cases"] == []
    assert report["max_band_given"] is not None and report["max_band_given"] < 9


async def test_report_suite_runs_the_mechanical_checks_on_a_full_offline_report():
    long_essay = " ".join(["Schools should teach cooking and budgeting to every student."] * 30)
    items = [essay_item(f"r{i}", 6.0) for i in range(2)]
    for item in items:
        item.response = long_essay
        item.prompt = adversarial.PROMPT
    report = await ielts_writing.run_ielts_report(
        items, ielts_writing.OfflineCompleter(), samples=1
    )
    assert report["n_items"] == 2 and report["reports_with_problems"] == 0
    assert report["feedback_status"] == {"partial": 2}  # the fake invents one quote each


def test_offline_cli_runs_the_adversarial_and_report_suites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _block_network(monkeypatch)
    local = _seed_local(tmp_path)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    code = cli.main([
        "--mode", "offline", "--suite", "ielts_adversarial", "--samples", "1",
        "--local-dir", str(local), "--ledger", str(tmp_path / "l.json"),
        "--out-dir", str(tmp_path / "runs"),
    ])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["regression_metrics"] == {"ielts_adversarial_pass_rate": 1.0}
