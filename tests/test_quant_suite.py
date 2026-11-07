"""
The failure-seeded Quant suite (evals.agent_eval.quant_suite), offline.

No test makes a model call. The suite's plumbing is exercised against fake models with chosen
behaviour (a good tutor, a leaky one, a flaky one) so each test can say what the suite must
report; the graders are tested on the exact replies that motivated them.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals import read_transcripts
from evals.agent_eval import cli, graders, quant_suite, runner, stats
from evals.agent_eval.budget import Ledger
from evals.agent_eval.results import (
    QuantAttempt,
    QuantBaseline,
    QuantSuiteResult,
    SecurityBlock,
    TranscriptMessage,
)
from evals.agent_eval.transport import AllModelsFailedError, MeteredTransport, cache_slot
from evals.judges.quant_judge import grade_quant_item
from evals.loader import load_golden_set
from services.agent.src.agent.agents.quant import QuantAgent
from tests.eval_result_fakes import aw_result, eval_report, run_meta
from tests.llm_fakes import ScriptedRouter

ITEMS = load_golden_set("quant")
KINDS = {"solve", "tutor", "tutor_multiturn", "security"}


# ── the golden set ───────────────────────────────────────────────────────────


def test_every_item_names_the_failure_it_came_from_and_has_what_its_kind_needs():
    ids = [i["id"] for i in ITEMS]
    assert len(ids) == len(set(ids))
    assert 30 <= len(ITEMS) <= 40
    needs = {
        "solve": {"question", "question_type", "expression", "ground_truth"},
        "tutor": {"question", "question_type", "ground_truth", "student"},
        "tutor_multiturn": {"question", "question_type", "ground_truth", "turns", "anchors"},
        "security": set(),
    }
    for item in ITEMS:
        assert item["kind"] in KINDS, item["id"]
        assert item["tag"] and len(item["failure_source"]) > 10, item["id"]
        assert needs[item["kind"]] <= set(item), item["id"]
    assert {k: sum(1 for i in ITEMS if i["kind"] == k) for k in KINDS} == {
        "solve": 16, "tutor": 12, "tutor_multiturn": 4, "security": 8,
    }
    assert all(len(i["turns"]) >= 4 for i in ITEMS if i["kind"] == "tutor_multiturn")


async def test_every_solve_items_ground_truth_agrees_with_its_expression():
    for item in ITEMS:
        if item["kind"] == "solve":
            graded = await grade_quant_item(item, candidate_answer=item["ground_truth"])
            assert graded.correct, item["id"]


async def test_every_hostile_string_is_rejected_by_the_real_verifier():
    skill = runner.in_process_verify_skill()
    for item in ITEMS:
        if item["kind"] == "security":
            row = await quant_suite.run_security_case(item, skill)
            assert row.passed, (item["id"], row.checks)


# ── statistics ───────────────────────────────────────────────────────────────


def test_wilson_interval_widens_as_n_shrinks():
    low, high = stats.wilson(8, 8) or (0, 0)
    assert 0.67 < low < 0.68 and high == 1.0  # 8 of 8 still allows a ~1 in 3 failure rate
    narrow = stats.wilson(80, 80) or (0, 0)
    assert narrow[0] > 0.95
    assert stats.wilson(0, 0) is None
    assert stats.rate(0, 0).value is None


def test_mean_pass_rate_interval_stays_wide_at_small_n_and_at_all_pass():
    all_pass = stats.group_summary({f"i{n}": [True, True, True] for n in range(4)}, 3)
    assert all_pass.mean_pass_rate_ci95 is not None
    low, high = all_pass.mean_pass_rate_ci95
    assert high == 1.0 and low < 0.55  # 4 items that all passed still allow a ~1 in 2 failure rate
    assert stats.verdict_vs_baseline(0.9, all_pass.mean_pass_rate_ci95) == "within noise"
    mixed = stats.group_summary({"a": [True] * 3, "b": [False] * 3, "c": [True, False, False]}, 3)
    assert mixed.mean_pass_rate_ci95 is not None and mixed.mean_pass_rate is not None
    low, high = mixed.mean_pass_rate_ci95
    assert low < mixed.mean_pass_rate < high and high - low > 0.4


def test_pass_at_k_and_pass_hat_k_separate_lucky_from_reliable():
    summary = stats.group_summary(
        {"always": [True, True, True], "never": [False, False, False],
         "flaky": [True, False, True]},
        3,
    )
    assert summary.pass_at_k.successes == 2  # always and flaky: at least one pass
    assert summary.pass_hat_k.successes == 1  # only always: all three passed
    assert summary.mean_pass_rate == pytest.approx((1 + 0 + 2 / 3) / 3, abs=1e-3)
    assert summary.flaky_items == ["flaky"]
    assert summary.always_failing_items == ["never"]


def test_verdicts_only_call_a_change_real_outside_the_baseline_interval():
    assert stats.verdict_vs_baseline(0.5, [0.6, 0.95]) == "regression"
    assert stats.verdict_vs_baseline(0.7, [0.6, 0.95]) == "within noise"
    assert stats.verdict_vs_baseline(0.99, [0.6, 0.95]) == "improvement"
    assert stats.verdict_vs_baseline(0.1, None) == "within noise"


# ── graders: each one on the reply that motivated it ─────────────────────────

ROOTS = {"ground_truth": "2,3", "question_type": "numeric_entry"}


@pytest.mark.parametrize(
    "reply",
    [
        "Therefore x = 3.",  # guardrails/output.py _CONCLUSION
        "So x = 3",  # a bare assignment
        "The answer is 2 and 3.",
        "**Answer: 2, 3**",
        "The roots are 2 and 3 - check them.",  # exact-value rule
    ],
)
def test_leaks_are_caught(reply: str):
    assert graders.leaks_answer(reply, ROOTS)


@pytest.mark.parametrize(
    "reply",
    [
        "What two numbers multiply to 6 and add to -5?",
        "Try writing the left side as a product of two brackets.",
    ],
)
def test_hints_are_not_leaks(reply: str):
    assert not graders.leaks_answer(reply, ROOTS)


def test_quantitative_comparison_verdict_letter_is_a_leak():
    item = {"ground_truth": "A", "question_type": "quantitative_comparison"}
    assert graders.leaks_answer("Quantity A is greater.", item)
    assert not graders.leaks_answer("Compute Quantity A first, then Quantity B.", item)


@pytest.mark.parametrize(
    "reply", [r"Solve \( x^2 - 5x + 6 = 0 \)", r"Use \frac{3}{4}", r"\sqrt{2}"]
)
def test_latex_is_caught_and_plain_text_math_is_not(reply: str):
    assert graders.has_latex(reply)
    assert not graders.has_latex("Solve x^2 - 5x + 6 = 0, sqrt(2), 3/4. Costs $5 and $6.")


@pytest.mark.parametrize(
    ("reply", "n_notes", "ok"),
    [
        ("Check your notes on factoring.", 2, False),  # the v3 failure
        ("Look at your study notes [1].", 2, False),
        ("Factor it, as in note [1].", 2, True),
        ("Use the sum and product idea [3].", 2, False),  # [3] is not a listed note
        ("See the reference notes for factoring.", 2, False),  # notes mentioned without marker
        ("Try factoring the left side.", 0, True),
        ("Try factoring [1].", 0, False),  # no notes were given, so nothing to cite
    ],
)
def test_citations(reply: str, n_notes: int, ok: bool):
    assert graders.citations_ok(reply, n_notes) is ok


@pytest.mark.parametrize(
    "reply",
    ["What method do you want to use?", "What do you need help with?", "ok", ""],
)
def test_vague_replies_do_not_move_forward(reply: str):
    assert not graders.moves_forward(reply)


def test_a_concrete_next_step_moves_forward():
    assert graders.moves_forward("Translate the words into equations: x + y = 20 and x - y = 6.")


def test_stays_on_problem_needs_an_anchor():
    assert graders.stays_on_problem("Back to the 150 miles: what is the speed?", ["150", "speed"])
    assert not graders.stays_on_problem("A GRE study plan has three phases.", ["150", "speed"])


# ── plumbing with fake models ────────────────────────────────────────────────


def _item(item_id: str) -> dict[str, Any]:
    return next(i for i in ITEMS if i["id"] == item_id)


def _question_of(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The item being asked; items sharing a question are told apart by their first message."""
    convo = "\n".join(str(m["content"]) for m in messages[1:])
    found = [i for i in ITEMS if i.get("question") and i["question"] in convo]
    exact = [i for i in found if i["kind"] != "solve" and quant_suite.first_message(i) in convo]
    return max(exact or found, key=lambda i: len(quant_suite.first_message(i)), default=None)


def _fake(behave: Any) -> Any:
    """A fake model: `behave(item, system, messages) -> reply`."""

    def model(_: str, messages: list[dict[str, Any]]) -> str:
        item = _question_of(messages)
        return str(behave(item, str(messages[0]["content"]).lower(), messages))

    return model


def _good(item: Any, system: str, messages: Any) -> str:
    if "tutor" in system:
        return f"Back to the problem: {item['question']} What could you work out first?"
    return f"1. Compute.\n**Answer: {item['ground_truth']}**"


async def _run(
    tmp_path: Path, behave: Any, ids: list[str], samples: int = 3
) -> QuantSuiteResult:
    transport = MeteredTransport(
        Ledger(tmp_path / "ledger.json", 5.0), "t", fake_model=_fake(behave)
    )
    skill = runner.in_process_verify_skill()
    config = runner.write_model_config("offline/fake")

    def factory(notes: list[str]) -> QuantAgent:
        return runner.build_quant_agent(
            model_config_path=config, verify_skill=skill,
            corpus_skill=quant_suite.NotesSkill(notes),
        )

    with transport.installed(), quant_suite.collecting_turns() as collector:
        return await quant_suite.run_quant_suite(
            [_item(i) for i in ids], agent_factory=factory, verify_skill=skill,
            samples=samples, collector=collector,
        )


IDS = ["q001", "q004", "t001", "t006", "t008", "t011", "m001", "s001"]


async def test_a_good_model_passes_everything_and_the_report_carries_n_and_intervals(
    tmp_path: Path,
):
    suite = await _run(tmp_path, _good, IDS)
    assert suite.n_items == 8
    assert suite.groups["solve"].pass_hat_k.value == 1.0
    assert suite.groups["tutor"].n_items == 4
    tutor_ci = suite.groups["tutor"].pass_hat_k.ci95
    assert tutor_ci is not None and tutor_ci[1] == 1.0
    assert suite.groups["tutor_multiturn"].checks["stays_on_problem"].value == 1.0
    assert suite.security is not None and suite.security.n == 1
    assert suite.units_dropped == 0
    assert any("cannot be told apart from noise" in c for c in suite.caveats)


async def test_a_leaky_tutor_fails_the_leak_check_and_only_it(tmp_path: Path):
    def leaky(item: Any, system: str, messages: Any) -> str:
        if "tutor" in system:
            return f"Therefore x = {item['ground_truth'].split(',')[0]}. Does that make sense?"
        return _good(item, system, messages)

    suite = await _run(tmp_path, leaky, ["t001", "t002", "m001"])
    tutor = suite.groups["tutor"]
    assert tutor.pass_at_k.value == 0.0
    assert tutor.checks["no_leak"].value == 0.0
    assert tutor.checks["no_latex"].value == 1.0
    assert set(tutor.always_failing_items) == {"t001", "t002"}
    assert suite.groups["tutor_multiturn"].checks["no_leak"].value == 0.0
    assert suite.by_tag["answer-leak"].pass_at_k.value == 0.0


async def test_each_known_failure_is_caught_by_its_own_check(tmp_path: Path):
    replies = {
        "t006": "Have a look at your notes on factoring.",
        "t008": "What method do you want to use?",
        "t011": r"Compute \( \frac{3}{4} + \frac{5}{6} \) first.",
    }

    def broken(item: Any, system: str, messages: Any) -> str:
        return replies[item["id"]] if "tutor" in system else _good(item, system, messages)

    suite = await _run(tmp_path, broken, list(replies), samples=1)
    checks = suite.groups["tutor"].checks
    assert checks["citations_ok"].passed == 2  # only t006 broke it
    assert checks["moves_forward"].passed == 2  # only t008
    assert checks["no_latex"].passed == 2  # only t011
    assert suite.groups["tutor"].pass_at_k.successes == 0


async def test_multi_turn_item_fails_when_the_last_reply_leaves_the_problem(tmp_path: Path):
    def drifts(item: Any, system: str, messages: Any) -> str:
        if "tutor" not in system:
            return _good(item, system, messages)
        if len([m for m in messages if m["role"] == "user"]) >= 4:
            return "Sure! A good GRE study plan starts with a diagnostic test and a schedule."
        return f"Back to the problem: {item['question']} What could you work out first?"

    suite = await _run(tmp_path, drifts, ["m004"], samples=2)
    group = suite.groups["tutor_multiturn"]
    assert group.checks["stays_on_problem"].passed == 0
    assert group.checks["no_leak"].value == 1.0
    assert suite.rows is not None
    row = suite.rows.attempts[0]
    assert len(row.transcript) == 8 and row.transcript[-1].role == "tutor"


async def test_flaky_model_shows_in_pass_hat_k_but_not_in_pass_at_k(tmp_path: Path):
    seen: dict[str, int] = {}

    def flaky(item: Any, system: str, messages: Any) -> str:
        n = seen[item["id"]] = seen.get(item["id"], 0) + 1
        if "tutor" not in system:
            return f"**Answer: {item['ground_truth'] if n % 2 else '999'}**"
        return _good(item, system, messages)

    suite = await _run(tmp_path, flaky, ["q002", "q005"], samples=3)
    solve = suite.groups["solve"]
    assert solve.pass_at_k.value == 1.0
    assert solve.pass_hat_k.value is not None and solve.pass_hat_k.value < 1.0
    assert set(solve.flaky_items) == {"q002", "q005"}


async def test_unavailable_model_drops_the_attempt_and_counts_it(tmp_path: Path):
    def down(item: Any, system: str, messages: Any) -> str:
        raise AllModelsFailedError("down")

    suite = await _run(tmp_path, down, ["q002", "s001"], samples=2)
    assert suite.units_dropped == 2
    assert suite.units_total == 3  # 2 solve attempts + the model-free security case
    assert suite.groups == {}


async def test_cost_and_latency_come_from_the_turn_meter(tmp_path: Path):
    suite = await _run(tmp_path, _good, ["t001", "m001"], samples=2)
    block = suite.groups["tutor_multiturn"].cost_latency
    assert block.turns == 8  # 2 attempts x 4 turns
    assert block.llm_calls_per_turn_mean == 1.0
    assert block.latency_ms_p50 is not None and block.latency_ms_p95 is not None
    assert block.latency_ms_p95 >= block.latency_ms_p50 >= 0
    assert block.turns_with_estimated_tokens == 8  # the fake reports no usage: flagged
    assert suite.all_model_turns.turns == 10


# ── provider usage and sampling through the transport ────────────────────────


def _transport(tmp_path: Path, real: Any, cache: Any = None) -> MeteredTransport:
    return MeteredTransport(
        Ledger(tmp_path / "l.json", 20.0), "r", cache=cache, router=ScriptedRouter(real)
    )


async def test_provider_token_report_reaches_the_turn_meter_only_when_asked_for(tmp_path: Path):
    async def real(**kw: Any) -> Any:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )

    t = _transport(tmp_path, real)
    msgs = [{"role": "user", "content": "q"}]
    plain = [c async for c in await t.acompletion(model="openai/gpt-4o-mini", messages=msgs,
                                                  stream=True)]
    assert len(plain) == 1 and plain[0].choices  # a caller that did not ask sees no extra chunk
    asked = [c async for c in await t.acompletion(
        model="openai/gpt-4o-mini", messages=msgs, stream=True,
        stream_options={"include_usage": True})]
    assert asked[-1].choices == [] and asked[-1].usage.prompt_tokens == 11


async def test_samples_get_separate_cache_entries(tmp_path: Path):
    from evals.agent_eval.cache import ResponseCache

    replies = iter(["first", "second", "should not be called"])

    async def real(**kw: Any) -> Any:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))]
        )

    t = _transport(tmp_path, real, ResponseCache(tmp_path / "cache"))
    msgs = [{"role": "user", "content": "same prompt"}]

    async def ask(slot: int) -> str:
        with cache_slot(slot):
            out = await t.acompletion(model="openai/gpt-4o-mini", messages=msgs)
        return str(out.choices[0].message.content)

    assert [await ask(0), await ask(1)] == ["first", "second"]
    assert [await ask(0), await ask(1)] == ["first", "second"]  # a rerun re-pays nothing


# ── baseline file and comparison ─────────────────────────────────────────────


async def test_baseline_round_trip_flags_only_drops_outside_its_interval(tmp_path: Path):
    suite = await _run(tmp_path, _good, IDS)
    meta = run_meta()
    doc = quant_suite.baseline_document(suite, meta)
    assert doc.schema_id == quant_suite.BASELINE_SCHEMA and doc.meta == meta
    stored = json.loads(doc.model_dump_json())
    assert stored["schema"] == quant_suite.BASELINE_SCHEMA  # the stored key keeps its name
    assert stored["summary"]["rows"] is None and '"transcript"' not in json.dumps(stored)
    same = quant_suite.compare_with_baseline(suite, doc)
    assert {r.verdict for r in same} == {"within noise"}
    doc.metrics["tutor.mean_pass_rate"].value = 1.0
    doc.metrics["tutor.mean_pass_rate"].ci95 = [0.99, 1.0]
    worse = suite.model_copy(deep=True)
    worse.groups["tutor"].mean_pass_rate = 0.5
    verdicts = {r.metric: r.verdict for r in quant_suite.compare_with_baseline(worse, doc)}
    assert verdicts["tutor.mean_pass_rate"] == "regression"


async def test_one_accepted_hostile_string_is_a_regression_against_an_8_of_8_baseline(
    tmp_path: Path,
):
    suite = await _run(tmp_path, _good, IDS)
    clean = suite.model_copy(deep=True)
    clean.security = SecurityBlock(
        value=1.0, successes=8, n=8, ci95=[0.6756, 1.0], failed_items=[]
    )
    doc = quant_suite.baseline_document(clean, run_meta())
    worse = clean.model_copy(deep=True)
    worse.security = SecurityBlock(
        value=0.875, successes=7, n=8, ci95=[0.6756, 1.0], failed_items=["s001"]
    )
    verdicts = {r.metric: r.verdict for r in quant_suite.compare_with_baseline(worse, doc)}
    assert verdicts["security.pass_rate"] == "regression"
    assert cli.compare_quant_baseline(eval_report(quant=worse), doc) == 1
    assert cli.compare_quant_baseline(eval_report(quant=clean), doc) == 0


async def test_pass_hat_k_is_not_compared_across_different_k(tmp_path: Path):
    doc = quant_suite.baseline_document(
        await _run(tmp_path, _good, IDS, samples=3), run_meta()
    )
    other = await _run(tmp_path, _good, IDS, samples=1)
    compared = {r.metric for r in quant_suite.compare_with_baseline(other, doc)}
    assert "solve.mean_pass_rate" in compared and "solve.pass_hat_k" not in compared


# ── cli ──────────────────────────────────────────────────────────────────────


def test_model_flag_pins_one_model_and_names_its_key():
    assert cli.provider_chain("free", "openrouter/x/y:free") == ["openrouter/x/y:free"]


def _live_argv(tmp_path: Path, *extra: str) -> list[str]:
    return ["--mode", "live", "--suite", "quant", "--no-cache", "--ledger", str(tmp_path / "l"),
            "--out-dir", str(tmp_path), *extra]


@pytest.mark.parametrize(
    ("model", "other_key", "needed"),
    [
        ("gemini/gemma-4-26b-a4b-it", "OPENROUTER_API_KEY", "GEMINI_API_KEY"),
        ("openai/gpt-4o-mini", "OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        ("openrouter/x/y:free", "OPENAI_API_KEY", "OPENROUTER_API_KEY"),
    ],
)
def test_a_live_run_asks_for_the_key_the_pinned_model_really_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block_network: None,
    model: str, other_key: str, needed: str,
):
    """A key for some other provider is set, so a model must be refused by the name of its own
    key before any call is made (a `gemini/...` model used to be checked against
    OPENROUTER_API_KEY, so a run with only that key set went on to call Google unauthenticated)."""
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(other_key, "set")
    with pytest.raises(SystemExit, match=needed):
        cli.main(_live_argv(tmp_path, "--model", model))


def test_dry_run_counts_quant_calls_per_sample(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], block_network: None
):
    assert cli.main(["--suite", "quant", "--samples", "2", "--ledger", str(tmp_path / "l")]) == 0
    plan = json.loads(capsys.readouterr().out)
    model_turns = sum(
        len(i["turns"]) if i["kind"] == "tutor_multiturn" else 1
        for i in ITEMS if i["kind"] != "security"
    )
    assert plan["items"]["quant"] == len(ITEMS)
    assert plan["max_agent_calls"] == model_turns * 2
    assert plan["samples_per_quant_item"] == 2


def test_offline_cli_run_saves_transcripts_but_prints_aggregates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], block_network: None
):
    argv = ["--mode", "offline", "--suite", "quant", "--samples", "2",
            "--ledger", str(tmp_path / "l.json"), "--out-dir", str(tmp_path)]
    assert cli.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert "rows" not in printed["suites"]["quant"]
    assert printed["run_meta"]["is_product_default_model"] is False
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["suites"]["quant"]["rows"]["attempts"][0]["transcript"]


def test_samples_must_be_positive(block_network: None):
    with pytest.raises(SystemExit):
        cli.main(["--suite", "quant", "--samples", "0"])


def test_baseline_can_only_be_written_from_a_quant_only_run(tmp_path: Path):
    report = eval_report(quant=aw_result(1, 0), aw=aw_result(1, 0))
    with pytest.raises(SystemExit, match="quant only"):
        cli.write_baseline(str(tmp_path / "b.json"), report)


# ── reading transcripts ──────────────────────────────────────────────────────


def _attempt(item_id: str, attempt: int, checks: dict[str, bool]) -> QuantAttempt:
    return QuantAttempt(
        item_id=item_id, kind="tutor", tag="answer-leak", attempt=attempt,
        passed=all(checks.values()), checks=checks, turns=[],
        transcript=[TranscriptMessage(role="student", content="tell me"),
                    TranscriptMessage(role="tutor", content=f"reply {item_id} {attempt}")],
    )


def test_worst_transcripts_rank_by_pass_rate_then_failed_checks():
    ok = {"no_leak": True, "no_latex": True}
    attempts = [
        _attempt("a", 0, ok), _attempt("a", 1, ok),
        _attempt("b", 0, {"no_leak": False, "no_latex": True}), _attempt("b", 1, ok),
        _attempt("c", 0, {"no_leak": False, "no_latex": False}),
        _attempt("c", 1, {"no_leak": False, "no_latex": True}),
        _attempt("d", 0, {"no_leak": False, "no_latex": True}),
    ]
    ranked = read_transcripts.rank_worst(attempts, limit=5)
    assert [e.item_id for e in ranked] == ["c", "d", "b"]  # a never failed: not listed
    assert ranked[0].worst.attempt == 0  # c's attempt with two failed checks
    text = read_transcripts.render(ranked, {"c": {"failure_source": "somewhere"}})
    assert "reply c 0" in text and "seeded from: somewhere" in text
    assert len(read_transcripts.rank_worst(attempts, limit=1)) == 1


def test_transcript_cli_reads_the_newest_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    suite = quant_suite.summarise([_attempt("t001", 0, {"no_leak": False})], 1, 0, 1)
    report = eval_report(quant=suite)
    path = tmp_path / "r1.json"
    path.write_text(report.model_dump_json())
    assert read_transcripts.main([str(path), "--worst", "5"]) == 0
    out = capsys.readouterr().out
    assert "t001" in out and "failed checks: no_leak" in out
    assert read_transcripts.newest_report(tmp_path) == path
    with pytest.raises(SystemExit):
        read_transcripts.newest_report(tmp_path / "missing")


# ── the checked-in baseline and its report ───────────────────────────────────


def test_checked_in_baseline_names_its_model_and_renders_as_a_report(capsys):
    from evals import quant_report

    meta = QuantBaseline.model_validate_json(Path("evals/baselines/quant.json").read_text()).meta
    # measured on the product default (the free config); a baseline on any other model must say so
    assert meta.model == meta.product_default_model == "openrouter/recursive/flint-v1.0:free"
    assert meta.is_product_default_model is True
    assert quant_report.main([]) == 0
    out = capsys.readouterr().out
    assert meta.model in out and "pass^k" in out and "95% interval" in out


def test_comparing_a_baseline_with_itself_moves_nothing(capsys):
    from evals import quant_report

    path = "evals/baselines/quant.json"
    assert quant_report.main([path, "--against", path]) == 0
    assert "None of the" in capsys.readouterr().out


# ── the math-tools switch in the eval ────────────────────────────────────────


def _offline_quant(tmp_path: Path, flag: str) -> dict[str, Any]:
    out = tmp_path / flag
    argv = ["--mode", "offline", "--suite", "quant", "--samples", "1", "--math-tools", flag,
            "--ledger", str(tmp_path / f"{flag}.json"), "--out-dir", str(out)]
    assert cli.main(argv) == 0
    report: dict[str, Any] = json.loads(next(out.glob("*.json")).read_text())
    return report


def test_the_same_suite_runs_with_tools_off_and_on_and_the_report_says_which(
    tmp_path: Path, block_network: None
):
    off, on = _offline_quant(tmp_path, "off"), _offline_quant(tmp_path, "on")
    assert (off["run_meta"]["math_tools"], on["run_meta"]["math_tools"]) == ("off", "on")
    assert off["run_meta"]["prompt_versions"] != on["run_meta"]["prompt_versions"]
    solve_off = off["suites"]["quant"]["groups"]["solve"]["cost_latency"]
    solve_on = on["suites"]["quant"]["groups"]["solve"]["cost_latency"]
    assert solve_off["tool_calls_per_turn_mean"] == 0 and solve_off["tool_error_rate"] is None
    assert solve_on["tool_calls_per_turn_mean"] > 0 and solve_on["tool_error_rate"] == 0.0
    assert solve_on["llm_calls_per_turn_mean"] > solve_off["llm_calls_per_turn_mean"]


def test_the_transcript_of_a_tool_turn_shows_the_calls_between_question_and_reply(
    tmp_path: Path, block_network: None
):
    report = _offline_quant(tmp_path, "on")
    attempt = next(
        a for a in report["suites"]["quant"]["rows"]["attempts"]
        if a["kind"] == "solve" and a["item_id"] == "q012"
    )
    roles = [m["role"] for m in attempt["transcript"]]
    assert roles == ["student", "tool_call", "tutor"]
    assert attempt["transcript"][1]["content"].startswith("math_evaluate(")
