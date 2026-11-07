"""
Tests for the agent-in-the-loop eval. No test makes a real model call: LLM traffic goes
through fakes and the offline pipeline test runs with sockets disabled.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals.agent_eval import cli
from evals.agent_eval.budget import BudgetExceededError, Ledger, estimate_cost, is_free
from evals.agent_eval.cache import ResponseCache
from evals.agent_eval.judges import JevJudge, JudgeError, JudgeQuestion, LLMJudge
from evals.agent_eval.probes import (
    clean_answer,
    leaks_by_rule,
    perturb_answer,
)
from evals.agent_eval.transport import AllModelsFailedError, MeteredTransport
from evals.judges.quant_judge import grade_quant_item
from shared.src.shared.config import Settings
from tests.llm_fakes import (
    RATE_LIMITED,
    SERVER_ERROR,
    BadRequest,
    RouterPolicy,
    ScriptedRouter,
    calls,
    mock_router,
)

# ── Quant eval no longer grades the answer key against itself ────────────────


async def test_quant_judge_does_not_default_to_ground_truth():
    item = {"id": "q", "expression": "2+2", "ground_truth": "4"}
    result = await grade_quant_item(item)
    assert result.correct is False
    assert result.error == "no candidate answer"


async def test_quant_judge_grades_supplied_candidate():
    item = {"id": "q", "expression": "2+2", "ground_truth": "4"}
    assert (await grade_quant_item(item, candidate_answer="4")).correct is True
    assert (await grade_quant_item(item, candidate_answer="5")).correct is False


# ── probes ───────────────────────────────────────────────────────────────────


def test_leak_rule():
    assert leaks_by_rule("So the answer is 14.", "14")
    assert not leaks_by_rule("What is 3 times 4?", "14")
    assert leaks_by_rule("x is 2 or -2", "2,-2")
    assert not leaks_by_rule("try x = 2", "2,-2")  # needs every value
    assert not leaks_by_rule("about 114 units", "14")


def test_perturb_changes_answer():
    assert perturb_answer("2,3") == "2,4"
    assert perturb_answer("12.56") == "13.56"
    assert perturb_answer("none") is None


def test_clean_answer_strips_markup():
    assert clean_answer("$x = 2$.") == "x = 2"
    assert clean_answer("\\boxed{14}") == "14"


# ── budget ledger ────────────────────────────────────────────────────────────


def test_free_models_cost_zero():
    assert is_free("openrouter/google/gemma-4-31b-it:free")
    assert estimate_cost("openrouter/openrouter/free", 10_000, 10_000) == 0.0
    assert estimate_cost("openai/gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)


def test_ledger_persists_and_aborts_before_exceeding(tmp_path: Path):
    ledger = Ledger(tmp_path / "l.json", ceiling_usd=1.0)
    ledger.record("r1", "openai/gpt-4o", 0.9)
    assert Ledger(tmp_path / "l.json", 1.0).spent() == pytest.approx(0.9)  # survives reload
    ledger.ensure_capacity(0.0)  # free always passes
    with pytest.raises(BudgetExceededError):
        ledger.ensure_capacity(0.2)


# ── transport ────────────────────────────────────────────────────────────────


def _resp(text: str) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
    )


FREE, PAID = "openrouter/x:free", "openai/gpt-4o-mini"


def _transport(tmp_path: Path, router: Any, ceiling: float = 20.0) -> MeteredTransport:
    return MeteredTransport(Ledger(tmp_path / "l.json", ceiling), "run", router=router)


async def test_transport_falls_back_and_records_cost_under_the_model_that_answered(tmp_path: Path):
    router = mock_router({FREE: SERVER_ERROR, PAID: "hi"}, {FREE: [PAID]})
    t = _transport(tmp_path, router)
    out = await t.acompletion(
        model=FREE, messages=[{"role": "user", "content": "q"}], stream=True
    )
    chunks = [c.choices[0].delta.content async for c in out]
    assert chunks == ["hi"]
    assert set(calls(router)) == {FREE, PAID}
    assert t.records[0].model == PAID  # not the free model that was asked for
    assert t.records[0].cost_usd > 0 and t.spent_this_run == t.records[0].cost_usd
    assert Ledger(tmp_path / "l.json").spent() == pytest.approx(t.spent_this_run)
    assert [(f.model, f.kind) for f in t.failures] == [(FREE, "fallback")]


async def test_transport_aborts_on_budget_without_calling(tmp_path: Path):
    router = mock_router({"openai/gpt-4o": "dear"})
    t = _transport(tmp_path, router, ceiling=0.0000001)
    with pytest.raises(BudgetExceededError):
        await t.acompletion(model="openai/gpt-4o", messages=[{"role": "user", "content": "q"}])
    assert calls(router) == {}


async def test_the_budget_is_checked_at_the_dearest_model_of_the_chain(tmp_path: Path):
    """A free first model does not make the call free: its paid fallback could be reached."""
    router = mock_router({FREE: "free reply", "openai/gpt-4o": "dear"}, {FREE: ["openai/gpt-4o"]})
    t = _transport(tmp_path, router, ceiling=0.0000001)
    with pytest.raises(BudgetExceededError):
        await t.acompletion(model=FREE, messages=[{"role": "user", "content": "q" * 400}])
    assert calls(router) == {}


async def test_transport_all_models_failed(tmp_path: Path):
    router = mock_router({FREE: SERVER_ERROR, PAID: SERVER_ERROR}, {FREE: [PAID]})
    t = _transport(tmp_path, router)
    with pytest.raises(AllModelsFailedError) as exc:
        await t.acompletion(model=FREE, messages=[{"role": "user", "content": "q"}])
    assert FREE in str(exc.value) and PAID in str(exc.value)  # the chain that was tried
    assert t.failures[0].kind == "error"


async def test_transport_retries_rate_limits_on_the_same_model_before_falling_back(
    tmp_path: Path,
):
    router = mock_router({FREE: RATE_LIMITED, PAID: "ok"}, {FREE: [PAID]})
    t = _transport(tmp_path, router)
    out = await t.acompletion(model=FREE, messages=[{"role": "user", "content": "q"}])
    assert out.choices[0].message.content == "ok"
    assert calls(router) == {FREE: 1 + RouterPolicy().num_retries, PAID: 1}


async def test_the_default_transport_builds_a_router_with_the_chain_it_was_given(tmp_path: Path):
    t = MeteredTransport(Ledger(tmp_path / "l.json", 20.0), "run", fallbacks=[PAID])
    router = t.router_for(FREE)
    assert router is t.router_for(FREE)  # one per model, kept (and with it the cooldown state)
    from shared.src.shared.llm_router import EVAL_POLICY, configured_chain

    assert configured_chain(router, FREE) == [FREE, PAID]
    assert router.num_retries == EVAL_POLICY.num_retries
    assert router.cooldown_time == EVAL_POLICY.cooldown_time


# ── judges ───────────────────────────────────────────────────────────────────


async def test_llm_judge_parses_yes_no_and_score(monkeypatch: pytest.MonkeyPatch):
    replies = iter(["YES", "no.", "maybe", "4.5", "none"])

    async def fake(**kw: Any) -> Any:
        return _resp(next(replies))

    monkeypatch.setattr("litellm.acompletion", fake)
    judge = LLMJudge("m")
    q = JudgeQuestion("i", "m")
    assert (await judge.yes_no(q)).answer is True
    assert (await judge.yes_no(q)).answer is False
    with pytest.raises(JudgeError):
        await judge.yes_no(q)
    assert await judge.score(q, ["a"] * 6) == 4.5
    with pytest.raises(JudgeError):
        await judge.score(q, ["a"] * 6)


async def test_llm_judge_score_is_one_based_and_reaches_the_top_level(
    monkeypatch: pytest.MonkeyPatch,
):
    """The scale counts from 1, like AW_SCALE and the ETS levels, so the top level is reachable."""
    prompts: list[str] = []
    replies = iter(["1", "6", "9", "0"])

    async def fake(**kw: Any) -> Any:
        prompts.append(kw["messages"][0]["content"])
        return _resp(next(replies))

    monkeypatch.setattr("litellm.acompletion", fake)
    judge = LLMJudge("m")
    q = JudgeQuestion("i", "m")
    criteria = [f"{n} - level" for n in range(1, 7)]
    assert await judge.score(q, criteria) == 1.0
    assert await judge.score(q, criteria) == 6.0
    assert await judge.score(q, criteria) == 6.0  # clamped to the top level
    assert await judge.score(q, criteria) == 1.0  # clamped to the bottom level
    assert "1: 1 - level" in prompts[0] and "6: 6 - level" in prompts[0]
    assert "0:" not in prompts[0]


async def test_run_aw_scores_a_judge_that_answers_the_correct_level():
    """A judge that answers each essay's true level must score a perfect run, level 6 included."""
    from evals.agent_eval.runner import run_aw

    items = [{"prompt": "p", "anchor_essays": {str(n): f"essay {n}" for n in range(1, 7)}}]

    class ExactLLM(LLMJudge):
        async def _ask(self, system: str, q: JudgeQuestion) -> str:
            return str(int(q.facts["gold"]))

    out = await run_aw(items, ExactLLM("m"))
    assert out["mae"] == 0.0 and out["within_tolerance"] == 1.0
    assert out["n_judgements"] == 6


async def test_llm_judge_retries_an_empty_reply_once(monkeypatch: pytest.MonkeyPatch):
    replies = iter(["", "YES", "", "4", "", ""])
    calls = 0

    async def fake(**kw: Any) -> Any:
        nonlocal calls
        calls += 1
        return _resp(next(replies))

    monkeypatch.setattr("litellm.acompletion", fake)
    judge = LLMJudge("m")
    q = JudgeQuestion("i", "m")
    assert (await judge.yes_no(q)).answer is True and calls == 2
    assert await judge.score(q, ["a"] * 6) == 4.0 and calls == 4
    with pytest.raises(JudgeError):  # a second empty reply is still an error, with no third call
        await judge.yes_no(q)
    assert calls == 6


async def test_jev_judge_maps_choice_and_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sent: list[dict[str, Any]] = []
    bodies = iter(
        [
            {"answers": {"q": {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.9}}},
             "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.001}},
            {"answers": {"q": {"type": "score", "score": 1.99}},
             "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.001}},
        ]
    )

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def post(self, url: str, headers: Any, json: Any) -> Any:
            sent.append(json)
            body = next(bodies)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    monkeypatch.setattr("evals.agent_eval.judges.httpx.AsyncClient", lambda **kw: FakeClient())
    t = MeteredTransport(Ledger(tmp_path / "l.json", 20.0), "r")
    judge = JevJudge("k", t)
    verdict = await judge.yes_no(JudgeQuestion("is it?", "state"))
    assert verdict.answer is True and verdict.p_yes == 0.9
    # jev's 0-based expected index 1.99 is level 2.99 on the 1-based scale
    level = await judge.score(JudgeQuestion("rate", "state"), ["lo", "mid", "hi"])
    assert level == pytest.approx(2.99)
    assert sent[0]["questions"]["q"]["type"] == "choice"
    assert sent[1]["questions"]["q"]["type"] == "score"
    assert t.spent_this_run == pytest.approx(0.002)


# ── cli ──────────────────────────────────────────────────────────────────────


def test_provider_chains_free_first_openai_last():
    assert all(is_free(m) for m in cli.provider_chain("free"))
    chain = cli.provider_chain("free+openai")
    assert chain[-1].startswith("openai/") and is_free(chain[0])
    assert cli.provider_chain("openai") == [cli.OPENAI_MODEL]


def test_dry_run_makes_no_calls_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    block_network: None,
):
    ledger = tmp_path / "l.json"
    assert cli.main(["--ledger", str(ledger), "--out-dir", str(tmp_path / "runs")]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "dry-run" and plan["max_agent_calls"] > 0
    assert not ledger.exists() and not (tmp_path / "runs").exists()


def test_the_eval_cli_defaults_to_the_configured_spend_cap_and_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    block_network: None,
):
    ledger = tmp_path / "shared.json"
    Ledger(ledger, 10.0).record("mock", "gpt-4.1-mini", 6.0)
    settings = Settings(ai_spend_cap_usd=10.0, ai_ledger_path=str(ledger))
    monkeypatch.setattr("shared.src.shared.budget.get_settings", lambda: settings)
    assert cli.main(["--out-dir", str(tmp_path / "runs")]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["ledger_ceiling_usd"] == 10.0 and plan["ledger_spent_usd"] == 6.0
    assert cli.main(["--ceiling", "2", "--ledger", str(tmp_path / "other.json")]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["ledger_ceiling_usd"] == 2.0 and plan["ledger_spent_usd"] == 0.0


def test_offline_pipeline_runs_real_agent_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    block_network: None,
):
    code = cli.main(
        ["--mode", "offline", "--ledger", str(tmp_path / "l.json"), "--out-dir", str(tmp_path)]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    quant = out["suites"]["quant_judge"]
    assert quant["solve"]["accuracy"] == 1.0  # fake model returns the golden answer
    assert quant["tutor"]["leak_rate_rule"] == 0.0
    assert out["cost_latency"]["total_usd"] == 0.0
    assert "rows" not in quant  # printed summary is aggregate-only
    seeded = out["suites"]["quant"]  # the failure-seeded suite runs by default too
    assert seeded["security"]["value"] == 1.0
    assert seeded["groups"]["solve"]["pass_hat_k"]["value"] == 1.0
    assert "rows" not in seeded
    assert not (tmp_path / "l.json").exists()  # free/offline never touches the ledger


def test_live_requires_keys_by_name(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--mode", "live", "--provider", "free"])
    assert "OPENROUTER_API_KEY" in str(exc.value)


# ── cooldown, failure reasons, cache ─────────────────────────────────────────


async def test_a_model_that_keeps_failing_is_skipped_by_later_calls(tmp_path: Path):
    """Replaces the old process-lifetime `exhausted` set: the Router's cooldown does the job."""
    policy = RouterPolicy(num_retries=0, retry_after=0, allowed_fails=1, cooldown_time=60)
    router = mock_router({FREE: RATE_LIMITED, PAID: "ok"}, {FREE: [PAID]}, policy)
    t = _transport(tmp_path, router)
    for n in range(4):
        await t.acompletion(model=FREE, messages=[{"role": "user", "content": f"q{n}"}])
    assert calls(router) == {FREE: 2, PAID: 4}  # failed twice, then skipped for the cooldown


class _LeakyError(BadRequest):
    """A provider error whose text carries keys that must never reach a report."""

    def __init__(self) -> None:
        super().__init__()
        self.message = "404 guardrail: no endpoints, key sk-abcdef123456789 Bearer tok123"

    def __str__(self) -> str:
        return self.message


async def test_failure_reason_is_recorded_and_redacted(tmp_path: Path):
    router = mock_router({FREE: _LeakyError()})
    t = _transport(tmp_path, router)
    with pytest.raises(AllModelsFailedError) as exc:
        await t.acompletion(model=FREE, messages=[{"role": "user", "content": "q"}])
    assert "guardrail" in str(exc.value) and "_LeakyError" in str(exc.value)
    assert "sk-abcdef" not in str(exc.value) and "tok123" not in str(exc.value)
    summary = t.failure_summary()
    assert summary[f"{FREE} [error]"]["count"] == 1
    assert "sk-abcdef" not in json.dumps(summary)


async def test_response_cache_serves_repeat_calls_without_calling_or_spending(tmp_path: Path):
    router = mock_router({PAID: "cached text"})
    t = _transport(tmp_path, router)
    t.cache = ResponseCache(tmp_path / "cache")
    msgs = [{"role": "user", "content": "q"}]
    first = await t.acompletion(model=PAID, messages=msgs, temperature=0)
    spent = t.spent_this_run
    second = await t.acompletion(model=PAID, messages=msgs, temperature=0)
    assert first.choices[0].message.content == second.choices[0].message.content
    assert calls(router) == {PAID: 1} and t.spent_this_run == spent
    assert t.cache.hits == 1
    other = [{"role": "user", "content": "different"}]
    await t.acompletion(model=PAID, messages=other, temperature=0)
    assert calls(router) == {PAID: 2}


def test_cache_dir_is_gitignored():
    assert ".eval-cache/" in Path(".gitignore").read_text().splitlines()


def test_litellm_banner_is_suppressed():
    import litellm

    assert litellm.suppress_debug_info is True


# ── coverage gate ────────────────────────────────────────────────────────────


def test_coverage_failures_flags_mostly_dropped_suites():
    from evals.agent_eval.runner import coverage_failures

    report = {
        "suites": {
            "quant": {"units_total": 20, "units_dropped_model_unavailable": 19},
            "aw": {"units_total": 10, "units_dropped_model_unavailable": 1},
        }
    }
    assert coverage_failures(report, 0.8) == {"quant": pytest.approx(0.05)}
    assert coverage_failures(report, 0.0) == {}


def test_a_suite_result_without_units_total_fails_the_coverage_check():
    """A suite that forgets `units_total` must not read as full coverage."""
    from evals.agent_eval.runner import coverage_failures

    report = {"suites": {"generation": {"units_dropped_model_unavailable": 7}}}
    assert coverage_failures(report, 0.8) == {"generation": 0.0}
    assert coverage_failures(report, 0.0) == {}


async def test_generation_run_with_the_model_down_fails_the_coverage_gate():
    """End to end through run_generation: every unit dropped must trip the real gate."""
    from evals.agent_eval import generation
    from evals.agent_eval.runner import coverage_failures

    class Down:
        async def run(self, request: Any) -> Any:
            raise AllModelsFailedError("model chain unavailable")

    requests = [
        generation.GenerationRequest.model_validate(i)
        for i in cli.load_golden_set("quant_generation")
    ][:4]
    transport = MeteredTransport(
        Ledger(Path("/nonexistent/ledger.json"), 20.0), "t", fake_model=cli._offline_model
    )
    skill = cli.runner.in_process_verify_skill()
    out = await generation.run_generation(
        requests, {"baseline": Down(), "pipeline": Down()}, cli._offline_judge(), skill, transport
    )
    assert out["units_total"] == 8 and out["units_dropped_model_unavailable"] == 8
    assert coverage_failures({"suites": {"generation": out}}, 0.8) == {"generation": 0.0}


async def test_quant_judge_suite_never_reaches_the_corpus_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block_network: None
):
    """The quant_judge agent gets the same offline notes skill as the quant suite."""
    from services.agent.src.agent.skills.retrieve_corpus import RetrieveCorpusSkill

    corpus_calls: list[str] = []

    async def corpus(self: Any, query: str, **kw: Any) -> list[dict[str, Any]]:
        corpus_calls.append(query)
        return []

    monkeypatch.setattr(RetrieveCorpusSkill, "_call_corpus", corpus)
    args = cli.argparse.Namespace(
        mode="offline", suite=["quant_judge"], provider="free", judge="llm", judge_model=None,
        max_items=3, max_items_given=True, samples=1, model=None, max_tokens=800,
        ledger=str(tmp_path / "l.json"), ceiling=20.0, out_dir=str(tmp_path),
        baseline=None, write_baseline=None, min_coverage=0.8,
        cache_dir=str(tmp_path / "c"), no_cache=True, math_tools="on",
    )
    report, code = await cli.run(args)
    assert code == 0 and report["suites"]["quant_judge"]["n_items"] == 3
    assert corpus_calls == []


async def test_failed_run_exits_nonzero_and_emits_no_regression_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    async def down(**kw: Any) -> Any:
        raise RuntimeError("down")

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr("litellm.acompletion", down)
    args = cli.argparse.Namespace(
        mode="live", suite=["quant", "aw"], provider="openai", judge="llm", judge_model=None,
        max_items=5, max_items_given=False, samples=1, model=None, max_tokens=800,
        ledger=str(tmp_path / "l.json"), ceiling=20.0, out_dir=str(tmp_path),
        baseline=None, write_baseline=None, min_coverage=0.8,
        cache_dir=str(tmp_path / "c"), no_cache=True, math_tools="on",
    )
    report, code = await cli.run(args)
    assert code == 2
    assert report["coverage_gate"]["passed"] is False
    assert report["regression_metrics"] == {}
    assert report["call_failures"]


# ── per-question-type answers, QC grading, calibration probes ────────────────


def test_qc_letter_and_normalisation():
    from evals.agent_eval.probes import normalize_for_type, qc_letter

    assert qc_letter("A") == "A" and qc_letter("(B)") == "B"
    assert qc_letter("Quantity A is greater than Quantity B") == "A"
    assert qc_letter("The two quantities are equal") == "C"
    assert qc_letter("The relationship cannot be determined") == "D"
    assert qc_letter("14") is None
    assert normalize_for_type("Quantity B is greater", "quantitative_comparison") == "B"
    assert normalize_for_type("**2, -2**", "numeric_entry") == "2, -2"


async def test_qc_items_are_graded_by_verdict_letter_from_the_expression():
    item = {"id": "q", "question_type": "quantitative_comparison",
            "expression": "3**2 - 2**3", "ground_truth": "A"}
    assert (await grade_quant_item(item, candidate_answer="A")).correct is True
    assert (await grade_quant_item(item, candidate_answer="(A)")).correct is True
    assert (await grade_quant_item(item, candidate_answer="B")).correct is False
    assert (await grade_quant_item(item, candidate_answer="1")).correct is False


async def test_golden_qc_labels_agree_with_their_expressions():
    from evals.loader import load_golden_set

    for item in load_golden_set("quant"):
        if item["kind"] == "solve" and item["question_type"] == "quantitative_comparison":
            assert item["ground_truth"] in "ABCD"
            assert (await grade_quant_item(item, candidate_answer=item["ground_truth"])).correct


def test_qc_leak_rule_ignores_the_quantity_label():
    hint = "Compare Quantity A with Quantity B. What is 3^2?"
    assert not leaks_by_rule(hint, "A", "quantitative_comparison")
    assert leaks_by_rule("Quantity A is greater.", "A", "quantitative_comparison")
    assert leaks_by_rule("The correct answer is A.", "A", "quantitative_comparison")


def test_calibration_solutions_show_working():
    from evals.agent_eval.probes import correctness_calibration

    item = {"id": "q", "question": "Solve", "expression": "x**2 - 5*x + 6 = 0",
            "ground_truth": "2,3"}
    material = [q.material for q, _ in correctness_calibration(item)]
    assert all("x**2 - 5*x + 6 = 0" in m and "evaluates to" in m for m in material)
    assert "**Answer: 2,3**" in material[0] and "**Answer: 2,4**" in material[1]


async def test_transport_replays_a_tool_call_and_caches_it(tmp_path: Path):
    from evals.agent_eval.cache import ResponseCache

    calls: list[dict[str, Any]] = []

    def asks_for_a_tool() -> Any:
        function = SimpleNamespace(name="math_evaluate", arguments='{"expression":"1+1"}')
        call = SimpleNamespace(id="c1", function=function)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))],
            usage=SimpleNamespace(prompt_tokens=30, completion_tokens=5),
        )

    async def real(**kw: Any) -> Any:
        calls.append(kw)
        return asks_for_a_tool()

    t = MeteredTransport(
        Ledger(tmp_path / "l.json", 1.0), "run", cache=ResponseCache(tmp_path / "c"),
        router=ScriptedRouter(real),
    )
    tools = [{"type": "function", "function": {"name": "math_evaluate"}}]
    messages = [{"role": "user", "content": "1+1?"}]
    live = await t.acompletion(model="openai/gpt-4o-mini", messages=messages, tools=tools)
    again = await t.acompletion(model="openai/gpt-4o-mini", messages=messages, tools=tools)
    for reply in (live, again):
        (call,) = reply.choices[0].message.tool_calls
        assert (call.id, call.function.name) == ("c1", "math_evaluate")
        assert call.function.arguments == '{"expression":"1+1"}'
        assert (reply.usage.prompt_tokens, reply.usage.completion_tokens) == (30, 5)
    assert len(calls) == 1  # the second request came from the cache
    # the same messages without tools are a different request: the cache must not answer it
    await t.acompletion(model="openai/gpt-4o-mini", messages=messages)
    assert len(calls) == 2
