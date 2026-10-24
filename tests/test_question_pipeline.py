"""
Tests for the LangGraph question-generation pipeline and its eval target. No test makes a
real model call: the pipeline runs against scripted completers or the offline fake model.
"""
import asyncio
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals.agent_eval import cli, generation, offline_generation
from evals.agent_eval.budget import BudgetExceededError, Ledger
from evals.agent_eval.judges import HeuristicJudge
from evals.agent_eval.runner import InProcessVerifySkill, write_model_config
from evals.agent_eval.transport import MeteredTransport
from evals.human_review_queue import HumanReviewQueue
from evals.loader import load_golden_set
from services.agent.src.agent.pipeline import tools
from services.agent.src.agent.pipeline.baseline import SingleCallGenerator
from services.agent.src.agent.pipeline.factory import build_baseline, build_pipeline
from services.agent.src.agent.pipeline.graph import QuestionPipeline, rule_based_type
from services.agent.src.agent.pipeline.llm import (
    LiteLLMCompleter,
    extract_json,
    load_prompt,
)
from services.agent.src.agent.pipeline.schema import (
    GenerationRequest,
    QuestionItem,
    QuestionType,
)
from shared.src.shared.config import load_model_config

SKILL = InProcessVerifySkill()

NUMERIC = {
    "question_type": "numeric_entry",
    "stem": "Find all real solutions of x^2 - 7x + 12 = 0. Enter them separated by commas.",
    "expression": "x**2 - 7*x + 12 = 0",
    "answer": "3,4",
}
MC_SINGLE = {
    "question_type": "multiple_choice_single",
    "stem": "What is the value of 2 + 3 * 4?",
    "choices": ["12", "14", "20", "24", "10"],
    "expression": "2 + 3*4",
    "answer": "14",
}
MC_MULTI = {
    "question_type": "multiple_choice_multiple",
    "stem": "Which of the following values satisfy x^2 = 49? Select all that apply.",
    "choices": ["7", "-7", "8", "-8", "0"],
    "expression": "x**2 - 49 = 0",
    "answer": "7,-7",
}
QC = {
    "question_type": "quantitative_comparison",
    "stem": "Quantity A: 3^2\nQuantity B: 2^3",
    "quantity_a": "3**2",
    "quantity_b": "2**3",
    "answer": "A",
}


def item(base: dict[str, Any], **changes: Any) -> QuestionItem:
    return QuestionItem.model_validate({**base, **changes})


# ── scripted completer ───────────────────────────────────────────────────────


class Scripted:
    """Replies per role from queues (the last reply repeats); records every call."""

    def __init__(self, planner: list[str], generator: list[str], critic: list[str] | None = None):
        self.queues = {
            "question_planner": planner,
            "question_generator": generator,
            "question_critic": critic or [json.dumps({"ambiguous": False, "leaks_answer": False})],
        }
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    def model_for(self, role: str) -> str:
        return f"scripted/{role}"

    async def complete(self, role: str, messages: list[dict[str, Any]], trace_id: str) -> str:
        self.calls.append((role, messages))
        queue = self.queues[role]
        n = sum(1 for r, _ in self.calls if r == role)
        return queue[min(n, len(queue)) - 1]


def plan_reply(qtype: str) -> str:
    return json.dumps(
        {"question_type": qtype, "learning_objective": "o", "constraints": ["integer data"]}
    )


def request(qtype: str | None = "numeric_entry", rid: str = "r1") -> GenerationRequest:
    return GenerationRequest.model_validate(
        {"request_id": rid, "topic": "quadratics", "question_type": qtype}
    )


def steps(result: Any) -> list[str]:
    return [e.step for e in result.trace]


# ── parsing ──────────────────────────────────────────────────────────────────


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```\nDone') == {"a": 1}
    assert extract_json("no json here") is None
    assert extract_json('{"a": {"b": 2}} trailing') == {"a": {"b": 2}}


def test_rule_based_type():
    assert rule_based_type("comparison of powers") == QuestionType.quantitative_comparison
    assert rule_based_type("linear equations") == QuestionType.numeric_entry
    assert rule_based_type("fractions") == QuestionType.multiple_choice_single


# ── deterministic tools ──────────────────────────────────────────────────────


async def test_solver_accepts_correct_keys_for_every_type():
    for base in (NUMERIC, MC_SINGLE, MC_MULTI, QC):
        assert all(c.passed for c in await tools.run_checks(item(base), SKILL)), base


async def test_solver_rejects_wrong_keys():
    for base, wrong in ((NUMERIC, "3,5"), (MC_SINGLE, "20"), (MC_MULTI, "7,8"), (QC, "B")):
        checks = await tools.run_checks(item(base, answer=wrong), SKILL)
        assert not all(c.passed for c in checks), base


async def test_solver_reports_unsolvable_expression():
    check = await tools.solve_check(item(NUMERIC, expression="x ** ("), SKILL)
    assert not check.passed and "cannot solve" in check.detail


async def test_qc_d_requires_a_genuinely_indeterminate_comparison():
    indeterminate = item(QC, quantity_a="x**2", quantity_b="x", answer="D")  # sign varies with x
    assert (await tools.solve_check(indeterminate, SKILL)).passed
    determinate = item(QC, quantity_a="x**2 + 1", quantity_b="x**2", answer="D")  # always A
    assert not (await tools.solve_check(determinate, SKILL)).passed


async def test_distractor_check_flags_equal_and_correct_distractors():
    dup = item(MC_SINGLE, choices=["12", "14", "14.0", "24", "10"])
    assert "numerically equal" in (await tools.distractor_check(dup, SKILL)).detail
    also_correct = item(MC_MULTI, answer="7")  # -7 is a valid solution but not keyed
    check = await tools.distractor_check(also_correct, SKILL)
    assert not check.passed


def test_structure_and_leak_checks():
    assert not tools.structure_check(item(MC_SINGLE, choices=["1", "2"])).passed
    assert not tools.structure_check(item(NUMERIC, choices=["1", "2", "3", "4"])).passed
    assert not tools.structure_check(item(MC_MULTI, stem="Which values satisfy x^2 = 49?")).passed
    assert not tools.structure_check(item(QC, answer="E")).passed
    assert not tools.leak_check(item(NUMERIC, stem=NUMERIC["stem"] + " The answer is 3,4.")).passed
    assert not tools.leak_check(item(MC_SINGLE, choices=["12", "14 (correct)", "20", "24"])).passed
    assert tools.leak_check(item(MC_SINGLE)).passed


# ── graph: happy path, routing, revise loop, bounded retries ─────────────────


async def test_accepts_a_good_item_and_traces_every_step():
    llm = Scripted([plan_reply("numeric_entry")], [json.dumps(NUMERIC)])
    result = await QuestionPipeline(llm, SKILL).run(request())
    assert result.status == "accepted" and result.item is not None
    assert result.attempts == 1 and result.llm_calls == 3  # planner + generator + critic
    assert steps(result) == ["plan", "generate", "verify", "critic", "accept"]
    verify = next(e for e in result.trace if e.step == "verify")
    assert {c["name"] for c in verify.data["checks"]} == {
        "structure", "leak", "solver", "distractors"
    }
    json.dumps([e.model_dump(mode="json") for e in result.trace])  # the trace is serialisable


async def test_router_sends_each_type_to_its_expert():
    cases = {
        "numeric_entry": (NUMERIC, "numeric-entry expert"),
        "multiple_choice_multiple": (MC_MULTI, "multiple-choice expert"),
        "quantitative_comparison": (QC, "quantitative-comparison expert"),
    }
    for qtype, (base, marker) in cases.items():
        llm = Scripted([plan_reply(qtype)], [json.dumps(base)])
        result = await QuestionPipeline(llm, SKILL).run(request(None))
        assert result.status == "accepted", qtype
        generator_system = next(m for r, m in llm.calls if r == "question_generator")[0]
        assert marker in generator_system["content"]
        plan_event = result.trace[0]
        assert plan_event.data["routed_to"].endswith("_expert")


async def test_planner_cannot_override_a_fixed_type_and_garbage_falls_back_to_rules():
    llm = Scripted([plan_reply("quantitative_comparison")], [json.dumps(NUMERIC)])
    fixed = await QuestionPipeline(llm, SKILL).run(request("numeric_entry"))
    assert fixed.plan is not None and fixed.plan.question_type == QuestionType.numeric_entry

    llm = Scripted(["I refuse to answer in JSON"], [json.dumps(MC_SINGLE)])
    result = await QuestionPipeline(llm, SKILL).run(request(None))
    assert result.trace[0].data["rule_based_fallback"] is True
    assert result.plan is not None and result.plan.question_type == rule_based_type("quadratics")
    assert result.status == "accepted"


async def test_revise_loop_feeds_failures_back_and_recovers():
    wrong = {**NUMERIC, "answer": "3,5"}
    llm = Scripted([plan_reply("numeric_entry")], [json.dumps(wrong), json.dumps(NUMERIC)])
    result = await QuestionPipeline(llm, SKILL).run(request())
    assert result.status == "accepted" and result.attempts == 2
    assert steps(result) == [
        "plan", "generate", "verify", "revise", "generate", "verify", "critic", "accept"
    ]
    second = [m for r, m in llm.calls if r == "question_generator"][1]
    payload = json.loads(second[-1]["content"])
    assert any("solver" in f for f in payload["feedback"])
    assert json.loads(payload["previous_attempt"])["answer"] == "3,5"


async def test_non_json_reply_is_a_schema_failure_then_recovers():
    llm = Scripted([plan_reply("numeric_entry")], ["Here you go!", json.dumps(NUMERIC)])
    result = await QuestionPipeline(llm, SKILL).run(request())
    assert result.status == "accepted"
    verify = next(e for e in result.trace if e.step == "verify")
    assert verify.data["checks"][0]["name"] == "schema"


async def test_retries_are_bounded_then_escalated_to_review():
    wrong = {**NUMERIC, "answer": "3,5"}
    llm = Scripted([plan_reply("numeric_entry")], [json.dumps(wrong)])
    result = await QuestionPipeline(llm, SKILL, max_attempts=3).run(request())
    assert result.status == "needs_review" and result.item is None
    assert result.candidate is not None and result.attempts == 3
    assert result.llm_calls == 1 + 3  # planner + 3 generations, the critic never ran
    assert steps(result)[-1] == "escalate" and steps(result).count("generate") == 3
    assert result.failures and "solver" in result.failures[0]


async def test_critic_flag_triggers_revision_and_unparseable_critic_does_not_block():
    flag = json.dumps({"ambiguous": True, "leaks_answer": False, "issues": ["two readings"]})
    ok = json.dumps({"ambiguous": False, "leaks_answer": False, "issues": []})
    llm = Scripted([plan_reply("numeric_entry")], [json.dumps(NUMERIC)], [flag, ok])
    result = await QuestionPipeline(llm, SKILL).run(request())
    assert result.status == "accepted" and result.attempts == 2
    revise = next(e for e in result.trace if e.step == "revise")
    assert revise.data["feedback"] == ["critic: two readings"]

    llm = Scripted([plan_reply("numeric_entry")], [json.dumps(NUMERIC)], ["not json"])
    assert (await QuestionPipeline(llm, SKILL).run(request())).status == "accepted"


async def test_baseline_ships_whatever_comes_back_unchecked():
    wrong = {**NUMERIC, "answer": "3,5"}
    llm = Scripted([plan_reply("numeric_entry")], [json.dumps(wrong)])
    result = await SingleCallGenerator(llm).run(request())
    assert result.status == "unchecked" and result.item is not None and result.llm_calls == 1
    llm = Scripted([], ["no json"])
    assert (await SingleCallGenerator(llm).run(request())).status == "unparseable"


# ── provider layer and budget ────────────────────────────────────────────────


def test_prompts_and_roles_exist_for_every_pipeline_step():
    cfg = load_model_config()
    for role in ("question_planner", "question_generator", "question_critic"):
        assert cfg.litellm_model(role)
    for name in ("planner", "numeric_expert", "choice_expert", "comparison_expert", "critic"):
        assert load_prompt(name, cfg.prompt_version("question_pipeline", name))
    assert load_prompt("baseline", cfg.prompt_version("question_pipeline", "baseline"))


async def test_completer_uses_role_model_and_trace_metadata(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    async def fake(**kw: Any) -> Any:
        seen.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])

    monkeypatch.setattr("litellm.acompletion", fake)
    completer = LiteLLMCompleter()
    reply = await completer.complete("question_critic", [{"role": "user", "content": "x"}], "t1")
    assert reply == "hi"
    assert seen["model"] == completer.model_for("question_critic")
    assert seen["stream"] is False and seen["metadata"]["trace_id"] == "t1"


async def test_pipeline_through_the_metered_transport_aborts_on_budget(tmp_path: Path):
    """A paid role with a ceiling that cannot cover one call never reaches the network."""

    async def real(**kw: Any) -> Any:
        raise AssertionError("must not be called")

    transport = MeteredTransport(Ledger(tmp_path / "l.json", ceiling_usd=1e-9), "run")
    transport._real = real
    pipeline = build_pipeline(SKILL, write_model_config("openai/gpt-4o"))
    with transport.installed(), pytest.raises(BudgetExceededError):
        await pipeline.run(request())
    assert not (tmp_path / "l.json").exists()


# ── the eval target: baseline vs pipeline on the offline fake ────────────────


def _offline_run(max_items: int = 12) -> dict[str, Any]:
    requests = [
        GenerationRequest.model_validate(i) for i in load_golden_set("quant_generation")
    ][:max_items]
    transport = MeteredTransport(
        Ledger(Path("/nonexistent/ledger.json"), 20.0), "t", fake_model=cli._offline_model
    )
    config = write_model_config("offline/fake")
    judge = HeuristicJudge(lambda q: bool(q.facts["gold"]), lambda q, c: 0.0)
    queue = HumanReviewQueue()

    async def go() -> dict[str, Any]:
        with transport.installed():
            return await generation.run_generation(
                requests,
                {"baseline": build_baseline(config), "pipeline": build_pipeline(SKILL, config)},
                judge, SKILL, transport, queue,
            )

    out = asyncio.run(go())
    out["queue_items"] = queue.list()
    return out


def test_pipeline_beats_the_single_call_baseline_offline():
    out = _offline_run()
    base, pipe = out["targets"]["baseline"], out["targets"]["pipeline"]
    assert pipe["valid_yield"] > base["valid_yield"] + 0.5
    assert pipe["shipped_defect_rate"] == 0.0 and base["shipped_defect_rate"] > 0.5
    assert base["unparseable_rate"] > 0 and pipe["unparseable_rate"] == 0
    assert pipe["mean_llm_calls"] > base["mean_llm_calls"] == 1.0  # the price of the loop
    assert out["comparison"]["valid_yield_gain"] == pytest.approx(
        pipe["valid_yield"] - base["valid_yield"]
    )


def test_unrepairable_items_go_to_review_not_to_the_candidate():
    out = _offline_run()
    stubborn = [r for r in out["rows"]["pipeline"] if r["status"] == "needs_review"]
    assert [r["request_id"] for r in stubborn] == ["g06"]  # FLAW_CYCLE: id % 7 == 6
    assert out["review_queue"] == 1
    assert out["queue_items"][0]["reason"] == "generation_needs_review"
    assert all(r["defects"] == [] for r in out["rows"]["pipeline"] if r["status"] == "accepted")
    assert stubborn[0]["trace"][-1]["step"] == "escalate"


def test_fake_flaw_schedule_is_deterministic():
    msgs = [
        {"role": "system", "content": "You are a GRE Quantitative Reasoning question writer."},
        {"role": "user", "content": json.dumps(
            {"request_id": "g01", "plan": {"question_type": "numeric_entry"}}
        )},
    ]
    first = offline_generation.respond("m", msgs)
    assert first == offline_generation.respond("m", msgs)
    assert offline_generation.respond("m", [{"role": "system", "content": "tutor"}]) is None


def test_cli_offline_generation_suite_reports_before_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def deny(*a: Any, **k: Any) -> Any:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", deny)
    code = cli.main(
        ["--mode", "offline", "--suite", "generation", "--max-items", "12",
         "--ledger", str(tmp_path / "l.json"), "--out-dir", str(tmp_path)]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    gen = out["suites"]["generation"]
    assert gen["comparison"]["valid_yield_gain"] > 0.5
    assert "rows" not in gen  # printed summary is aggregate only
    assert out["regression_metrics"]["gen_pipeline_valid_yield"] > 0.9
    run_file = next(tmp_path.glob("*.json"))
    rows = json.loads(run_file.read_text())["suites"]["generation"]["rows"]
    assert rows["pipeline"][0]["trace"]  # the saved report keeps the per-item audit trail
    assert not (tmp_path / "l.json").exists()


def test_dry_run_plan_counts_generation_calls(capsys: pytest.CaptureFixture[str]):
    assert cli.main(["--suite", "generation"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["items"]["generation"] == 5 and plan["max_agent_calls"] == 5 * 8


def test_item_schema_tolerates_numbers_and_nulls_from_real_models():
    parsed = QuestionItem.model_validate(
        {**MC_SINGLE, "choices": [9, 10, 14], "answer": 14, "quantity_a": None, "explanation": None}
    )
    assert parsed.choices == ["9", "10", "14"] and parsed.answer == "14"
    assert parsed.quantity_a == "" and parsed.explanation == ""
    assert QuestionItem.model_validate({**NUMERIC, "choices": None}).choices == []
