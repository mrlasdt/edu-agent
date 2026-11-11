"""
The tool-use eval's own behaviour: the tasks are well-formed against the real tool list, the
offline run scores as designed, and the scoring tells a right call from a wrong one.
"""
import asyncio
import json
from typing import Any

import pytest

from evals import tool_use_eval as tue
from evals.agent_eval.budget import Ledger
from evals.agent_eval.transport import MeteredTransport
from mcp_servers.candidate_server import mcp


def test_every_task_names_a_real_tool_and_only_arguments_that_tool_takes():
    specs = {s["function"]["name"]: s["function"]["parameters"] for s in asyncio.run(
        tue.tool_specs(mcp))}
    tasks = tue.load_tasks()
    assert len(tasks) == 10
    for task in tasks:
        assert task.tool in specs, task.id
        assert set(task.args) <= set(specs[task.tool]["properties"]), task.id
        assert set(specs[task.tool]["required"]) <= set(task.args) | {"query"}, task.id
    assert {t.tool for t in tasks} == set(specs)


def test_the_model_is_shown_the_same_tool_list_a_client_gets():
    specs = asyncio.run(tue.tool_specs(mcp))
    assert [s["function"]["name"] for s in specs] == [t.name for t in asyncio.run(mcp.list_tools())]
    assert all(s["function"]["description"] for s in specs)


def test_a_call_is_right_only_for_the_expected_tool_arguments_and_query_words():
    task = tue.Task("t", "ask", "corpus_search", {"response_format": "detailed"}, ("essay",))
    assert task.accepts("corpus_search", {"query": "Issue ESSAY structure",
                                          "response_format": "detailed"})
    assert not task.accepts("corpus_search", {"query": "essay structure"})
    assert not task.accepts("corpus_search", {"query": "structure", "response_format": "detailed"})
    assert not task.accepts("practice_get_question", {"response_format": "detailed"})


def test_the_executor_hands_a_tool_error_back_with_the_valid_values():
    execute = tue.executor(mcp)
    result = asyncio.run(execute("practice_get_question",
                                 {"section": "quant", "question_type": "multiple_choice"}))
    assert set(result) == {"error"} and "'multiple_choice_single'" in result["error"]
    ok = asyncio.run(execute("essay_get_model", {"prompt_id": "aw001", "score_tier": 6}))
    assert ok["prompt_id"] == "aw001"


def test_offline_run_scores_the_keyword_router_and_counts_errors_per_call(tmp_path):
    transport = MeteredTransport(
        Ledger(tmp_path / "ledger.jsonl", 1.0), "t", fake_model=tue.keyword_router)
    summary = asyncio.run(tue.evaluate(transport, "offline-keyword-router"))
    assert (summary.right_tool.successes, summary.right_tool.n) == (10, 10)
    # the router never asks for "detailed" and does not know multiple_choice_single
    assert (summary.args_match.successes, summary.args_match.n) == (8, 10)
    assert [r.task_id for r in summary.results if not r.args_match] == ["tu08", "tu10"]
    assert summary.tool_error_rate.successes == 0 and summary.tool_error_rate.n == 10


def test_a_wrong_first_tool_and_an_error_are_both_counted(tmp_path):
    from evals.agent_eval.transport import FakeToolCalls

    def confused(model: str, messages: list[dict[str, Any]]) -> str | FakeToolCalls:
        if any(m.get("role") == "tool" for m in messages):
            return "done"
        return FakeToolCalls([("corpus_search", {"query": " "})])

    transport = MeteredTransport(Ledger(tmp_path / "ledger.jsonl", 1.0), "t", fake_model=confused)
    summary = asyncio.run(tue.evaluate(transport, "confused", tue.load_tasks()[1:3]))
    assert summary.right_tool.successes == 0 and summary.args_match.successes == 0
    assert (summary.tool_error_rate.successes, summary.tool_error_rate.n) == (2, 2)
    assert "query is empty" in summary.results[0].error_texts[0]


def test_baseline_json_has_the_metrics_with_n_and_interval(tmp_path):
    transport = MeteredTransport(
        Ledger(tmp_path / "ledger.jsonl", 1.0), "t", fake_model=tue.keyword_router)
    summary = asyncio.run(tue.evaluate(transport, "offline-keyword-router"))
    data = json.loads(json.dumps(tue.baseline_json(summary, "offline-keyword-router", "offline")))
    assert set(data["metrics"]) == {
        "tool_use.right_tool_rate", "tool_use.args_match_rate", "tool_use.tool_error_rate"}
    assert data["metrics"]["tool_use.args_match_rate"]["n"] == 10
    assert len(data["tasks"]) == 10


def test_committed_baseline_matches_the_current_task_list():
    data = json.loads(tue.DEFAULT_BASELINE.read_text())
    assert data["suite"] == "tool_use" and len(data["tasks"]) == len(tue.load_tasks())
    assert data["meta"]["mode"] == "live"
    assert not [t for t in data["tasks"] if t["task_id"] not in {x.id for x in tue.load_tasks()}]


@pytest.mark.parametrize("name", ["tool_use_eval"])
def test_module_runs_as_a_cli_offline(name, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_LEDGER_PATH", str(tmp_path / "ledger.jsonl"))
    from shared.src.shared.config import get_spend_settings

    get_spend_settings.cache_clear()
    assert tue.main(["--mode", "offline"]) == 0
    out = capsys.readouterr().out
    assert "right tool" in out and "100%" in out
    get_spend_settings.cache_clear()
