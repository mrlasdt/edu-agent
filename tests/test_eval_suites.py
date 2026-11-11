"""
The suite list and the backend: a suite is declared once, a mode is chosen once.
"""
import argparse
import json
from pathlib import Path

import pytest
from litellm import ModelResponse, ModelResponseStream

from evals.agent_eval import cli, transport
from evals.agent_eval.backend import (
    OFFLINE_MODEL,
    Backend,
    LiveBackend,
    OfflineBackend,
    missing_keys,
)
from evals.agent_eval.budget import Ledger
from evals.agent_eval.results import SUITE_MODELS, AwResult
from evals.agent_eval.suites import (
    SUITES,
    default_suite_names,
    regression_metrics,
    selected_suites,
    suite_names,
)
from tests.eval_result_fakes import eval_report

ALL_SUITES = ["quant", "quant_judge", "aw", "generation", "ielts_writing", "ielts_report",
              "ielts_adversarial"]
KEYS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")


# ── one list of suites ───────────────────────────────────────────────────────


def test_every_suite_is_listed_once_in_run_order():
    assert suite_names() == ALL_SUITES
    assert default_suite_names() == ["quant", "quant_judge", "aw", "generation"]


def test_a_stored_report_loads_each_suite_as_the_type_its_suite_declares():
    """`results.SUITE_MODELS` reads stored reports back; it must not drift from the list."""
    assert {s.name: s.result_type for s in SUITES} == SUITE_MODELS


def test_the_cli_offers_exactly_the_listed_suites(
    capsys: pytest.CaptureFixture[str], block_network: None
):
    with pytest.raises(SystemExit):
        cli.main(["--suite", "no_such_suite"])
    error = capsys.readouterr().err
    assert all(name in error for name in suite_names())


def test_suites_run_in_list_order_whatever_order_they_were_named():
    args = argparse.Namespace(suite=["aw", "quant", "ielts_adversarial", "generation"])
    assert [s.name for s in selected_suites(args)] == [
        "quant", "aw", "generation", "ielts_adversarial"
    ]


def test_a_dry_run_plans_every_suite_and_counts_only_the_selected_ones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], block_network: None
):
    argv = ["--suite", "aw", "--suite", "generation", "--max-items", "2", "--samples", "2",
            "--ledger", str(tmp_path / "l")]
    assert cli.main(argv) == 0
    plan = json.loads(capsys.readouterr().out)
    assert list(plan["items"]) == suite_names()
    assert plan["items"] == {name: 2 if name in ("aw", "generation") else 0
                             for name in suite_names()}
    assert plan["samples_per_quant_item"] is None
    # aw: 3 judge calls per item; generation: 8 agent and 2 judge calls per item
    assert plan["max_agent_calls"] == 2 * 8
    assert plan["max_judge_calls"] == 2 * 3 + 2 * 2


def test_the_plan_names_the_samples_the_quant_suite_repeats_each_item(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], block_network: None
):
    assert cli.main(["--suite", "quant", "--samples", "4", "--ledger", str(tmp_path / "l")]) == 0
    assert json.loads(capsys.readouterr().out)["samples_per_quant_item"] == 4


def test_regression_metrics_come_from_each_suite_that_ran():
    within = AwResult(
        n_judgements=4, within_tolerance=0.75, tolerance=1.0, mae=0.5, errors=0,
        units_total=4, units_dropped=0,
    )
    assert regression_metrics(eval_report(aw=within)) == {"judge_aw_within_tolerance": 0.75}
    assert regression_metrics(eval_report()) == {}


# ── one backend per run ──────────────────────────────────────────────────────


def _chain() -> list[str]:
    return cli.provider_chain("free")


def test_a_dry_run_has_no_backend():
    with pytest.raises(ValueError, match="dry-run"):
        Backend.for_mode("dry-run", chain=_chain(), judge_model=None, cache_dir=None)


def test_an_offline_backend_names_the_fake_model_and_needs_no_key(
    monkeypatch: pytest.MonkeyPatch,
):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    backend = Backend.for_mode("offline", chain=_chain(), judge_model=None, cache_dir=None)
    assert isinstance(backend, OfflineBackend) and not backend.live
    backend.preflight()
    assert backend.model == OFFLINE_MODEL and backend.report_chain == [OFFLINE_MODEL]
    assert backend.cache is None


def test_a_live_backend_runs_on_the_first_model_of_its_chain_and_caches_on_request(
    tmp_path: Path,
):
    chain = ["openrouter/a:free", "openrouter/b:free"]
    backend = Backend.for_mode("live", chain=chain, judge_model=None, cache_dir=tmp_path)
    assert isinstance(backend, LiveBackend) and backend.live
    assert backend.model == "openrouter/a:free" and backend.report_chain == chain
    assert backend.cache is not None
    uncached = Backend.for_mode("live", chain=chain, judge_model=None, cache_dir=None)
    assert uncached.cache is None


def test_a_live_backend_checks_the_judge_models_key_too(monkeypatch: pytest.MonkeyPatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "set")
    backend = Backend.for_mode(
        "live", chain=["openrouter/a:free"], judge_model="openai/gpt-4o-mini", cache_dir=None
    )
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        backend.preflight()


def test_a_key_either_provider_name_satisfies_is_not_reported_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    google = "gemini/gemma-4-26b-a4b-it"
    assert missing_keys([google]) == ["GOOGLE_API_KEY or GEMINI_API_KEY"]
    monkeypatch.setenv("GEMINI_API_KEY", "set")
    assert missing_keys([google]) == []
    assert missing_keys([google, "openai/gpt-4o-mini", "openai/gpt-4o"]) == ["OPENAI_API_KEY"]


async def test_the_backend_that_run_was_given_decides_the_mode_of_the_run(
    tmp_path: Path, block_network: None
):
    args = argparse.Namespace(
        mode="offline", suite=["aw"], provider="free", judge="llm", judge_model=None,
        max_items=1, max_items_given=True, samples=1, model=None, max_tokens=800,
        ledger=str(tmp_path / "l.json"), ceiling=20.0, min_coverage=0.8,
        cache_dir=str(tmp_path / "c"), no_cache=True, math_tools="off", context_budget="on",
        retrieval="upfront",
    )
    report, code = await cli.run(args)
    assert code == 0 and report.model_chain == [OFFLINE_MODEL] and report.mode == "offline"


# ── the replayed reply is litellm's own types ────────────────────────────────


async def test_a_replayed_stream_is_litellm_stream_chunks_with_the_usage_report():
    chunks = [c async for c in transport.MeteredTransport._wrap("hello", True, (3, 4))]
    assert all(isinstance(c, ModelResponseStream) for c in chunks)
    assert chunks[0].choices[0].delta.content == "hello"
    assert chunks[1].choices == []
    usage = chunks[1].usage  # type: ignore[attr-defined]
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (3, 4, 7)
    assert len([c async for c in transport.MeteredTransport._wrap("hello", True)]) == 1


def test_a_replayed_reply_is_a_litellm_response_with_its_tool_calls_and_usage():
    call = {"id": "c1", "name": "math_evaluate", "arguments": '{"expression": "1+1"}'}
    reply = transport._response("", [call], (30, 5))
    assert isinstance(reply, ModelResponse)
    message = reply.choices[0].message  # type: ignore[union-attr]
    assert message.content is None and transport._tool_calls_of(message) == [call]
    assert (reply.usage.prompt_tokens, reply.usage.completion_tokens) == (30, 5)  # type: ignore[attr-defined]
    plain = transport._response("hi")
    assert plain.choices[0].message.content == "hi"  # type: ignore[union-attr]
    assert getattr(plain, "usage", None) is None


async def test_an_offline_tool_call_comes_back_as_a_litellm_response(tmp_path: Path):
    """The fake model's tool call is asked of the Router as litellm's `mock_tool_calls` (its own
    kwarg: the mock-requests page documents only `mock_response`, so the tool-call mock is
    kept) and replayed to the agent as a litellm response."""
    from evals.agent_eval.backend import offline_model
    from evals.loader import load_golden_set

    item = next(i for i in load_golden_set("quant") if i["kind"] == "solve" and i.get("expression"))
    tool_transport = transport.MeteredTransport(
        Ledger(tmp_path / "l.json", 20.0), "t", fake_model=offline_model
    )
    reply = await tool_transport.acompletion(
        model=OFFLINE_MODEL,
        messages=[{"role": "system", "content": "You may call math_evaluate."},
                  {"role": "user", "content": item["question"]}],
    )
    assert isinstance(reply, ModelResponse)
    calls = transport._tool_calls_of(reply.choices[0].message)  # type: ignore[union-attr]
    assert [c["name"] for c in calls] in (["math_evaluate"], ["math_solve"])
    assert json.loads(calls[0]["arguments"]) in ({"expression": item["expression"]},
                                                 {"equation": item["expression"]})
