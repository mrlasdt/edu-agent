"""
Tests for the agent-in-the-loop eval. No test makes a real model call: LLM traffic goes
through fakes and the offline pipeline test runs with sockets disabled.
"""
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals.agent_eval import cli
from evals.agent_eval.budget import BudgetExceededError, Ledger, estimate_cost, is_free
from evals.agent_eval.judges import JevJudge, JudgeError, JudgeQuestion, LLMJudge
from evals.agent_eval.probes import (
    clean_answer,
    leaks_by_rule,
    perturb_answer,
)
from evals.agent_eval.transport import AllModelsFailedError, MeteredTransport
from evals.judges.quant_judge import grade_quant_item

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


def _transport(tmp_path: Path, real: Any, fallbacks: list[str], ceiling: float = 20.0):
    t = MeteredTransport(Ledger(tmp_path / "l.json", ceiling), "run", fallbacks=fallbacks)
    t._real = real
    return t


async def test_transport_falls_back_and_records_cost(tmp_path: Path):
    calls: list[str] = []

    async def real(**kw: Any) -> Any:
        calls.append(kw["model"])
        if kw["model"].endswith(":free"):
            raise RuntimeError("rate limited")
        return _resp("hi")

    t = _transport(tmp_path, real, ["openai/gpt-4o-mini"])
    out = await t.acompletion(
        model="openrouter/x:free", messages=[{"role": "user", "content": "q"}], stream=True
    )
    chunks = [c.choices[0].delta.content async for c in out]
    assert chunks == ["hi"]
    assert calls == ["openrouter/x:free", "openai/gpt-4o-mini"]
    assert t.records[0].cost_usd > 0 and t.spent_this_run == t.records[0].cost_usd
    assert Ledger(tmp_path / "l.json").spent() == pytest.approx(t.spent_this_run)


async def test_transport_aborts_on_budget_without_calling(tmp_path: Path):
    async def real(**kw: Any) -> Any:
        raise AssertionError("must not be called")

    t = _transport(tmp_path, real, [], ceiling=0.0000001)
    with pytest.raises(BudgetExceededError):
        await t.acompletion(model="openai/gpt-4o", messages=[{"role": "user", "content": "q"}])


async def test_transport_all_models_failed(tmp_path: Path):
    async def real(**kw: Any) -> Any:
        raise RuntimeError("down")

    t = _transport(tmp_path, real, ["openai/gpt-4o-mini"])
    with pytest.raises(AllModelsFailedError):
        await t.acompletion(model="openrouter/x:free", messages=[{"role": "user", "content": "q"}])


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
    assert await judge.score(JudgeQuestion("rate", "state"), ["lo", "hi"]) == 1.99
    assert sent[0]["questions"]["q"]["type"] == "choice"
    assert sent[1]["questions"]["q"]["type"] == "score"
    assert t.spent_this_run == pytest.approx(0.002)


# ── cli ──────────────────────────────────────────────────────────────────────


def test_provider_chains_free_first_openai_last():
    assert all(is_free(m) for m in cli.provider_chain("free"))
    chain = cli.provider_chain("free+openai")
    assert chain[-1].startswith("openai/") and is_free(chain[0])
    assert cli.provider_chain("openai") == [cli.OPENAI_MODEL]


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(*a: Any, **k: Any) -> Any:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def test_dry_run_makes_no_calls_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _block_network(monkeypatch)
    ledger = tmp_path / "l.json"
    assert cli.main(["--ledger", str(ledger), "--out-dir", str(tmp_path / "runs")]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "dry-run" and plan["max_agent_calls"] > 0
    assert not ledger.exists() and not (tmp_path / "runs").exists()


def test_offline_pipeline_runs_real_agent_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _block_network(monkeypatch)
    code = cli.main(
        ["--mode", "offline", "--ledger", str(tmp_path / "l.json"), "--out-dir", str(tmp_path)]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    quant = out["suites"]["quant"]
    assert quant["solve"]["accuracy"] == 1.0  # fake model returns the golden answer
    assert quant["tutor"]["leak_rate_rule"] == 0.0
    assert out["cost_latency"]["total_usd"] == 0.0
    assert "rows" not in quant  # printed summary is aggregate-only
    assert not (tmp_path / "l.json").exists()  # free/offline never touches the ledger


def test_live_requires_keys_by_name(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--mode", "live", "--provider", "free"])
    assert "OPENROUTER_API_KEY" in str(exc.value)


async def test_transport_retries_rate_limit_before_falling_back(tmp_path: Path):
    class RateLimitError(Exception):
        pass

    calls: list[str] = []

    async def real(**kw: Any) -> Any:
        calls.append(kw["model"])
        if len(calls) < 3:
            raise RateLimitError("429")
        return _resp("ok")

    async def no_sleep(_: float) -> None:
        return None

    t = _transport(tmp_path, real, ["openai/gpt-4o-mini"])
    t.sleep = no_sleep
    msgs = [{"role": "user", "content": "q"}]
    out = await t.acompletion(model="openrouter/x:free", messages=msgs)
    assert out.choices[0].message.content == "ok"
    assert calls == ["openrouter/x:free"] * 3
