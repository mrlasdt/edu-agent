"""
Per-turn measurement of the live chat path (shared.turn_meter and evals.reports.turn_report).

The proof that matters: a fake model with KNOWN usage goes through the real agents and the real
/chat endpoint, and the recorded counts, tokens and cost equal what the fake reported.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from evals.reports.turn_report import main as report_main
from evals.reports.turn_report import merge_turns, percentile, summarise
from services.agent.src.agent.agents.aw import AWAgent
from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.capabilities.retrieve_corpus import CorpusChunk
from services.agent.src.agent.capabilities.verify_math import VerifyMathResult
from services.gateway.src.gateway.guardrails.verdict import Verdict
from services.gateway.src.gateway.main import app
from shared.src.shared.budget import estimate_cost, estimate_tokens
from shared.src.shared.feedback import latest_votes
from shared.src.shared.models import Mode, Session
from shared.src.shared.turn_meter import TurnLog, TurnMeter
from shared.src.shared.turn_outcome import Outcome
from tests.agent.agent_events import reply_of
from tests.gateway.gateway_fakes import agent_replies

MODEL = "openai/gpt-4o-mini"


def content_chunk(text: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def usage_chunk(prompt: int, completion: int) -> Any:
    """What a provider sends last when asked for usage: no choices, only the totals."""
    return SimpleNamespace(
        choices=[], usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
    )


def fake_stream(tokens: list[str], usage: tuple[int, int] | None = (120, 30)) -> Any:
    async def _stream() -> Any:
        for token in tokens:
            yield content_chunk(token)
        if usage is not None:
            yield usage_chunk(*usage)

    return _stream()


class FakeClock:
    """Each read advances 10 ms, so every duration is a known multiple of 10."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.010
        return self.now


def session(mode: Mode = Mode.tutor, section: str = "quant") -> Session:
    return Session(candidate_id="c1", test_type="gre", section=section, mode=mode)


# ── TurnMeter around one call ────────────────────────────────────────────────


async def test_stream_records_provider_usage_exactly_and_asks_for_it():
    meter = TurnMeter("t1", "agent", "quant_agent", "tutor", "v3")
    with patch("litellm.acompletion", new_callable=AsyncMock) as llm:
        llm.return_value = fake_stream(["Think", " first."], usage=(120, 30))
        seen = [c async for c in meter.stream(model=MODEL, messages=[], temperature=0.0)]

    kwargs = llm.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert kwargs["temperature"] == 0.0  # caller arguments pass through
    assert [c.choices[0].delta.content for c in seen] == ["Think", " first."]  # no usage chunk

    record = meter.finish()
    assert record["llm_calls"] == 1
    assert (record["input_tokens"], record["output_tokens"]) == (120, 30)
    assert record["usage"] == "provider"
    assert record["cost_usd"] == pytest.approx(estimate_cost(MODEL, 120, 30))


async def test_stream_without_usage_falls_back_to_the_estimate_and_says_so():
    messages = [{"role": "user", "content": "x" * 400}]
    meter = TurnMeter("t1", "agent", "quant_agent", "tutor")
    with patch("litellm.acompletion", new_callable=AsyncMock) as llm:
        llm.return_value = fake_stream(["a" * 40], usage=None)
        async for _ in meter.stream(model=MODEL, messages=messages):
            pass

    record = meter.finish()
    assert record["input_tokens"] == estimate_tokens("x" * 400)
    assert record["output_tokens"] == estimate_tokens("a" * 40)
    assert record["usage"] == "estimate"
    assert record["calls"][0]["estimated"] is True


async def test_mocks_without_integer_usage_are_treated_as_no_usage():
    chunk = MagicMock()  # a MagicMock invents .usage.prompt_tokens; that is not a measurement
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = "hi"

    async def stream() -> Any:
        yield chunk

    meter = TurnMeter("t1", "agent", "a", "tutor")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=stream()):
        async for _ in meter.stream(model=MODEL, messages=[]):
            pass
    assert meter.finish()["usage"] == "estimate"


async def test_ttft_and_total_time_are_measured_from_the_clock():
    meter = TurnMeter("t1", "agent", "a", "tutor", clock=FakeClock())  # started: 10 ms
    with patch("litellm.acompletion", new_callable=AsyncMock) as llm:
        llm.return_value = fake_stream(["a", "b"])
        async for _ in meter.stream(model=MODEL, messages=[]):  # start 20, first token 30
            pass
    record = meter.finish()  # call ends 40, finish reads 50
    call = record["calls"][0]
    assert call["ttft_ms"] == pytest.approx(10.0)
    assert call["ms"] == pytest.approx(20.0)
    assert record["ttft_ms"] == pytest.approx(10.0)
    assert record["total_ms"] == pytest.approx(40.0)


async def test_phase_time_accumulates_by_name():
    meter = TurnMeter("t1", "agent", "a", "tutor", clock=FakeClock())
    with meter.phase("retrieval"):
        pass
    with meter.phase("retrieval"):
        pass
    assert meter.finish()["phases_ms"] == {"retrieval": pytest.approx(20.0)}


async def test_complete_records_usage_of_a_non_streamed_call():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="PASS"))],
        usage=SimpleNamespace(prompt_tokens=300, completion_tokens=2),
    )
    meter = TurnMeter("t1", "gateway", "gre/quant", "tutor")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=response) as llm:
        got = await meter.complete(model=MODEL, messages=[], temperature=0)
    assert got is response
    assert llm.call_args.kwargs["stream"] is False
    record = meter.finish()
    assert (record["llm_calls"], record["input_tokens"], record["output_tokens"]) == (1, 300, 2)
    assert record["ttft_ms"] is None  # nothing streamed


async def test_a_failed_call_is_still_counted_and_the_error_is_raised():
    meter = TurnMeter("t1", "agent", "a", "tutor")
    with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("down")):
        with pytest.raises(RuntimeError):
            async for _ in meter.stream(model=MODEL, messages=[]):
                pass
    record = meter.finish()
    assert record["llm_calls"] == 1
    assert record["calls"][0]["error"] == "RuntimeError"


async def test_a_call_that_failed_before_any_response_costs_nothing():
    messages = [{"role": "user", "content": "x" * 4000}]
    for failing in ("stream", "complete"):
        meter = TurnMeter("t1", "agent", "a", "tutor")
        with patch(
            "litellm.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("401")
        ):
            with pytest.raises(RuntimeError):
                if failing == "stream":
                    async for _ in meter.stream(model=MODEL, messages=messages):
                        pass
                else:
                    await meter.complete(model=MODEL, messages=messages)
        record = meter.finish()
        assert record["llm_calls"] == 1
        assert (record["input_tokens"], record["output_tokens"]) == (0, 0)
        assert record["cost_usd"] == 0
        assert record["calls"][0]["error"] == "RuntimeError"
        assert record["usage"] == "provider"


async def test_mixed_usage_is_labelled_mixed():
    meter = TurnMeter("t1", "agent", "a", "solve")
    with patch("litellm.acompletion", new_callable=AsyncMock) as llm:
        llm.side_effect = [fake_stream(["a"], usage=(10, 1)), fake_stream(["b"], usage=None)]
        for _ in range(2):
            async for _chunk in meter.stream(model=MODEL, messages=[]):
                pass
    record = meter.finish()
    assert record["llm_calls"] == 2
    assert record["usage"] == "mixed"


# ── The log ──────────────────────────────────────────────────────────────────


def test_finish_appends_one_line_and_is_idempotent(turn_log):
    meter = TurnMeter("t1", "agent", "a", "tutor")
    first = meter.finish()
    assert meter.finish() is first
    assert [r["trace_id"] for r in turn_log.read()] == ["t1"]


def test_a_turn_without_a_trace_id_is_measured_but_not_written(turn_log):
    TurnMeter("", "agent", "a", "tutor").finish()
    assert turn_log.read() == []


def test_an_unwritable_log_never_breaks_the_turn(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    meter = TurnMeter("t1", "agent", "a", "tutor", log=TurnLog(blocked / "turns.jsonl"))
    assert meter.finish()["trace_id"] == "t1"


def test_the_log_skips_a_torn_line(turn_log):
    turn_log.append({"trace_id": "t1"})
    with turn_log.path.open("a") as fh:
        fh.write('{"trace_id": "t2", "llm_cal')
    assert [r["trace_id"] for r in turn_log.read()] == ["t1"]


def test_records_written_before_the_version_key_was_dropped_still_load(turn_log):
    """Old lines carry `"v": 1`; nothing reads it, so they load and report like any other."""
    legacy = {"v": 1, "ts": 1.0, "trace_id": "t1", "source": "agent", "agent": "quant_agent",
              "mode": "tutor", "llm_calls": 1, "input_tokens": 10, "output_tokens": 5,
              "cost_usd": 0.0, "usage": "provider"}
    turn_log.path.parent.mkdir(parents=True, exist_ok=True)
    turn_log.path.write_text(json.dumps(legacy, separators=(",", ":")) + "\n", encoding="utf-8")
    turn_log.append({"ts": 2.0, "trace_id": "t2", "vote": "down"})
    assert [r["trace_id"] for r in turn_log.read()] == ["t1", "t2"]
    assert turn_log.last_with_trace("t1") == legacy
    assert latest_votes(turn_log.read()) == {"t2": {"ts": 2.0, "trace_id": "t2", "vote": "down"}}
    assert [t["trace_id"] for t in merge_turns(turn_log.read())] == ["t1", "t2"]


# ── The real agents with a fake model of known usage ─────────────────────────


def known_usage_stream(prompt: int, completion: int) -> Any:
    return lambda **_: fake_stream(["Ask what the roots add up to."], usage=(prompt, completion))


async def test_quant_tutor_turn_records_what_the_fake_model_reported(turn_log):
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")
    chunks = [
        CorpusChunk(text="n", tier="global", source_uri="u", page_or_section="s", score=0.5)
        for _ in range(3)
    ]
    with (
        patch("litellm.acompletion", side_effect=known_usage_stream(812, 41)),
        patch.object(agent, "_retrieve", new_callable=AsyncMock, return_value=chunks),
    ):
        text = [i async for i in agent.run(session(), "Solve x^2-5x+6=0", trace_id="trace-q")]

    assert "Ask what the roots add up to." in reply_of(text)
    (record,) = turn_log.read()
    assert record["trace_id"] == "trace-q"
    assert (record["source"], record["agent"], record["mode"]) == ("agent", "quant_agent", "tutor")
    assert record["prompt_version"] == agent._model_config.prompt_version("quant_agent", "tutor")
    assert record["llm_calls"] == 1
    assert (record["input_tokens"], record["output_tokens"]) == (812, 41)
    assert record["usage"] == "provider"
    assert record["cost_usd"] == pytest.approx(estimate_cost(MODEL, 812, 41))
    assert record["notes_retrieved"] == 3
    assert "retrieval" in record["phases_ms"]
    assert record["ttft_ms"] is not None


async def test_quant_solve_with_retry_and_escalation_counts_every_call(turn_log):
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")
    with (
        patch("litellm.acompletion", side_effect=known_usage_stream(500, 100)),
        patch.object(
            agent,
            "verify_well_formed",
            new_callable=AsyncMock,
            return_value=VerifyMathResult(verified=False, computed="[2,3]"),
        ),
    ):
        async for _ in agent.run(session(Mode.solve), "Solve x^2-5x+6=0", trace_id="trace-s"):
            pass

    (record,) = turn_log.read()
    assert record["llm_calls"] == 3  # attempt, retry, escalation
    assert (record["input_tokens"], record["output_tokens"]) == (3 * 500, 3 * 100)
    assert "verify" in record["phases_ms"]


async def test_quant_solve_record_says_when_the_answer_was_not_independently_checked(turn_log):
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")
    with (
        patch("litellm.acompletion", side_effect=known_usage_stream(50, 10)),
        patch.object(
            agent,
            "verify_well_formed",
            new_callable=AsyncMock,
            return_value=VerifyMathResult(verified=True, computed="[2,3]"),
        ),
    ):
        async for _ in agent.run(session(Mode.solve), "Solve x^2-5x+6=0", trace_id="trace-i"):
            pass

    (record,) = turn_log.read()
    # the stubbed model wrote no <expr> tag, so there was nothing to compute
    assert record["verified_independent"] == "not_checkable"
    assert record["independent_reason"]


async def test_quant_tutor_record_has_no_verification_field(turn_log):
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")
    with patch("litellm.acompletion", side_effect=known_usage_stream(50, 10)):
        async for _ in agent.run(session(Mode.tutor), "Help", trace_id="trace-j"):
            pass

    (record,) = turn_log.read()
    assert "verified_independent" not in record


async def test_a_direct_call_without_trace_id_writes_nothing(turn_log):
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")
    with patch("litellm.acompletion", side_effect=known_usage_stream(10, 5)):
        async for _ in agent.run(session(), "hi"):
            pass
    assert turn_log.read() == []


async def test_aw_solve_counts_the_essay_and_the_style_critic_call(turn_log):
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")
    critic_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="PASS"))],
        usage=SimpleNamespace(prompt_tokens=700, completion_tokens=3),
    )

    async def fake(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return fake_stream(["An essay."], usage=(400, 600))
        return critic_response

    with (
        patch("litellm.acompletion", side_effect=fake),
        patch.object(agent, "_retrieve", new_callable=AsyncMock, return_value=[]),
        patch.object(agent, "_check_citations", return_value=(True, [])),
    ):
        async for _ in agent.run(session(Mode.solve, "aw"), "Write an essay", trace_id="trace-aw"):
            pass

    (record,) = turn_log.read()
    assert record["llm_calls"] == 2
    assert (record["input_tokens"], record["output_tokens"]) == (400 + 700, 600 + 3)


# ── The gateway half, joined by the report ───────────────────────────────────

client = TestClient(app)
_BODY = {"candidate_id": "c1", "test_type": "gre", "section": "quant", "message": "Solve it"}


def post_chat(tokens: list[str], *, safe: bool = True, body: dict[str, Any] | None = None) -> Any:
    async def agent_stream(payload: dict[str, Any], meta: dict[str, Any] | None = None) -> Any:
        for token in tokens:
            yield token.encode()

    with (
        patch(
            "services.gateway.src.gateway.chat.gates.check_content_safety",
            new_callable=AsyncMock,
            return_value=Verdict.allow()
            if safe
            else Verdict.stop(Outcome.content_blocked, "I can't help with that request (x)."),
        ),
        patch(
            "services.gateway.src.gateway.chat.gates.check_completeness",
            new_callable=AsyncMock,
            return_value=Verdict.allow(),
        ),
        patch(
            "services.gateway.src.gateway.guardrails.output._call_leak_checker",
            new_callable=AsyncMock,
            return_value=False,
        ),
        agent_replies(agent_stream),
    ):
        return client.post("/chat", json={**_BODY, **(body or {})})


def test_chat_records_the_guardrail_outcome_and_first_byte(turn_log):
    assert post_chat(["Isolate x first."]).status_code == 200
    (record,) = turn_log.read()
    assert (record["source"], record["agent"], record["mode"]) == ("gateway", "gre/quant", "tutor")
    assert record["outcome"] == "ok"
    assert record["flags"] == []
    assert record["first_byte_ms"] is not None
    assert record["total_ms"] >= record["first_byte_ms"]


def test_chat_records_why_a_turn_was_stopped(turn_log):
    post_chat(["never streamed"], safe=False)
    post_chat(["Isolate x.\nThe answer is 4.\nmore"])
    outcomes = [r["outcome"] for r in turn_log.read()]
    assert outcomes == ["content_blocked", "answer_leak_blocked"]


def test_chat_records_non_blocking_events_as_flags(turn_log):
    post_chat(["Mail me at a@b.com when done.\n"], body={"message": "my email is x@y.com"})
    (record,) = turn_log.read()
    assert record["outcome"] == "ok"
    assert set(record["flags"]) == {"input_pii_redacted", "output_pii_redacted"}


async def test_guardrail_model_calls_are_counted_on_the_gateway_record(turn_log):
    from services.gateway.src.gateway.orchestrator.enrich import check_completeness

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"complete": true}'))],
        usage=SimpleNamespace(prompt_tokens=250, completion_tokens=8),
    )
    meter = TurnMeter("trace-g", "gateway", "gre/quant", "tutor")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=response):
        verdict = await check_completeness("Solve it", session(), "trace-g", meter)
    assert verdict.allowed is True
    record = meter.finish()
    assert (record["llm_calls"], record["input_tokens"], record["output_tokens"]) == (1, 250, 8)


def test_the_report_joins_gateway_and_agent_records_by_trace_id(turn_log):
    gateway = TurnMeter("t1", "gateway", "gre/quant", "tutor", log=turn_log)
    gateway.annotate(outcome="ok", flags=[], first_byte_ms=900.0)
    gateway._add_call(MODEL, [], "", (200, 5), 0.0, None, None)
    gateway.finish()
    agent = TurnMeter("t1", "agent", "quant_agent", "tutor", "v3", log=turn_log)
    agent.annotate(notes_retrieved=3)
    agent._add_call(MODEL, [], "", (800, 40), 0.0, 120.0, None)
    agent.finish()

    (turn,) = merge_turns(turn_log.read())
    assert turn["agent"] == "quant_agent"
    assert turn["llm_calls"] == 2
    assert (turn["agent_calls"], turn["guardrail_calls"]) == (1, 1)
    assert (turn["input_tokens"], turn["output_tokens"]) == (1000, 45)
    assert turn["cost_usd"] == pytest.approx(
        estimate_cost(MODEL, 200, 5) + estimate_cost(MODEL, 800, 40)
    )
    assert turn["guardrail_cost_usd"] == pytest.approx(estimate_cost(MODEL, 200, 5))
    assert turn["notes_retrieved"] == 3
    assert turn["outcome"] == "ok"
    assert turn["estimated"] is False


def test_percentile_is_nearest_rank():
    assert percentile([], 0.5) is None
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.5) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95) == 10
    assert percentile([1, 2], 0.5) == 1
    assert percentile([1, 2, 3], 0.5) == 2


def test_report_cli_prints_a_summary_and_json(turn_log, capsys):
    for trace_id, usage in (("a", (100, 10)), ("b", (300, 30))):
        gateway = TurnMeter(trace_id, "gateway", "gre/quant", "tutor", log=turn_log)
        gateway.annotate(outcome="ok", flags=[], first_byte_ms=500.0)
        gateway.finish()
        agent = TurnMeter(trace_id, "agent", "quant_agent", "tutor", "v3", log=turn_log)
        agent._add_call(MODEL, [], "", usage, 0.0, 120.0, None)
        agent.finish()

    assert report_main(["--log", str(turn_log.path), "--turns"]) == 0
    out = capsys.readouterr().out
    assert "quant_agent/tutor" in out and "2 turn(s)" in out and "outcomes: ok 2" in out

    assert report_main(["--log", str(turn_log.path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    group = data["summary"]["groups"]["quant_agent/tutor"]
    assert group["turns"] == 2
    assert group["input_tokens_mean"] == 200
    assert group["cost_usd_mean"] == pytest.approx(
        (estimate_cost(MODEL, 100, 10) + estimate_cost(MODEL, 300, 30)) / 2
    )
    assert summarise(merge_turns(turn_log.read()))["turns"] == 2


def test_report_cli_on_an_empty_log_says_so(tmp_path, capsys):
    assert report_main(["--log", str(tmp_path / "none.jsonl")]) == 1
    assert "no turns recorded" in capsys.readouterr().err


def test_report_can_price_recorded_tokens_as_another_model(turn_log, capsys):
    free = "openrouter/some-model:free"  # recorded at 0 USD
    gateway = TurnMeter("a", "gateway", "gre/quant", "tutor", log=turn_log)
    gateway.annotate(outcome="ok", flags=[])
    gateway._add_call(free, [], "", (200, 10), 0.0, None, None)
    gateway.finish()
    agent = TurnMeter("a", "agent", "quant_agent", "tutor", log=turn_log)
    agent._add_call(free, [], "", (800, 90), 0.0, 100.0, None)
    agent.finish()

    assert report_main(["--log", str(turn_log.path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["turns"][0]["cost_usd"] == 0.0

    assert report_main(["--log", str(turn_log.path), "--json", "--price-as", "gpt-4o-mini"]) == 0
    turn = json.loads(capsys.readouterr().out)["turns"][0]
    assert turn["cost_usd"] == pytest.approx(estimate_cost(MODEL, 1000, 100))
    assert turn["guardrail_cost_usd"] == pytest.approx(estimate_cost(MODEL, 200, 10))


def test_the_turn_report_counts_the_tool_calls_of_a_turn(tmp_path):
    from evals.reports.turn_report import render

    calls = [{"name": "math_evaluate", "ok": True}, {"name": "math_solve", "ok": False}]
    records = [
        {"trace_id": "t1", "source": "agent", "agent": "quant", "mode": "solve", "ts": 1.0,
         "llm_calls": 3, "tool_calls": calls, "tool_errors": 1, "total_ms": 900.0},
        {"trace_id": "t1", "source": "gateway", "agent": "quant", "mode": "solve", "ts": 1.0,
         "llm_calls": 0, "total_ms": 1000.0, "outcome": "ok"},
    ]
    (turn,) = merge_turns(records)
    assert (turn["tool_calls"], turn["tool_errors"]) == (2, 1)
    assert "tools" in render(summarise([turn]), [turn], per_turn=True)
