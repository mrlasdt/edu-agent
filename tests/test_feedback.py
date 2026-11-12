"""
Closing the feedback loop: the trace id reaches the client, a thumbs comes back against it, and a
thumbs-down becomes a draft eval item.

Behaviors under test:
  1. /chat returns the turn's trace id in `X-Trace-Id`, and it matches the turn log
  2. POST /feedback appends one line (trace id, vote, comment, time) and never any message text
  3. an unknown trace id, a bad vote, an oversize field or an extra field is rejected; so is a
     burst past the rate limit; with feedback switched off the endpoint says so
  4. message-text capture is off by default; opted in, it keeps the question and the reply (no
     trailer) keyed by trace id in a file of its own
  5. a user-feedback score goes to Langfuse only when Langfuse is configured
  6. `turn_report --feedback` shows the thumbs-down rate per agent and prompt version, and lists
     each thumbs-down
  7. `feedback_to_item` writes a draft item: complete when text was captured, a stub otherwise
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from evals.feedback_to_item import main as to_item_main
from evals.turn_report import main as report_main
from services.gateway.src.gateway import feedback_api
from services.gateway.src.gateway.guardrails.rate_limit import RateLimiter
from services.gateway.src.gateway.guardrails.verdict import Verdict
from services.gateway.src.gateway.limits import feedback_limiter
from services.gateway.src.gateway.main import app
from shared.src.shared import observability
from shared.src.shared.config import FeedbackSettings, get_feedback_settings
from shared.src.shared.feedback import default_capture_log, default_feedback_log
from shared.src.shared.turn_meter import TurnLog
from tests.gateway_fakes import agent_replies, override, rate_limiter_of

client = TestClient(app)

QUESTION = "Solve zebra-quokka-7731 = 0 for x"
REPLY = "Factor the zebra-quokka expression first."


@pytest.fixture(autouse=True)
def offline_gateway():
    """Every guardrail model call and the agent stream are faked: a turn costs nothing."""

    async def fake_stream(payload: dict[str, Any], meta: dict[str, Any] | None = None):
        yield REPLY.encode()
        yield b'\x1e{"math_verified":true}'

    with (
        patch(
            "services.gateway.src.gateway.chat.gates.check_content_safety",
            new_callable=AsyncMock,
            return_value=Verdict.allow(),
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
        agent_replies(fake_stream),
        rate_limiter_of(RateLimiter()),
        override(feedback_limiter, RateLimiter()),
        patch.object(feedback_api, "post_user_feedback") as langfuse,
    ):
        yield langfuse


def chat(question: str = QUESTION) -> Any:
    return client.post(
        "/chat",
        json={
            "candidate_id": "c1",
            "test_type": "gre",
            "section": "quant",
            "message": question,
            "mode": "solve",
        },
    )


def lines(log: TurnLog) -> list[dict[str, Any]]:
    return log.read()


def test_chat_returns_the_trace_id_that_the_turn_log_holds(turn_log):
    resp = chat()
    trace_id = resp.headers["x-trace-id"]
    assert trace_id
    assert {r["trace_id"] for r in lines(turn_log)} == {trace_id}


def test_feedback_is_appended_with_no_message_text(turn_log):
    trace_id = chat().headers["x-trace-id"]
    resp = client.post("/feedback", json={"trace_id": trace_id, "vote": "down", "comment": "wrong"})
    assert resp.status_code == 200

    log = default_feedback_log()
    assert log is not None
    [record] = lines(log)
    assert set(record) == {"ts", "trace_id", "vote", "comment"}
    assert (record["trace_id"], record["vote"], record["comment"]) == (trace_id, "down", "wrong")
    raw = log.path.read_text(encoding="utf-8")
    assert "zebra-quokka" not in raw


def test_comment_is_optional_and_left_out_when_absent(turn_log):
    trace_id = chat().headers["x-trace-id"]
    assert client.post("/feedback", json={"trace_id": trace_id, "vote": "up"}).status_code == 200
    log = default_feedback_log()
    assert log is not None
    assert "comment" not in lines(log)[0]


def test_unknown_trace_id_is_rejected_and_nothing_is_written(turn_log):
    chat()
    resp = client.post("/feedback", json={"trace_id": "not-a-real-turn", "vote": "down"})
    assert resp.status_code == 404
    log = default_feedback_log()
    assert log is not None
    assert lines(log) == []


@pytest.mark.parametrize(
    "body",
    [
        {"trace_id": "t", "vote": "sideways"},
        {"trace_id": "", "vote": "up"},
        {"trace_id": "t" * 65, "vote": "up"},
        {"trace_id": "t", "vote": "up", "comment": "x" * 1001},
        {"trace_id": "t", "vote": "up", "message": "the learner's question"},
        {"vote": "up"},
    ],
)
def test_malformed_or_oversize_feedback_is_a_422(body):
    assert client.post("/feedback", json=body).status_code == 422


def test_feedback_is_rate_limited(turn_log):
    trace_id = chat().headers["x-trace-id"]
    body = {"trace_id": trace_id, "vote": "up"}
    with override(feedback_limiter, RateLimiter(max_per_minute=2)):
        codes = [client.post("/feedback", json=body).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_feedback_blocking_log_scan_runs_off_the_event_loop(turn_log):
    trace_id = chat().headers["x-trace-id"]
    real = TurnLog.last_with_trace
    on_loop: list[bool] = []

    def spy(self: TurnLog, tid: str) -> dict[str, Any] | None:
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return real(self, tid)

    with patch.object(TurnLog, "last_with_trace", spy):
        resp = client.post("/feedback", json={"trace_id": trace_id, "vote": "up"})
    assert resp.status_code == 200
    assert on_loop == [False]


def test_feedback_says_so_when_it_is_switched_off(turn_log, monkeypatch):
    trace_id = chat().headers["x-trace-id"]
    monkeypatch.setenv("FEEDBACK_LOG_PATH", "")
    get_feedback_settings.cache_clear()
    resp = client.post("/feedback", json={"trace_id": trace_id, "vote": "up"})
    assert resp.status_code == 503


# ── opt-in text capture ───────────────────────────────────────────────────────


def test_text_capture_is_off_by_default(monkeypatch):
    monkeypatch.delenv("TURN_CAPTURE_TEXT", raising=False)
    assert FeedbackSettings(_env_file=None).turn_capture_text is False  # type: ignore[call-arg]


def test_nothing_is_captured_unless_opted_in(turn_log):
    chat()
    assert default_capture_log() is None
    assert not Path(get_feedback_settings().turn_capture_path).exists()


def test_opted_in_capture_keeps_question_and_reply_without_the_trailer(turn_log, monkeypatch):
    monkeypatch.setenv("TURN_CAPTURE_TEXT", "1")
    get_feedback_settings.cache_clear()
    trace_id = chat().headers["x-trace-id"]

    capture = default_capture_log()
    assert capture is not None
    record = capture.last_with_trace(trace_id)
    assert record is not None
    assert record["question"] == QUESTION
    assert record["reply"] == REPLY
    # the vote file still holds no text
    client.post("/feedback", json={"trace_id": trace_id, "vote": "down"})
    log = default_feedback_log()
    assert log is not None
    assert "zebra-quokka" not in log.path.read_text(encoding="utf-8")
    assert capture.path != log.path


# ── Langfuse ──────────────────────────────────────────────────────────────────


def test_the_vote_is_also_sent_to_langfuse_when_configured(turn_log, offline_gateway):
    trace_id = chat().headers["x-trace-id"]
    client.post("/feedback", json={"trace_id": trace_id, "vote": "down", "comment": "bad"})
    offline_gateway.assert_called_once_with(trace_id, "down", "bad")


def test_langfuse_score_is_a_no_op_without_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_feedback_settings.cache_clear()
    with patch.object(observability, "_langfuse_client") as make:
        assert observability.post_user_feedback("t", "down") is False
    make.assert_not_called()


def test_langfuse_score_is_posted_on_the_trace(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-real")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-real")
    get_feedback_settings.cache_clear()
    fake = MagicMock()
    with patch.object(observability, "_langfuse_client", return_value=fake):
        assert observability.post_user_feedback("trace-1", "up", "nice") is True
    fake.score.assert_called_once_with(
        id="trace-1-user-feedback",
        trace_id="trace-1",
        name="user-feedback",
        value=1,
        data_type="NUMERIC",
        comment="nice",
    )


def test_a_changed_vote_reuses_the_langfuse_score_id(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-real")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-real")
    get_feedback_settings.cache_clear()
    fake = MagicMock()
    with patch.object(observability, "_langfuse_client", return_value=fake):
        observability.post_user_feedback("trace-1", "up")
        observability.post_user_feedback("trace-1", "down")
    ids = {call.kwargs["id"] for call in fake.score.call_args_list}
    assert len(fake.score.call_args_list) == 2 and len(ids) == 1


def test_a_langfuse_failure_never_raises(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-real")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-real")
    get_feedback_settings.cache_clear()
    with patch.object(observability, "_langfuse_client", side_effect=RuntimeError("down")):
        assert observability.post_user_feedback("t", "down") is False


# ── report and draft item ─────────────────────────────────────────────────────


def turn_with_agent_record(turn_log) -> str:
    """One real /chat turn plus the agent-side record that names the agent and prompt version."""
    trace_id = chat().headers["x-trace-id"]
    turn_log.append(
        {
            "v": 1, "ts": 1.0, "trace_id": trace_id, "source": "agent", "agent": "quant_agent",
            "mode": "solve", "prompt_version": "v2", "llm_calls": 1, "input_tokens": 10,
            "output_tokens": 5, "cost_usd": 0.0, "usage": "provider",
        }
    )
    return trace_id


def rated_turn(turn_log, vote: str = "down", comment: str | None = "wrong sign") -> str:
    trace_id = turn_with_agent_record(turn_log)
    body: dict[str, Any] = {"trace_id": trace_id, "vote": vote}
    if comment:
        body["comment"] = comment
    assert client.post("/feedback", json=body).status_code == 200
    return trace_id


def test_report_shows_thumbs_down_rate_per_agent_and_prompt_version(turn_log, capsys):
    bad = rated_turn(turn_log, "down")
    rated_turn(turn_log, "up", comment=None)
    turn_with_agent_record(turn_log)  # a turn nobody rated

    assert report_main(["--log", str(turn_log.path), "--feedback"]) == 0
    out = capsys.readouterr().out
    assert "quant_agent v2" in out
    assert "50%" in out
    assert bad in out
    assert "wrong sign" in out
    assert out.count("thumbs-down (") == 1  # the heading
    assert sum(1 for line in out.splitlines() if line.strip().startswith(bad)) == 1

    assert report_main(["--log", str(turn_log.path), "--feedback", "--json"]) == 0
    group = json.loads(capsys.readouterr().out)["feedback"]["groups"]["quant_agent v2"]
    assert (group["turns"], group["rated"], group["up"], group["down"]) == (3, 2, 1, 1)
    assert group["down_rate"] == 0.5


def test_the_latest_vote_on_a_turn_wins(turn_log, capsys):
    trace_id = rated_turn(turn_log, "down")
    client.post("/feedback", json={"trace_id": trace_id, "vote": "up"})
    assert report_main(["--log", str(turn_log.path), "--feedback", "--json"]) == 0
    fb = json.loads(capsys.readouterr().out)["feedback"]
    assert fb["thumbs_down"] == 0
    assert fb["groups"]["quant_agent v2"]["up"] == 1


def test_report_without_feedback_does_not_mention_it(turn_log, capsys):
    rated_turn(turn_log)
    assert report_main(["--log", str(turn_log.path)]) == 0
    assert "feedback" not in capsys.readouterr().out


def test_draft_item_is_a_stub_without_captured_text(turn_log, tmp_path, capsys):
    trace_id = rated_turn(turn_log)
    out = tmp_path / "drafts.jsonl"
    assert to_item_main([trace_id, "--out", str(out), "--log", str(turn_log.path)]) == 0
    [item] = [json.loads(line) for line in out.read_text().splitlines()]
    assert item["id"] == f"fb-{trace_id[:8]}"
    assert item["draft"] is True
    assert item["kind"] == "solve"
    assert item["question"] == ""
    assert item["reply_under_review"] == ""
    assert "question" in item["needs"] and "ground_truth" in item["needs"]
    assert trace_id in item["failure_source"] and "wrong sign" in item["failure_source"]
    assert item["trace"]["prompt_version"] == "v2"
    assert "re-enter the question" in capsys.readouterr().out


def test_draft_item_is_complete_when_text_was_captured(turn_log, tmp_path, monkeypatch):
    monkeypatch.setenv("TURN_CAPTURE_TEXT", "1")
    get_feedback_settings.cache_clear()
    trace_id = rated_turn(turn_log)
    out = tmp_path / "drafts.jsonl"
    assert to_item_main([trace_id, "--out", str(out), "--log", str(turn_log.path)]) == 0
    [item] = [json.loads(line) for line in out.read_text().splitlines()]
    assert item["question"] == QUESTION
    assert item["reply_under_review"] == REPLY
    assert "question" not in item["needs"]


def test_draft_item_ignores_an_old_capture_file_once_capture_is_off(
    turn_log, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("TURN_CAPTURE_TEXT", "1")
    get_feedback_settings.cache_clear()
    trace_id = rated_turn(turn_log)
    capture = default_capture_log()
    assert capture is not None and capture.last_with_trace(trace_id) is not None

    monkeypatch.setenv("TURN_CAPTURE_TEXT", "0")
    get_feedback_settings.cache_clear()
    out = tmp_path / "drafts.jsonl"
    assert to_item_main([trace_id, "--out", str(out), "--log", str(turn_log.path)]) == 0
    [item] = [json.loads(line) for line in out.read_text().splitlines()]
    assert item["question"] == "" and item["reply_under_review"] == ""
    assert "question" in item["needs"]
    assert item["trace"]["trace_id"] == trace_id
    assert "re-enter the question" in capsys.readouterr().out


def test_drafting_twice_replaces_the_draft(turn_log, tmp_path):
    trace_id = rated_turn(turn_log)
    out = tmp_path / "drafts.jsonl"
    args = [trace_id, "--out", str(out), "--log", str(turn_log.path)]
    assert to_item_main(args) == 0
    assert to_item_main(args) == 0
    assert len(out.read_text().splitlines()) == 1


def test_drafting_an_unknown_trace_id_fails(turn_log, tmp_path, capsys):
    out = tmp_path / "drafts.jsonl"
    assert to_item_main(["nope", "--out", str(out), "--log", str(turn_log.path)]) == 1
    assert not out.exists()
    assert "unknown trace id" in capsys.readouterr().err
