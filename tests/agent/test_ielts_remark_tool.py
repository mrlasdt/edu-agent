"""
The IELTS coach's `mark_revision` tool: an essay-sized message on a coached attempt may be
re-marked once per turn through the gateway; the learner reads the gateway's numbers as a card,
a refusal is shown as the gateway worded it, and a short message streams as before. The model
and the gateway are faked; nothing leaves the machine.
"""
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from scripts.seed_ielts_attempt import ESSAY, PROMPT, build_report
from services.agent.src.agent.agents.ielts_writing import IeltsWritingAgent
from services.agent.src.agent.tools import ielts_tools
from services.agent.src.agent.tools.ielts_tools import RevisionTools, card, offers_remark
from services.gateway.src.gateway.ielts.digest import build_digest
from shared.src.shared.events import ToolTrace
from shared.src.shared.models import Session
from tests.agent.agent_events import reply_of

CONFIG = "config/model_config.free.yaml"
DISCLAIMER = "This is a mock score, not an official IELTS result."
LABEL = "Practice re-mark (untimed, not saved as a mock attempt)"
GRADED = {
    "status": "graded",
    "label": LABEL,
    "ungraded_reason": None,
    "word_count": 260,
    "overall": 6.5,
    "range": [6.0, 7.0],
    "confidence": "medium",
    "overall_delta": 0.5,
    "criteria": [
        {"id": "task_response", "name": "Task Response", "band": 6.5, "delta": 0.5},
        {"id": "coherence_cohesion", "name": "Coherence and Cohesion", "band": 6.0, "delta": 0.0},
    ],
    "disclaimer": DISCLAIMER,
}
LONG_MESSAGE = "Please re-mark this: " + "word " * 120


def session(attempt_id: str = "a1") -> Session:
    return Session(
        candidate_id="c1", test_type="ielts", section="writing",
        report_digest=build_digest(build_report(), ESSAY, PROMPT), attempt_id=attempt_id,
    )


def gateway(status: int = 200, body: Any = None) -> tuple[httpx.AsyncClient, list[dict[str, Any]]]:
    posts: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ielts/remark"
        posts.append(json.loads(request.content))
        return httpx.Response(status, json=GRADED if body is None else body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handle)), posts


def reply(text: str | None, calls: list[Any] | None = None) -> Any:
    message = SimpleNamespace(content=text, tool_calls=calls)
    usage = SimpleNamespace(prompt_tokens=300, completion_tokens=20)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def tool_call(call_id: str = "c1", essay: str = "my revised essay") -> Any:
    function = SimpleNamespace(name="mark_revision", arguments=json.dumps({"essay": essay}))
    return reply(None, [SimpleNamespace(id=call_id, function=function)])


class Model:
    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0)


async def run(message: str, model: Model, client: httpx.AsyncClient, sess: Session | None = None):
    agent = IeltsWritingAgent(model_config_path=CONFIG)
    with (
        patch("litellm.acompletion", new=model),
        patch.object(ielts_tools.pool, "current", return_value=client),
    ):
        return [i async for i in agent.run(sess or session(), message, "t-remark")]


# ── when the tool is offered ─────────────────────────────────────────────────


def test_the_tool_is_offered_only_for_an_essay_sized_message_on_a_coached_attempt():
    assert offers_remark(LONG_MESSAGE, "a1")
    assert not offers_remark("why is my Coherence 6.0?", "a1")
    assert not offers_remark(LONG_MESSAGE, "")


@pytest.mark.asyncio
async def test_a_short_message_streams_without_the_tool_or_the_gateway():
    client, posts = gateway()
    model = AsyncMock(return_value=_stream(["Linking ", "first."]))
    with (
        patch("litellm.acompletion", new=model),
        patch.object(ielts_tools.pool, "current", return_value=client),
    ):
        agent = IeltsWritingAgent(model_config_path=CONFIG)
        items = [i async for i in agent.run(session(), "what do I fix first?", "t1")]
    assert reply_of(items) == "Linking first."
    assert "tools" not in model.await_args.kwargs and posts == []


def _stream(tokens: list[str]):
    async def gen():
        for token in tokens:
            chunk = SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=token))]
            )
            yield chunk

    return gen()


# ── a re-mark ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_remark_shows_the_gateways_card_then_the_models_coaching():
    client, posts = gateway()
    model = Model(tool_call(essay="my revised essay"), reply("Nice gain in Task Response."))
    items = await run(LONG_MESSAGE, model, client)
    text = reply_of(items)
    assert posts == [{"attempt_id": "a1", "essay": "my revised essay"}]
    assert text.startswith(LABEL)
    assert "Estimated band 6.5 (range 6.0 to 7.0), +0.5 since" in text
    assert "- Task Response: 6.5 (+0.5)" in text
    assert "- Coherence and Cohesion: 6.0 (no change)" in text
    assert DISCLAIMER in text
    assert text.endswith("Nice gain in Task Response.")
    assert [c.name for c in next(i for i in items if isinstance(i, ToolTrace)).calls] == [
        "mark_revision"
    ]


@pytest.mark.asyncio
async def test_the_model_is_told_the_numbers_but_not_the_whole_payload():
    client, _ = gateway()
    model = Model(tool_call(), reply("ok"))
    await run(LONG_MESSAGE, model, client)
    tool_message = model.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    result = json.loads(tool_message["content"])
    assert result["overall"] == 6.5 and result["overall_delta"] == 0.5
    assert "disclaimer" not in result


@pytest.mark.asyncio
async def test_a_second_call_in_one_turn_is_refused_without_a_second_gateway_request():
    client, posts = gateway()
    tools = RevisionTools("a1", gateway_url="http://gw.test")
    with patch.object(ielts_tools.pool, "current", return_value=client):
        first = await tools.call("mark_revision", {"essay": "e"})
        second = await tools.call("mark_revision", {"essay": "e"})
    assert "overall" in first and "error" in second
    assert len(posts) == 1


# ── refusals ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "detail"),
    [
        (402, "The AI spend cap is used up, so no more marks can run."),
        (429, "Too many requests. Try again in a minute."),
        (502, "The marker is unavailable right now. Nothing was spent."),
    ],
)
async def test_a_refusal_is_shown_as_the_gateway_worded_it_and_no_band_is_made_up(
    status, detail
):
    client, _ = gateway(status, {"detail": detail})
    model = Model(tool_call(), reply("I could not re-mark that just now."))
    text = reply_of(await run(LONG_MESSAGE, model, client))
    assert text.startswith(detail)
    assert "Estimated band" not in text and DISCLAIMER not in text
    told = json.loads(model.requests[1]["messages"][-1]["content"])
    assert told["error"] == detail
    assert "Do not call it a technical error" in told["note"]


@pytest.mark.asyncio
async def test_an_unreachable_gateway_reads_as_an_outage_not_a_cap():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    model = Model(tool_call(), reply("Sorry."))
    text = reply_of(await run(LONG_MESSAGE, model, client))
    assert "marker is unreachable" in text and "cap" not in text.lower()


@pytest.mark.asyncio
async def test_an_ungraded_essay_says_why_and_shows_no_band():
    ungraded = {**GRADED, "status": "ungraded", "ungraded_reason": "it is not in English",
                "overall": None, "range": None, "criteria": [], "overall_delta": None}
    client, _ = gateway(200, ungraded)
    model = Model(tool_call(), reply("Please paste an English essay."))
    text = reply_of(await run(LONG_MESSAGE, model, client))
    assert "could not be marked: it is not in English" in text
    assert "Estimated band" not in text


def test_the_card_names_a_missing_earlier_band():
    remark = {**GRADED, "overall_delta": None, "criteria": []}
    assert "no earlier band since" in card(remark)
