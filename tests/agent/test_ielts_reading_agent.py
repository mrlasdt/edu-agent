"""
The IELTS Reading coach agent: one metered model call on the ielts_reading_coach role, the digest
as delimited data in the learner's turn, and a safe reply when there is no marked result. The
model is faked; nothing leaves the machine.
"""
from unittest.mock import AsyncMock, patch

import pytest

from services.agent.src.agent.agents.ielts_reading import IeltsReadingAgent
from services.gateway.src.gateway.ielts import reading
from services.gateway.src.gateway.ielts.models import Timing
from services.gateway.src.gateway.ielts.reading_digest import build_digest
from shared.src.shared.models import Session
from shared.src.shared.prompts import read_prompt
from tests.agent.agent_events import reply_of
from tests.agent.test_ielts_coach_agent import stream_of

CONFIG = "config/model_config.free.yaml"
PAPER = reading.get_set()
REPORT = reading.mark(
    PAPER,
    {q.id: ("zzz" if q.task_type == "matching_headings" else q.key) for q in PAPER.questions},
    Timing(allowed_s=3600, used_s=1500, late=False),
)
DIGEST = build_digest(REPORT)


def make_session(digest: str = DIGEST) -> Session:
    return Session(candidate_id="c1", test_type="ielts", section="reading", report_digest=digest)


async def run_turn(message: str, session: Session | None = None, trace_id: str = ""):
    agent = IeltsReadingAgent(model_config_path=CONFIG)
    with patch("litellm.acompletion", new_callable=AsyncMock) as llm:
        llm.return_value = stream_of(["Match the main idea, ", "not a detail."])
        items = [i async for i in agent.run(session or make_session(), message, trace_id)]
    return items, llm


@pytest.mark.asyncio
async def test_the_reply_streams_from_one_model_call_on_the_reading_coach_role():
    items, llm = await run_turn("why did I miss the matching headings?")
    assert reply_of(items) == "Match the main idea, not a detail."
    llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_digest_rides_the_learners_turn_not_the_system_prompt():
    _, llm = await run_turn("what do I fix first?")
    system, last = llm.await_args.kwargs["messages"][0], llm.await_args.kwargs["messages"][-1]
    assert DIGEST not in system["content"]
    assert system["content"] == read_prompt("ielts_reading_coach", "tutor", "v1")
    assert last["role"] == "user" and DIGEST in last["content"]


@pytest.mark.asyncio
async def test_a_turn_without_a_result_asks_for_one_and_makes_no_model_call():
    items, llm = await run_turn("coach me", make_session(digest=""))
    assert "Reading" in reply_of(items)
    llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_turn_is_costed_like_a_gre_turn(turn_log):
    await run_turn("why?", trace_id="t-read")
    record = turn_log.last_with_trace("t-read")
    assert record is not None
    assert record["agent"] == "ielts_reading_coach" and record["llm_calls"] == 1
