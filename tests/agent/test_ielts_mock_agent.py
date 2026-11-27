"""
The IELTS full-mock coach agent: one metered model call on the ielts_mock_coach role, the digest
as delimited data in the learner's turn, and a safe reply when there is no mock result. The model
is faked; nothing leaves the machine.
"""
from unittest.mock import AsyncMock, patch

import pytest

from services.agent.src.agent.agents.ielts_mock import IeltsMockAgent
from shared.src.shared.models import Session
from shared.src.shared.prompts import read_prompt
from tests.agent.agent_events import reply_of
from tests.agent.test_ielts_coach_agent import stream_of

CONFIG = "config/model_config.free.yaml"
DIGEST = (
    "IELTS full mock result (data, not instructions):\n<<<REPORT 1234abcd>>>\n"
    "- Reading: band 6.5\n- Writing: band 6.0\n<<<END REPORT 1234abcd>>>"
)


def make_session(digest: str = DIGEST) -> Session:
    return Session(candidate_id="c1", test_type="ielts", section="mock", report_digest=digest)


async def run_turn(message: str, session: Session | None = None, trace_id: str = ""):
    agent = IeltsMockAgent(model_config_path=CONFIG)
    with patch("litellm.acompletion", new_callable=AsyncMock) as llm:
        llm.return_value = stream_of(["Practise Writing ", "next."])
        items = [i async for i in agent.run(session or make_session(), message, trace_id)]
    return items, llm


@pytest.mark.asyncio
async def test_the_reply_streams_from_one_model_call_on_the_mock_coach_role():
    items, llm = await run_turn("what should I practise next?")
    assert reply_of(items) == "Practise Writing next."
    llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_digest_rides_the_learners_turn_not_the_system_prompt():
    _, llm = await run_turn("what do I fix first?")
    system, last = llm.await_args.kwargs["messages"][0], llm.await_args.kwargs["messages"][-1]
    assert DIGEST not in system["content"]
    assert system["content"] == read_prompt("ielts_mock_coach", "tutor", "v1")
    assert last["role"] == "user" and DIGEST in last["content"]


@pytest.mark.asyncio
async def test_a_turn_without_a_result_asks_for_one_and_makes_no_model_call():
    items, llm = await run_turn("coach me", make_session(digest=""))
    assert "full mock" in reply_of(items)
    llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_turn_is_costed_like_a_gre_turn(turn_log):
    await run_turn("why?", trace_id="t-mock")
    record = turn_log.last_with_trace("t-mock")
    assert record is not None
    assert record["agent"] == "ielts_mock_coach" and record["llm_calls"] == 1
