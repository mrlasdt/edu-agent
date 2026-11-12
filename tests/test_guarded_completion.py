"""
`guarded_acompletion`: the one budget-check, Router call and ledger entry both metered call
sites (the IELTS completer and the eval transport) go through.
"""
from pathlib import Path

import pytest

from services.gateway.src.gateway.guarded_completion import (
    InFlight,
    failure_reason,
    guarded_acompletion,
)
from shared.src.shared.budget import BudgetExceededError, Ledger, estimate_cost
from tests.llm_fakes import SERVER_ERROR, mock_router

FREE = "openrouter/x:free"
PAID = "openai/gpt-4o"
MESSAGES = [{"role": "user", "content": "q" * 400}]


def _ledger(tmp_path: Path, ceiling: float = 20.0) -> Ledger:
    return Ledger(tmp_path / "l.json", ceiling)


async def test_a_call_is_recorded_under_the_model_that_answered(tmp_path: Path):
    router = mock_router({FREE: SERVER_ERROR, PAID: "ok"}, {FREE: [PAID]})
    ledger = _ledger(tmp_path)
    call = await guarded_acompletion(router, FREE, MESSAGES, ledger, "run", max_tokens=100)
    assert call.text == "ok" and call.answered == PAID and call.chain == [FREE, PAID]
    assert call.cost_usd > 0 and ledger.spent() == pytest.approx(call.cost_usd)


async def test_the_budget_is_checked_at_the_dearest_model_before_any_attempt(tmp_path: Path):
    router = mock_router({FREE: "free reply", PAID: "dear"}, {FREE: [PAID]})
    worst = estimate_cost(PAID, 100, 100)
    ledger = _ledger(tmp_path, ceiling=worst / 2)
    with pytest.raises(BudgetExceededError):
        await guarded_acompletion(router, FREE, MESSAGES, ledger, "run", max_tokens=100)
    assert router.total_calls == {}  # not even the free first model was tried
    assert ledger.spent() == 0


async def test_fallbacks_given_replace_the_configured_chain(tmp_path: Path):
    router = mock_router({FREE: "a", PAID: "b"}, {FREE: [PAID]})
    ledger = _ledger(tmp_path, ceiling=estimate_cost(PAID, 100, 100) / 2)
    # the caller resumes at FREE with nothing behind it, so the dear model is not in the chain
    call = await guarded_acompletion(
        router, FREE, MESSAGES, ledger, "run", max_tokens=100, fallbacks=[]
    )
    assert call.chain == [FREE] and call.cost_usd == 0


async def test_a_provider_failure_propagates_and_records_nothing(tmp_path: Path):
    router = mock_router({FREE: SERVER_ERROR})
    ledger = _ledger(tmp_path)
    with pytest.raises(Exception, match="(?i)server"):
        await guarded_acompletion(router, FREE, MESSAGES, ledger, "run", max_tokens=100)
    assert ledger.spent() == 0


async def test_in_flight_calls_count_against_the_cap_and_are_released(tmp_path: Path):
    router = mock_router({PAID: "ok"})
    one = estimate_cost(PAID, 100, 100)
    ledger = _ledger(tmp_path, ceiling=one * 1.5)
    inflight = InFlight(usd=one)  # another call already holds one worst case
    with pytest.raises(BudgetExceededError):
        await guarded_acompletion(
            router, PAID, MESSAGES, ledger, "run", max_tokens=100, inflight=inflight
        )
    assert inflight.usd == one  # a refused call reserves nothing
    inflight.usd = 0.0
    await guarded_acompletion(
        router, PAID, MESSAGES, ledger, "run", max_tokens=100, inflight=inflight
    )
    assert inflight.usd == 0.0  # released once the call is done


def test_failure_reason_redacts_keys_and_is_bounded():
    reason = failure_reason(RuntimeError("bad key sk-abcdefgh12345 " + "x" * 500), limit=60)
    assert reason.startswith("RuntimeError: bad key [redacted]") and len(reason) <= 60 + 14
    assert "sk-abc" not in reason
    assert "[redacted]" in failure_reason(RuntimeError("Authorization: Bearer abc.def"))
