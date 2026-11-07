"""
The IELTS completer's fallback chain: the role's own model first, then each configured fallback in
order, and `ModelUnavailableError` only once every one of them has failed or come back empty.

The retries, the order and the failover are the LiteLLM Router's (see test_llm_router); these
tests run the completer on a real Router whose providers are litellm mocks, and check what the
completer adds: the empty-reply rule, the error it raises, and the ledger.
"""
from pathlib import Path

import pytest

from services.agent.src.agent.ielts import llm
from services.agent.src.agent.ielts.llm import LiteLLMCompleter, ModelUnavailableError
from shared.src.shared.budget import Ledger
from shared.src.shared.config import Settings, load_model_config
from shared.src.shared.llm_router import build_router, configured_chain
from tests.llm_fakes import FAST, SERVER_ERROR, calls, mock_router

GOOGLE = "gemini/gemma-4-26b-a4b-it"
PRIMARY = "openrouter/google/gemma-4-26b-a4b-it:free"
BACKUP = "openrouter/google/gemma-4-31b-it:free"
LAST = "openrouter/google/gemma-4-12b-it:free"
MESSAGES = [{"role": "user", "content": "grade this"}]


@pytest.fixture
def chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "role_model", lambda role: PRIMARY)


async def run(router, tmp_path: Path) -> str:
    ledger = Ledger(tmp_path / "l.json", ceiling_usd=5.0)
    completer = LiteLLMCompleter(ledger, router=router)
    return await completer(llm.GRADER_ROLE, MESSAGES, temperature=0, max_tokens=500)


async def test_a_working_primary_never_touches_the_fallback(chain: None, tmp_path: Path):
    router = mock_router({PRIMARY: "from 26b", BACKUP: "from 31b"}, {PRIMARY: [BACKUP]})
    assert await run(router, tmp_path) == "from 26b"
    assert calls(router) == {PRIMARY: 1}


async def test_a_primary_error_falls_back_to_the_next_model(chain: None, tmp_path: Path):
    router = mock_router({PRIMARY: SERVER_ERROR, BACKUP: "from 31b"}, {PRIMARY: [BACKUP]})
    assert await run(router, tmp_path) == "from 31b"
    assert set(calls(router)) == {PRIMARY, BACKUP}


async def test_an_empty_primary_reply_falls_back_too(chain: None, tmp_path: Path):
    router = mock_router({PRIMARY: "  ", BACKUP: "from 31b"}, {PRIMARY: [BACKUP]})
    assert await run(router, tmp_path) == "from 31b"
    assert calls(router) == {PRIMARY: 1, BACKUP: 1}


async def test_an_empty_reply_in_the_middle_of_the_chain_moves_on_to_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(llm, "role_model", lambda role: PRIMARY)
    router = mock_router(
        {PRIMARY: SERVER_ERROR, BACKUP: "", LAST: "from the last"}, {PRIMARY: [BACKUP, LAST]}
    )
    assert await run(router, tmp_path) == "from the last"
    assert set(calls(router)) == {PRIMARY, BACKUP, LAST}


async def test_every_model_failing_raises_model_unavailable_naming_each(
    chain: None, tmp_path: Path
):
    router = mock_router({PRIMARY: SERVER_ERROR, BACKUP: ""}, {PRIMARY: [BACKUP]})
    with pytest.raises(ModelUnavailableError) as raised:
        await run(router, tmp_path)
    assert PRIMARY in str(raised.value) and BACKUP in str(raised.value)


async def test_a_router_failure_names_the_whole_chain_it_tried(chain: None, tmp_path: Path):
    router = mock_router({PRIMARY: SERVER_ERROR, BACKUP: SERVER_ERROR}, {PRIMARY: [BACKUP]})
    with pytest.raises(ModelUnavailableError) as raised:
        await run(router, tmp_path)
    assert PRIMARY in str(raised.value) and BACKUP in str(raised.value)


async def test_the_shipped_free_config_feeds_the_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        llm, "get_settings",
        lambda: Settings(model_config_path="config/model_config.free.yaml", gemini_api_key="k"),
    )
    config = load_model_config("config/model_config.free.yaml")
    outcomes = {GOOGLE: SERVER_ERROR, PRIMARY: SERVER_ERROR, BACKUP: "from 31b"}
    router = build_router(
        config, policy=FAST,
        deployment_params={m: {"mock_response": o, "api_key": "k"} for m, o in outcomes.items()},
    )
    assert configured_chain(router, GOOGLE) == [GOOGLE, PRIMARY, BACKUP]
    assert await run(router, tmp_path) == "from 31b"
    assert set(calls(router)) == {GOOGLE, PRIMARY, BACKUP}


async def test_google_calls_retry_transient_errors_before_leaving_google(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The ADR-0014 case: Google answers its own blips with a 500/503, about 1 call in 6."""
    monkeypatch.setattr(llm, "role_model", lambda role: GOOGLE)
    router = mock_router({GOOGLE: SERVER_ERROR, PRIMARY: "ok"}, {GOOGLE: [PRIMARY]})
    assert await run(router, tmp_path) == "ok"
    assert calls(router) == {GOOGLE: 1 + FAST.num_retries, PRIMARY: 1}


async def test_a_missing_google_key_hands_the_call_to_openrouter_at_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A 401 is not a transient error: no retry, straight to the fallback."""
    from tests.llm_fakes import Unauthorized

    monkeypatch.setattr(llm, "role_model", lambda role: GOOGLE)
    router = mock_router({GOOGLE: Unauthorized(), PRIMARY: "ok"}, {GOOGLE: [PRIMARY]})
    assert await run(router, tmp_path) == "ok"
    assert calls(router) == {GOOGLE: 1, PRIMARY: 1}


async def test_a_google_reply_costs_the_ledger_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(llm, "role_model", lambda role: GOOGLE)
    router = mock_router({GOOGLE: "graded"})
    ledger = Ledger(tmp_path / "l.json", ceiling_usd=0.01)
    await LiteLLMCompleter(ledger, router=router)(
        llm.GRADER_ROLE, MESSAGES, temperature=0, max_tokens=4000
    )
    assert ledger.spent() == 0.0


async def test_the_ledger_is_charged_for_the_model_that_answered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(llm, "role_model", lambda role: GOOGLE)
    router = mock_router({GOOGLE: SERVER_ERROR, "openai/gpt-4.1-mini": "paid reply"},
                         {GOOGLE: ["openai/gpt-4.1-mini"]})
    ledger = Ledger(tmp_path / "l.json", ceiling_usd=5.0)
    await LiteLLMCompleter(ledger, router=router)(
        llm.GRADER_ROLE, MESSAGES, temperature=0, max_tokens=500
    )
    assert ledger.spent() > 0.0  # the free Google model was skipped, the paid one answered


async def test_a_call_is_refused_before_any_model_when_a_fallback_could_pass_the_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The cap is checked at the dearest model of the chain: no fallback can take it past."""
    from shared.src.shared.budget import BudgetExceededError

    monkeypatch.setattr(llm, "role_model", lambda role: GOOGLE)
    router = mock_router({GOOGLE: SERVER_ERROR, "openai/gpt-4.1-mini": "paid reply"},
                         {GOOGLE: ["openai/gpt-4.1-mini"]})
    ledger = Ledger(tmp_path / "l.json", ceiling_usd=5.0)
    ledger.record("earlier", "gpt-4.1-mini", 4.9999)
    with pytest.raises(BudgetExceededError):  # not wrapped as a model outage
        await LiteLLMCompleter(ledger, router=router)(
            llm.GRADER_ROLE, MESSAGES, temperature=0, max_tokens=2000
        )
    assert calls(router) == {}  # the free first model was not tried either


async def test_the_cap_is_checked_against_the_chain_the_router_will_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A fallback the Router would use counts, whatever the role's own list says."""
    from shared.src.shared.budget import BudgetExceededError

    monkeypatch.setattr(llm, "role_model", lambda role: GOOGLE)
    router = mock_router({GOOGLE: SERVER_ERROR, "openai/gpt-4.1-mini": "paid reply"},
                         {GOOGLE: ["openai/gpt-4.1-mini"]})
    ledger = Ledger(tmp_path / "l.json", ceiling_usd=5.0)
    ledger.record("earlier", "gpt-4.1-mini", 4.9999)
    with pytest.raises(BudgetExceededError):
        await LiteLLMCompleter(ledger, router=router)(
            llm.GRADER_ROLE, MESSAGES, temperature=0, max_tokens=2000
        )
    assert calls(router) == {}


async def test_a_paid_call_after_an_empty_reply_is_checked_against_the_cap_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The empty first reply costs money; the next paid model must not run past the cap."""
    from shared.src.shared.budget import BudgetExceededError, estimate_cost, estimate_tokens

    first, second = "openai/gpt-4.1-mini", "openai/gpt-4o"
    monkeypatch.setattr(llm, "role_model", lambda role: first)
    router = mock_router({first: "  ", second: "paid reply"}, {first: [second]})
    worst_case = estimate_cost(second, estimate_tokens(MESSAGES[0]["content"]), 2000)
    ledger = Ledger(tmp_path / "l.json", ceiling_usd=1.0 + worst_case)
    ledger.record("earlier", "gpt-4.1-mini", 1.0)
    with pytest.raises(BudgetExceededError):
        await LiteLLMCompleter(ledger, router=router)(
            llm.GRADER_ROLE, MESSAGES, temperature=0, max_tokens=2000
        )
    assert calls(router) == {first: 1}
    assert ledger.spent() > 1.0  # the empty reply was paid for and recorded
