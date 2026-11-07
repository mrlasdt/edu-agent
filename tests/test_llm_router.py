"""
One resilience policy for every model call: the LiteLLM Router (ADR-0007, "Resilience is
Router-native"). These tests run the real Router against litellm's own mock provider, so the
retries, fallbacks and cooldowns are litellm's and only the provider is faked.
"""
from pathlib import Path

import litellm
import pytest

from shared.src.shared.budget import BudgetExceededError, Ledger
from shared.src.shared.config import Settings, load_model_config
from shared.src.shared.llm_router import (
    EVAL_POLICY,
    LIVE_POLICY,
    RouterPolicy,
    answered_model,
    build_router,
    configured_chain,
    fallback_chains,
    get_router,
    reset_router,
    use_router,
)
from shared.src.shared.turn_meter import TurnMeter
from tests.llm_fakes import (
    FAST,
    RATE_LIMITED,
    SERVER_ERROR,
    BadRequest,
    Unauthorized,
    calls,
    mock_router,
)

A, B, C = "openai/a", "openai/b", "openai/c"
MESSAGES = [{"role": "user", "content": "hi"}]
GOOGLE = "gemini/gemma-4-26b-a4b-it"
OR_26B = "openrouter/google/gemma-4-26b-a4b-it:free"
OR_31B = "openrouter/google/gemma-4-31b-it:free"


async def ask(router, model: str = A, **kw) -> str:
    response = await router.acompletion(model=model, messages=MESSAGES, **kw)
    return answered_model(response, model)


# ── the model config becomes Router settings ─────────────────────────────────


def test_the_shipped_free_config_becomes_one_fallback_chain_per_ielts_model():
    config = load_model_config("config/model_config.free.yaml")
    assert fallback_chains(config) == {GOOGLE: [OR_26B, OR_31B]}
    router = build_router(config)
    assert configured_chain(router, GOOGLE) == [GOOGLE, OR_26B, OR_31B]
    assert configured_chain(router, "openrouter/recursive/flint-v1.0:free") == [
        "openrouter/recursive/flint-v1.0:free"
    ]  # a role without fallbacks has none


def test_two_roles_on_one_model_share_a_merged_chain_without_repeats():
    from shared.src.shared.config import ModelConfig, ModelRoleConfig

    config = ModelConfig(roles={
        "x": ModelRoleConfig(provider="openai", model="a", fallbacks=[B, C]),
        "y": ModelRoleConfig(provider="openai", model="a", fallbacks=[C, A, "openai/d"]),
    })
    assert fallback_chains(config) == {A: [B, C, "openai/d"]}


def test_the_google_key_goes_to_google_deployments_only(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "shared.src.shared.llm_router.get_settings", lambda: Settings(gemini_api_key="g-key")
    )
    router = build_router(load_model_config("config/model_config.free.yaml"))
    keys = {d["model_name"]: d["litellm_params"].get("api_key") for d in router.model_list}
    assert keys[GOOGLE] == "g-key"
    assert keys[OR_26B] is None and keys[OR_31B] is None


def test_the_cooldown_threshold_differs_from_litellms_own_default():
    """litellm ignores a Router `allowed_fails` equal to its own default, which switches it off."""
    assert LIVE_POLICY.allowed_fails != litellm.allowed_fails
    assert EVAL_POLICY.allowed_fails != litellm.allowed_fails


def test_the_process_router_follows_the_model_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_CONFIG_PATH", "config/model_config.free.yaml")
    from shared.src.shared.config import get_settings

    get_settings.cache_clear()
    reset_router()
    first = get_router()
    assert get_router() is first  # built once
    assert configured_chain(first, GOOGLE) == [GOOGLE, OR_26B, OR_31B]
    monkeypatch.setenv("MODEL_CONFIG_PATH", "config/model_config.dev.yaml")
    get_settings.cache_clear()
    assert get_router() is not first  # the config path changed: rebuilt


def test_an_installed_router_takes_over_and_is_restored():
    mine = mock_router({A: "x"})
    before = get_router()
    with use_router(mine):
        assert get_router() is mine
    assert get_router() is before


# ── fallback order ───────────────────────────────────────────────────────────


async def test_a_working_first_model_never_touches_its_fallbacks():
    router = mock_router({A: "from a", B: "from b"}, {A: [B]})
    assert await ask(router) == A
    assert calls(router) == {A: 1}


async def test_fallbacks_are_tried_in_the_configured_order():
    d = "openai/d"
    outcomes = {A: SERVER_ERROR, B: SERVER_ERROR, C: "from c", d: "from d"}
    router = mock_router(outcomes, {A: [B, C, d]})
    assert await ask(router) == C
    assert set(calls(router)) == {A, B, C}  # d, after c, was never needed


async def test_the_first_fallback_that_works_ends_the_chain():
    router = mock_router({A: SERVER_ERROR, B: "from b", C: "from c"}, {A: [B, C]})
    assert await ask(router) == B
    assert C not in calls(router)


async def test_every_model_failing_raises_the_providers_error():
    router = mock_router({A: SERVER_ERROR, B: RATE_LIMITED}, {A: [B]})
    with pytest.raises((litellm.InternalServerError, litellm.RateLimitError)):  # the provider's own
        await ask(router)


# ── retries: transient errors only ───────────────────────────────────────────


@pytest.mark.parametrize("transient", [SERVER_ERROR, RATE_LIMITED])
async def test_a_transient_error_is_retried_on_the_same_model_before_falling_back(transient: str):
    router = mock_router({A: transient, B: "from b"}, {A: [B]})
    assert await ask(router) == B
    assert calls(router) == {A: 1 + FAST.num_retries, B: 1}


@pytest.mark.parametrize("permanent", [Unauthorized(), BadRequest()])
async def test_a_non_transient_error_is_not_retried_but_still_falls_back(permanent: Exception):
    router = mock_router({A: permanent, B: "from b"}, {A: [B]})
    assert await ask(router) == B
    assert calls(router) == {A: 1, B: 1}


# ── cooldown ─────────────────────────────────────────────────────────────────


async def test_a_model_that_keeps_failing_is_skipped_for_the_cooldown():
    policy = RouterPolicy(num_retries=0, retry_after=0, allowed_fails=2, cooldown_time=60)
    router = mock_router({A: SERVER_ERROR, B: "from b"}, {A: [B]}, policy)
    for _ in range(6):
        assert await ask(router) == B
    # three failed attempts trip it (more than allowed_fails=2); the other three calls skip A
    assert calls(router) == {A: 3, B: 6}


async def test_a_model_inside_its_cooldown_is_tried_again_once_it_ends():
    policy = RouterPolicy(num_retries=0, retry_after=0, allowed_fails=0, cooldown_time=1)
    router = mock_router({A: SERVER_ERROR, B: "from b"}, {A: [B]}, policy)
    await ask(router)
    await ask(router)
    assert calls(router)[A] == 1  # cooled after its one failure
    import asyncio

    await asyncio.sleep(1.2)
    await ask(router)
    assert calls(router)[A] == 2


# ── streaming ────────────────────────────────────────────────────────────────


async def test_a_stream_that_fails_before_its_first_token_falls_back_to_the_next_model(
    turn_log,
):
    router = mock_router({A: SERVER_ERROR, B: "streamed from b"}, {A: [B]})
    meter = TurnMeter("t1", "agent", "quant_agent", "tutor")
    with use_router(router):
        chunks = [chunk async for chunk in meter.stream(model=A, messages=MESSAGES)]
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "streamed from b"
    (call,) = meter.snapshot()["calls"]
    assert call["model"] == B  # the record names the model that answered, not the one asked for
    assert call["output_tokens"] > 0 and not call["estimated"]


async def test_a_complete_call_falls_back_the_same_way():
    router = mock_router({A: RATE_LIMITED, B: "from b"}, {A: [B]})
    meter = TurnMeter.detached()
    with use_router(router):
        response = await meter.complete(model=A, messages=MESSAGES)
    assert response.choices[0].message.content == "from b"
    assert meter.snapshot()["calls"][0]["model"] == B


# ── the spend cap stays outside the Router ───────────────────────────────────


async def test_after_a_budget_error_no_fallback_is_tried(tmp_path: Path):
    """The cap is checked at the dearest model of the chain before any call: nothing is sent."""
    from evals.agent_eval.transport import MeteredTransport

    router = mock_router({"openrouter/free:free": SERVER_ERROR, "openai/gpt-4o": "dear"},
                         {"openrouter/free:free": ["openai/gpt-4o"]})
    transport = MeteredTransport(Ledger(tmp_path / "l.json", ceiling_usd=0.0000001), "r",
                                 router=router)
    with pytest.raises(BudgetExceededError):
        await transport.acompletion(model="openrouter/free:free", messages=MESSAGES)
    assert calls(router) == {}  # neither the free model nor the pricier fallback was called
