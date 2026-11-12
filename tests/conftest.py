import socket
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def turn_log(tmp_path):
    """
    Turns write their measurement record to a log; point it at a temp file so tests never
    touch the real `.turn-log/`. Tests that assert on records use the returned TurnLog.
    """
    from shared.src.shared.turn_meter import TurnLog

    log = TurnLog(tmp_path / "turn-log" / "turns.jsonl")
    with patch("shared.src.shared.turn_meter.default_turn_log", return_value=log):
        yield log


@pytest.fixture(autouse=True)
def local_feedback_files(tmp_path, monkeypatch):
    """
    Feedback votes and the opt-in text capture are local files. Point them at temp paths and
    pin capture off, whatever the developer's `.env` says; tests that need capture turn it on.
    """
    from shared.src.shared.config import clear_settings_caches

    monkeypatch.setenv("FEEDBACK_LOG_PATH", str(tmp_path / "turn-log" / "feedback.jsonl"))
    monkeypatch.setenv("TURN_CAPTURE_PATH", str(tmp_path / "turn-log" / "capture.jsonl"))
    monkeypatch.setenv("TURN_CAPTURE_TEXT", "0")
    clear_settings_caches()
    yield
    clear_settings_caches()


@pytest.fixture(autouse=True)
def dev_model_config(monkeypatch):
    """
    Tests assert on the dev model config (OpenAI roles). A developer's .env or shell may set
    MODEL_CONFIG_PATH (for example to config/model_config.free.yaml), so pin it here.
    """
    from shared.src.shared.config import clear_settings_caches

    monkeypatch.setenv("MODEL_CONFIG_PATH", "config/model_config.dev.yaml")
    clear_settings_caches()
    yield
    clear_settings_caches()


@pytest.fixture(autouse=True)
def fresh_router_with_instant_retries(monkeypatch):
    """
    Every model call goes through the LiteLLM Router (shared.llm_router). The process Router is
    built per test (each test has its own event loop), and its backoff between retries of a
    transient error is skipped so a test that provokes a retry does not sleep.
    """
    import litellm

    from shared.src.shared.llm_router import reset_router

    monkeypatch.setattr(litellm, "_calculate_retry_after", lambda *a, **k: 0)
    reset_router()
    yield
    reset_router()


@pytest.fixture(autouse=True)
def routers_unhook_from_litellm(monkeypatch):
    """
    Every Router registers its usage and failure callbacks in litellm's process-wide lists, and
    only `Router.discard()` takes them out. litellm refuses to add past 100 (MAX_CALLBACKS), so
    once the suite had built that many Routers the next one silently lost the failure callback
    its cooldown depends on, and the cooldown tests failed depending on test order. Discard each
    Router a test builds when the test ends.
    """
    from litellm.router import Router

    built: list[Router] = []
    real_init = Router.__init__

    def tracking_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(Router, "__init__", tracking_init)
    yield
    for router in built:
        router.discard()


@pytest.fixture(autouse=True)
def fresh_gateway_rate_limiter():
    """
    The gateway's /chat limiter (60 a minute per candidate) is one shared object, and the tests
    share one candidate id: with it shared, the whole suite sat a few requests under the limit
    and the next added /chat test made unrelated crisis-path and moderation tests answer "Too
    many requests". Each test gets its own.
    """
    from services.gateway.src.gateway.guardrails.rate_limit import RateLimiter
    from tests.gateway_fakes import rate_limiter_of

    with rate_limiter_of(RateLimiter()):
        yield


@pytest.fixture(autouse=True)
def no_live_corpus_service(monkeypatch: pytest.MonkeyPatch):
    """
    Agents build a RetrieveCorpus by default; keep tests off a corpus service that
    may be running on localhost (e.g. the docker compose stack). Tests that exercise
    retrieval inject their own corpus capability.

    Uses the same `monkeypatch` a test uses to re-patch `_call_corpus`, so both unwind in
    one last-in-first-out order and no stub outlives its test.
    """
    from services.agent.src.agent.capabilities.retrieve_corpus import RetrieveCorpus

    monkeypatch.setattr(RetrieveCorpus, "_call_corpus", AsyncMock(return_value=[]))
    yield


@pytest.fixture(autouse=True)
def math_tools_off_whatever_the_environment_says(monkeypatch: pytest.MonkeyPatch):
    """
    The product default is the original single-call streamed path, and most agent tests script
    one streamed model reply. A developer's `.env` with MATH_TOOLS_ENABLED=true must not change
    that. Tests of the tool path build their agent with `math_tools_enabled=True`.
    """
    from services.agent.src.agent.settings import get_settings

    monkeypatch.setattr(get_settings(), "math_tools_enabled", False)


@pytest.fixture(autouse=True)
def upfront_retrieval_whatever_the_environment_says(monkeypatch: pytest.MonkeyPatch):
    """
    The product default is the up-front search and most agent tests script one streamed reply.
    A developer's `.env` with RETRIEVAL_MODE=just_in_time must not change that. Tests of the
    just-in-time path build their agent with `retrieval_mode="just_in_time"`.
    """
    from services.agent.src.agent.settings import get_settings

    monkeypatch.setattr(get_settings(), "retrieval_mode", "upfront")


@pytest.fixture
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Fail any test that tries to reach the network. Local Unix sockets stay open: the math
    verifier forks each check from a helper process it talks to over one.
    """
    real_connect = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> Any:
        if self.family == socket.AF_UNIX:
            return real_connect(self, address)
        raise AssertionError("network access attempted")

    def refuse(*a: Any, **k: Any) -> Any:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "create_connection", refuse)
