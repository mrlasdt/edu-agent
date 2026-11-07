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
    from shared.src.shared.config import get_settings

    monkeypatch.setenv("FEEDBACK_LOG_PATH", str(tmp_path / "turn-log" / "feedback.jsonl"))
    monkeypatch.setenv("TURN_CAPTURE_PATH", str(tmp_path / "turn-log" / "capture.jsonl"))
    monkeypatch.setenv("TURN_CAPTURE_TEXT", "0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def dev_model_config(monkeypatch):
    """
    Tests assert on the dev model config (OpenAI roles). A developer's .env or shell may set
    MODEL_CONFIG_PATH (for example to config/model_config.free.yaml), so pin it here.
    """
    from shared.src.shared.config import get_settings

    monkeypatch.setenv("MODEL_CONFIG_PATH", "config/model_config.dev.yaml")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def fresh_router_with_instant_retries(monkeypatch):
    """
    Every model call goes through the LiteLLM Router (shared.llm_router). The process Router is
    built per test (each test has its own event loop), and its backoff between retries of a
    transient error is skipped so a test that provokes a retry does not sleep.

    Each Router registers its callbacks on litellm's process-global lists, which refuse new ones
    past 100. A suite that builds hundreds of Routers would then leave a later Router without
    its cooldown callback (a failing model never cools down), so the lists are restored after
    every test.
    """
    import litellm

    from shared.src.shared.llm_router import reset_router

    monkeypatch.setattr(litellm, "_calculate_retry_after", lambda *a, **k: 0)
    callback_lists = (
        "callbacks", "success_callback", "failure_callback",
        "_async_success_callback", "_async_failure_callback",
    )
    saved = {name: list(getattr(litellm, name)) for name in callback_lists}
    reset_router()
    yield
    reset_router()
    for name, callbacks in saved.items():
        setattr(litellm, name, callbacks)


@pytest.fixture(autouse=True)
def fresh_gateway_rate_limiter(monkeypatch):
    """
    The gateway's /chat limiter (60 a minute per candidate) is module state, and the tests share
    one candidate id: with it shared, the whole suite sat a few requests under the limit and the
    next added /chat test made unrelated crisis-path and moderation tests answer "Too many
    requests". Each test gets its own.
    """
    from services.gateway.src.gateway.guardrails.rate_limit import RateLimiter

    monkeypatch.setattr("services.gateway.src.gateway.main._rate_limiter", RateLimiter())


@pytest.fixture(autouse=True)
def no_live_corpus_service():
    """
    Agents build a RetrieveCorpusSkill by default; keep tests off a corpus service that
    may be running on localhost (e.g. the docker compose stack). Tests that exercise
    retrieval inject their own skill.
    """
    with patch(
        "services.agent.src.agent.skills.retrieve_corpus.RetrieveCorpusSkill._call_corpus",
        new=AsyncMock(return_value=[]),
    ):
        yield


@pytest.fixture(autouse=True)
def math_tools_off_whatever_the_environment_says(monkeypatch: pytest.MonkeyPatch):
    """
    The product default is the original single-call streamed path, and most agent tests script
    one streamed model reply. A developer's `.env` with MATH_TOOLS_ENABLED=true must not change
    that. Tests of the tool path build their agent with `math_tools_enabled=True`.
    """
    from shared.src.shared.config import get_settings

    monkeypatch.setattr(get_settings(), "math_tools_enabled", False)


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
