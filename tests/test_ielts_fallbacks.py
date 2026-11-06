"""
The IELTS completer's fallback chain: the role's own model first, then each configured fallback in
order, and `ModelUnavailableError` only once every one of them has failed or come back empty.
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from services.agent.src.agent.ielts import llm
from services.agent.src.agent.ielts.llm import LiteLLMCompleter, ModelUnavailableError
from shared.src.shared.budget import Ledger
from shared.src.shared.config import Settings

PRIMARY = "openrouter/google/gemma-4-26b-a4b-it:free"
BACKUP = "openrouter/google/gemma-4-31b-it:free"
MESSAGES = [{"role": "user", "content": "grade this"}]


def reply(text: str) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
    )


class Script:
    """A fake `litellm.acompletion` per model: a string is the reply, an Exception is raised."""

    def __init__(self, outcomes: dict[str, Any]) -> None:
        self.outcomes = outcomes
        self.tried: list[str] = []

    async def __call__(self, **kw: Any) -> Any:
        self.tried.append(kw["model"])
        outcome = self.outcomes[kw["model"]]
        if isinstance(outcome, Exception):
            raise outcome
        return reply(outcome)


@pytest.fixture
def chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "role_model", lambda role: PRIMARY)
    monkeypatch.setattr(llm, "role_fallbacks", lambda role: [BACKUP])


async def run(script: Script, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    monkeypatch.setattr(llm.litellm, "acompletion", script)
    ledger = Ledger(tmp_path / "l.json", ceiling_usd=5.0)
    return await LiteLLMCompleter(ledger)(llm.GRADER_ROLE, MESSAGES, temperature=0, max_tokens=500)


async def test_a_working_primary_never_touches_the_fallback(
    chain: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    script = Script({PRIMARY: "from 26b", BACKUP: "from 31b"})
    assert await run(script, monkeypatch, tmp_path) == "from 26b"
    assert script.tried == [PRIMARY]


async def test_a_primary_error_falls_back_to_the_next_model(
    chain: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    script = Script({PRIMARY: RuntimeError("429 rate limited"), BACKUP: "from 31b"})
    assert await run(script, monkeypatch, tmp_path) == "from 31b"
    assert script.tried == [PRIMARY, BACKUP]


async def test_an_empty_primary_reply_falls_back_too(
    chain: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    script = Script({PRIMARY: "  ", BACKUP: "from 31b"})
    assert await run(script, monkeypatch, tmp_path) == "from 31b"
    assert script.tried == [PRIMARY, BACKUP]


async def test_every_model_failing_raises_model_unavailable_naming_each(
    chain: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    script = Script({PRIMARY: RuntimeError("429"), BACKUP: ""})
    with pytest.raises(ModelUnavailableError) as raised:
        await run(script, monkeypatch, tmp_path)
    assert script.tried == [PRIMARY, BACKUP]
    assert PRIMARY in str(raised.value) and BACKUP in str(raised.value)


async def test_the_shipped_free_config_feeds_the_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        llm, "get_settings", lambda: Settings(model_config_path="config/model_config.free.yaml")
    )
    script = Script({PRIMARY: RuntimeError("429"), BACKUP: "from 31b"})
    assert await run(script, monkeypatch, tmp_path) == "from 31b"
    assert script.tried == [PRIMARY, BACKUP]
