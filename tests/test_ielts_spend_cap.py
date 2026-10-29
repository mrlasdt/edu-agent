"""
The AI spend cap on the IELTS mock: every paid call is checked against the shared ledger before
it is made and recorded after it, and once the cap is used up the mock refuses new attempts and
markings with a message that says why, while a saved essay survives.
"""
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from services.agent.src.agent.ielts import llm
from services.agent.src.agent.ielts.llm import LiteLLMCompleter
from services.gateway.src.gateway import ielts_api
from services.gateway.src.gateway.ielts_store import AttemptStore
from services.gateway.src.gateway.main import app
from shared.src.shared.budget import DEFAULT_CEILING_USD, BudgetExceededError, Ledger
from shared.src.shared.config import Settings
from tests.test_ielts_marking import ESSAY, Scripted, feedback_reply, grade_reply

MESSAGES = [{"role": "system", "content": "s" * 400}, {"role": "user", "content": "u" * 400}]


def test_the_default_cap_is_five_dollars_everywhere():
    assert DEFAULT_CEILING_USD == 5.0
    assert Settings().ai_spend_cap_usd == 5.0
    assert Settings().ai_ledger_path == ".eval-ledger/ledger.json"


# ── the completer meters itself ──────────────────────────────────────────────


class FakeLiteLLM:
    def __init__(self, prompt_tokens: int = 1000, completion_tokens: int = 100) -> None:
        self.calls = 0
        self.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )

    async def __call__(self, **kw: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=self.usage
        )


@pytest.fixture
def fake_litellm(monkeypatch: pytest.MonkeyPatch) -> FakeLiteLLM:
    fake = FakeLiteLLM()
    monkeypatch.setattr(llm.litellm, "acompletion", fake)
    monkeypatch.setattr(llm, "role_model", lambda role: "openai/gpt-4.1-mini")
    return fake


async def test_a_paid_call_is_recorded_from_the_reported_usage(
    tmp_path: Path, fake_litellm: FakeLiteLLM
):
    ledger = Ledger(tmp_path / "l.json", ceiling_usd=5.0)
    text = await LiteLLMCompleter(ledger)(
        llm.GRADER_ROLE, MESSAGES, temperature=0, max_tokens=500, trace_id="a1"
    )
    assert text == "ok"
    # gpt-4.1-mini: 1000 in at 0.40/M + 100 out at 1.60/M
    assert ledger.spent() == pytest.approx(0.00056)
    assert Ledger(tmp_path / "l.json", 5.0).spent() == pytest.approx(0.00056)  # persisted


async def test_a_free_model_costs_nothing(
    tmp_path: Path, fake_litellm: FakeLiteLLM, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(llm, "role_model", lambda role: "openrouter/openrouter/free")
    ledger = Ledger(tmp_path / "l.json", ceiling_usd=0.0)
    await LiteLLMCompleter(ledger)(llm.GRADER_ROLE, MESSAGES, temperature=0, max_tokens=500)
    assert ledger.spent() == 0 and fake_litellm.calls == 1


async def test_a_call_that_could_pass_the_cap_is_refused_before_it_is_made(
    tmp_path: Path, fake_litellm: FakeLiteLLM
):
    ledger = Ledger(tmp_path / "l.json", ceiling_usd=5.0)
    ledger.record("earlier", "gpt-4.1-mini", 4.9999)
    completer = LiteLLMCompleter(ledger)
    with pytest.raises(BudgetExceededError):  # not wrapped as a model outage
        await completer(llm.GRADER_ROLE, MESSAGES, temperature=0, max_tokens=2000)
    assert fake_litellm.calls == 0
    assert ledger.spent() == pytest.approx(4.9999)


async def test_calls_in_flight_count_against_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Three samples run together, so each must see what the others have reserved."""
    import asyncio

    started = asyncio.Event()
    release = asyncio.Event()
    n = 0

    async def slow(**kw: Any) -> Any:
        nonlocal n
        n += 1
        started.set()
        await release.wait()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
        )

    monkeypatch.setattr(llm.litellm, "acompletion", slow)
    monkeypatch.setattr(llm, "role_model", lambda role: "openai/gpt-4.1-mini")
    # worst case per call here is (200/4 prompt tokens at 0.40 + 2000 out at 1.60)/M = 0.00322
    ledger = Ledger(tmp_path / "l.json", ceiling_usd=0.005)  # room for one call, not two
    completer = LiteLLMCompleter(ledger)
    first = asyncio.create_task(
        completer(llm.GRADER_ROLE, MESSAGES[:1], temperature=0, max_tokens=2000)
    )
    await started.wait()
    with pytest.raises(BudgetExceededError):
        await completer(llm.GRADER_ROLE, MESSAGES[:1], temperature=0, max_tokens=2000)
    release.set()
    assert await first == "ok" and n == 1
    # the reservation is released: a later call fits again
    await completer(llm.GRADER_ROLE, MESSAGES[:1], temperature=0, max_tokens=2000)


async def test_a_provider_failure_still_releases_its_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    async def boom(**kw: Any) -> Any:
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm.litellm, "acompletion", boom)
    monkeypatch.setattr(llm, "role_model", lambda role: "openai/gpt-4.1-mini")
    ledger = Ledger(tmp_path / "l.json", ceiling_usd=5.0)
    completer = LiteLLMCompleter(ledger)
    with pytest.raises(llm.ModelUnavailableError):
        await completer(llm.GRADER_ROLE, MESSAGES, temperature=0, max_tokens=500)
    assert completer._reserved == 0 and ledger.spent() == 0


# ── the API refuses once the cap is used up ──────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> AttemptStore:
    return AttemptStore(tmp_path / "ielts-local" / "attempts.sqlite3")


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.json", ceiling_usd=5.0)


class Metered:
    """A scripted model behind the same pre-call check the production completer makes."""

    def __init__(self, inner: Scripted, ledger: Ledger) -> None:
        self.inner, self.ledger = inner, ledger

    async def __call__(self, role: str, messages: list[dict[str, str]], **kw: Any) -> str:
        self.ledger.ensure_capacity(0.003)
        return await self.inner(role, messages, **kw)


@pytest.fixture
def completer(ledger: Ledger) -> Metered:
    return Metered(Scripted([grade_reply(6, 6, 6, 6)] * 30, feedback_reply()), ledger)


@pytest.fixture
def client(store: AttemptStore, ledger: Ledger, completer: Scripted) -> Iterator[TestClient]:
    app.dependency_overrides[ielts_api.get_store] = lambda: store
    app.dependency_overrides[ielts_api.get_completer] = lambda: completer
    app.dependency_overrides[ielts_api.get_ledger] = lambda: ledger
    ielts_api._limiter._minute_window.clear()
    ielts_api._limiter._day_counts.clear()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def use_up(ledger: Ledger, left: float = 0.0) -> None:
    ledger.record("earlier", "gpt-4.1-mini", ledger.ceiling_usd - left)


def test_the_budget_endpoint_reports_spend_against_the_cap(client: TestClient, ledger: Ledger):
    ledger.record("earlier", "gpt-4.1-mini", 0.25)
    budget = client.get("/ielts/budget").json()
    assert budget == {
        "cap_usd": 5.0, "spent_usd": 0.25, "remaining_usd": 4.75, "exhausted": False,
        "message": None,
    }


def test_a_new_mock_is_refused_with_a_clear_message_once_the_cap_is_reached(
    client: TestClient, ledger: Ledger
):
    use_up(ledger)
    budget = client.get("/ielts/budget").json()
    assert budget["exhausted"] and "$5.00" in budget["message"]
    refused = client.post("/ielts/attempts", json={"prompt_id": "ielts_writing_prompt_1"})
    assert refused.status_code == 402
    assert "AI spend cap of $5.00 is used up" in refused.json()["detail"]
    assert "AI_SPEND_CAP_USD" in refused.json()["detail"]
    assert client.get("/ielts/attempts").json() == []  # nothing was created


def test_a_mock_that_could_not_be_marked_is_refused_before_the_clock_starts(
    client: TestClient, ledger: Ledger
):
    use_up(ledger, left=ielts_api.MOCK_RESERVE_USD / 2)
    refused = client.post("/ielts/attempts", json={"prompt_id": "ielts_writing_prompt_1"})
    assert refused.status_code == 402


def test_a_mock_is_allowed_while_there_is_room_for_one(client: TestClient, ledger: Ledger):
    use_up(ledger, left=ielts_api.MOCK_RESERVE_USD * 2)
    created = client.post("/ielts/attempts", json={"prompt_id": "ielts_writing_prompt_1"})
    assert created.status_code == 201


def test_the_cap_reached_mid_attempt_keeps_the_essay_and_marking_resumes_when_it_is_raised(
    client: TestClient, ledger: Ledger
):
    attempt_id = client.post(
        "/ielts/attempts", json={"prompt_id": "ielts_writing_prompt_1"}
    ).json()["id"]
    client.post(f"/ielts/attempts/{attempt_id}/start")
    use_up(ledger)  # the cap is reached while the candidate is writing
    client.post(f"/ielts/attempts/{attempt_id}/submit", json={"text": ESSAY})
    view = client.get(f"/ielts/attempts/{attempt_id}").json()
    assert view["status"] == "ungraded" and view["retryable"] and view["draft"] == ESSAY
    assert "AI spend cap of $5.00 is used up" in view["error"]
    assert client.post(f"/ielts/attempts/{attempt_id}/regrade").status_code == 402

    ledger.ceiling_usd = 10.0  # the owner raised the cap
    assert client.post(f"/ielts/attempts/{attempt_id}/regrade").status_code == 200
    assert client.get(f"/ielts/attempts/{attempt_id}").json()["status"] == "graded"


def test_a_budget_stop_inside_the_grader_is_not_reported_as_a_model_outage(
    client: TestClient, completer: Metered
):
    completer.inner.grades = [BudgetExceededError("budget guard: ...")] * 3
    attempt_id = client.post(
        "/ielts/attempts", json={"prompt_id": "ielts_writing_prompt_1"}
    ).json()["id"]
    client.post(f"/ielts/attempts/{attempt_id}/start")
    client.post(f"/ielts/attempts/{attempt_id}/submit", json={"text": ESSAY})
    view = client.get(f"/ielts/attempts/{attempt_id}").json()
    assert view["status"] == "ungraded" and "spend cap" in view["error"]
    assert "unavailable" not in view["error"]


def test_a_budget_stop_in_the_feedback_stage_keeps_the_essay_and_says_why(
    client: TestClient, completer: Metered
):
    completer.inner.feedback = BudgetExceededError("budget guard: ...")
    attempt_id = client.post(
        "/ielts/attempts", json={"prompt_id": "ielts_writing_prompt_1"}
    ).json()["id"]
    client.post(f"/ielts/attempts/{attempt_id}/start")
    client.post(f"/ielts/attempts/{attempt_id}/submit", json={"text": ESSAY})
    view = client.get(f"/ielts/attempts/{attempt_id}").json()
    assert view["status"] == "ungraded" and view["retryable"] and view["draft"] == ESSAY
    assert "spend cap" in view["error"]

    completer.inner.feedback = feedback_reply()
    assert client.post(f"/ielts/attempts/{attempt_id}/regrade").status_code == 200
    assert client.get(f"/ielts/attempts/{attempt_id}").json()["status"] == "graded"
