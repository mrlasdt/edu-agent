"""
POST /ielts/remark: the practice re-mark the coach's `mark_revision` tool asks for. It marks a
revised essay with the same grader, spend cap and rate limit as a mock, compares it with the
graded attempt being coached from, and stores nothing. Refusals say why.
"""
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from services.gateway.src.gateway import ielts_api
from services.gateway.src.gateway.ielts.llm import GRADER_ROLE
from services.gateway.src.gateway.ielts_store import AttemptStore
from services.gateway.src.gateway.limits import ielts_limiter
from services.gateway.src.gateway.main import app
from shared.src.shared.budget import BudgetExceededError, Ledger
from tests.gateway.test_ielts_marking import ESSAY, Scripted, feedback_reply, grade_reply


class Metered(Scripted):
    """A scripted model behind the same pre-call cap check the production completer makes."""

    def __init__(self, ledger: Ledger, grades: list[object]) -> None:
        super().__init__(grades, feedback_reply())
        self.ledger = ledger

    async def __call__(self, role: str, messages: list[dict[str, str]], **kw: Any) -> str:
        self.ledger.ensure_capacity(0.003)
        return await super().__call__(role, messages, **kw)


@pytest.fixture
def store(tmp_path: Path) -> AttemptStore:
    return AttemptStore(tmp_path / "attempts.sqlite3")


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.json", ceiling_usd=5.0)


@pytest.fixture
def completer(ledger: Ledger) -> Metered:
    return Metered(ledger, [grade_reply(6, 6, 6, 6)] * 3)  # the mock itself


@pytest.fixture
def client(store: AttemptStore, ledger: Ledger, completer: Metered) -> Iterator[TestClient]:
    app.dependency_overrides[ielts_api.get_store] = lambda: store
    app.dependency_overrides[ielts_api.get_completer] = lambda: completer
    app.dependency_overrides[ielts_api.get_ledger] = lambda: ledger
    ielts_limiter().reset()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def attempt_id(client: TestClient, completer: Metered) -> str:
    """A graded mock (all 6.0), after which the scripted model is ready for the re-mark."""
    made = client.post("/ielts/attempts", json={"prompt_id": "ielts_writing_prompt_1"}).json()["id"]
    client.post(f"/ielts/attempts/{made}/start")
    client.post(f"/ielts/attempts/{made}/submit", json={"text": ESSAY})
    assert client.get(f"/ielts/attempts/{made}").json()["status"] == "graded"
    completer.grades = [grade_reply(7, 6, 6, 7)] * 3
    return made


def remark(client: TestClient, attempt_id: str, essay: str = ESSAY):
    return client.post("/ielts/remark", json={"attempt_id": attempt_id, "essay": essay})


def test_a_remark_returns_the_new_band_and_the_change_against_the_attempt(
    client: TestClient, attempt_id: str
):
    response = remark(client, attempt_id)
    assert response.status_code == 200
    out = response.json()
    assert out["status"] == "graded" and out["label"].startswith("Practice re-mark")
    assert "untimed" in out["label"] and "not saved" in out["label"]
    assert out["overall"] > 6.0 and out["overall_delta"] == pytest.approx(out["overall"] - 6.0)
    deltas = {c["id"]: c["delta"] for c in out["criteria"]}
    assert deltas["TR"] > 0 and deltas["CC"] == 0
    assert "mock score" in out["disclaimer"].lower()


def test_a_remark_is_stored_nowhere_and_does_not_touch_the_attempt(
    client: TestClient, attempt_id: str, store: AttemptStore
):
    before = client.get(f"/ielts/attempts/{attempt_id}").json()
    remark(client, attempt_id)
    assert client.get(f"/ielts/attempts/{attempt_id}").json() == before
    assert len(client.get("/ielts/attempts").json()) == 1  # no new mock attempt


def test_a_remark_is_a_costed_call_the_ledger_can_stop(
    client: TestClient, attempt_id: str, ledger: Ledger
):
    ledger.record("earlier", "gpt-4.1-mini", ledger.ceiling_usd - 0.01)  # below one mock's reserve
    refused = remark(client, attempt_id)
    assert refused.status_code == 402
    assert "AI spend cap of $5.00 is used up" in refused.json()["detail"]
    assert "unavailable" not in refused.json()["detail"]


def test_a_cap_reached_while_marking_is_the_cap_message_not_an_outage(
    client: TestClient, attempt_id: str, completer: Metered
):
    completer.grades = [BudgetExceededError("budget guard")] * 3
    refused = remark(client, attempt_id)
    assert refused.status_code == 402 and "spend cap" in refused.json()["detail"]


def test_a_model_outage_is_a_502_that_says_so(
    client: TestClient, attempt_id: str, completer: Metered
):
    from services.gateway.src.gateway.ielts.llm import ModelUnavailableError

    completer.grades = [ModelUnavailableError(GRADER_ROLE, "down")] * 3
    refused = remark(client, attempt_id)
    assert refused.status_code == 502 and "spend cap" not in refused.json()["detail"]


def test_a_remark_counts_against_the_ielts_rate_limit(
    client: TestClient, attempt_id: str, completer: Metered
):
    completer.grades = [grade_reply(7, 6, 6, 7)] * 60
    statuses = [remark(client, attempt_id).status_code for _ in range(12)]
    assert 429 in statuses and statuses[0] == 200
    assert "Too many" in remark(client, attempt_id).json()["detail"]


def test_an_unknown_attempt_an_ungraded_attempt_and_an_empty_essay_are_refused(
    client: TestClient, attempt_id: str
):
    assert remark(client, "nope").status_code == 404
    fresh = client.post(
        "/ielts/attempts", json={"prompt_id": "ielts_writing_prompt_1"}
    ).json()["id"]
    assert remark(client, fresh).status_code == 409  # nothing to compare with
    assert remark(client, attempt_id, "   ").status_code == 422
