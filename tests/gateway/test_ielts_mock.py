"""
The full IELTS mock: an ordered list of modules over the existing Reading and Writing attempts, a
combined overall band computed in code (partial while Listening and Speaking are not assessed),
a next-module recommendation, and the coach's digest and /chat path. A scripted completer replaces
the marking model; the chat gates that call a model are mocked.
"""
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from services.gateway.src.gateway import ielts_api
from services.gateway.src.gateway.guardrails.verdict import Verdict
from services.gateway.src.gateway.ielts.mock import SCORING_SOURCE_URL, overall_band
from services.gateway.src.gateway.ielts_coach import (
    MOCK_NO_REPORT_NOTICE,
    MOCK_SOLVE_MODE_NOTICE,
    MOCK_UNKNOWN_ATTEMPT_NOTICE,
)
from services.gateway.src.gateway.ielts_store import AttemptStore
from services.gateway.src.gateway.limits import ielts_limiter
from services.gateway.src.gateway.main import app
from shared.src.shared.budget import Ledger
from shared.src.shared.stats import round_half_band
from shared.src.shared.wire import MAX_REPORT_DIGEST_CHARS, GenerateRequest
from tests.gateway.gateway_fakes import agent_replies
from tests.gateway.test_ielts_reading import wrong_on
from tests.gateway.test_ielts_writing_module import ESSAY_FULL, LETTER, Clock, TaskAware

GATEWAY = "services.gateway.src.gateway.chat.gates"


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path: Path, clock: Clock) -> AttemptStore:
    return AttemptStore(tmp_path / "ielts-local" / "attempts.sqlite3", clock)


@pytest.fixture
def client(store: AttemptStore, tmp_path: Path) -> Iterator[TestClient]:
    ledger = Ledger(tmp_path / "ledger.json", ceiling_usd=5.0)
    app.dependency_overrides[ielts_api.get_store] = lambda: store
    completer = TaskAware((6, 6, 6, 6), (7, 7, 7, 7))
    app.dependency_overrides[ielts_api.get_completer] = lambda: completer
    app.dependency_overrides[ielts_api.get_ledger] = lambda: ledger
    ielts_limiter().reset()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def sample_prompts() -> Iterator[None]:
    """The mock starts Writing with random prompts; the sample letter and essay answer these."""
    wanted = {"ielts_letter_prompt_1", "ielts_writing_prompt_1"}

    def pick(items):
        return next((i for i in items if i.id in wanted), items[0])

    with patch("services.gateway.src.gateway.ielts_writing_api.random.choice", pick):
        yield


def modules_of(body: dict) -> dict[str, dict]:
    return {m["key"]: m for m in body["modules"]}


def new_mock(client: TestClient) -> str:
    resp = client.post("/ielts/mock")
    assert resp.status_code == 201
    return str(resp.json()["id"])


def start(client: TestClient, mock_id: str, key: str):
    return client.post(f"/ielts/mock/{mock_id}/modules/{key}/attempt")


def sit_reading(client: TestClient, mock_id: str) -> dict:
    body = start(client, mock_id, "reading").json()
    attempt_id = modules_of(body)["reading"]["attempt_id"]
    client.post(f"/ielts/reading/attempts/{attempt_id}/start")
    answers = wrong_on("matching_headings")
    done = client.post(f"/ielts/reading/attempts/{attempt_id}/submit", json={"answers": answers})
    assert done.status_code == 200
    return client.get(f"/ielts/mock/{mock_id}").json()


def sit_writing(client: TestClient, mock_id: str) -> dict:
    body = start(client, mock_id, "writing").json()
    attempt_id = modules_of(body)["writing"]["attempt_id"]
    client.post(f"/ielts/writing/attempts/{attempt_id}/start")
    done = client.post(
        f"/ielts/writing/attempts/{attempt_id}/submit", json={"t1": LETTER, "t2": ESSAY_FULL}
    )
    assert done.status_code == 200
    return client.get(f"/ielts/mock/{mock_id}").json()


# the combined band


@pytest.mark.parametrize(
    ("bands", "expected"),
    [
        ([6.0, 6.5], 6.5),  # mean 6.25 rounds up to the next half band
        ([6.5, 7.0], 7.0),  # mean 6.75 rounds up to the next whole band
        ([6.0, 6.0, 6.5], 6.0),  # mean 6.17 rounds to the nearest half band, 6.0
        ([7.0], 7.0),
        ([], None),
    ],
)
def test_the_overall_band_is_the_mean_of_the_assessed_bands_rounded_the_ielts_way(bands, expected):
    assert overall_band(bands) == expected


def test_there_is_one_shared_rounding_rule():
    assert overall_band([6.0, 6.5]) == round_half_band(6.25) == 6.5


# the flow


def test_a_new_mock_lists_the_modules_in_order_and_assesses_none(client: TestClient):
    body = client.get(f"/ielts/mock/{new_mock(client)}").json()
    assert [m["key"] for m in body["modules"]] == ["listening", "reading", "writing", "speaking"]
    by_key = modules_of(body)
    assert by_key["listening"]["state"] == by_key["speaking"]["state"] == "not_assessed"
    assert by_key["reading"]["state"] == by_key["writing"]["state"] == "not_started"
    assert body["overall"] is None and body["partial"] is True
    assert body["next_module"]["key"] == "reading"


def test_writing_cannot_start_before_reading_is_sat(client: TestClient):
    mock_id = new_mock(client)
    resp = start(client, mock_id, "writing")
    assert resp.status_code == 409 and "Reading" in resp.json()["detail"]
    start(client, mock_id, "reading")  # created, not yet sat
    assert start(client, mock_id, "writing").status_code == 409
    assert start(client, mock_id, "reading").status_code == 409  # one attempt per module


def test_listening_and_speaking_cannot_be_started_yet(client: TestClient):
    mock_id = new_mock(client)
    assert start(client, mock_id, "listening").status_code == 409
    assert start(client, mock_id, "speaking").status_code == 409


def test_an_unknown_mock_is_not_found(client: TestClient):
    assert client.get("/ielts/mock/nope").status_code == 404
    assert start(client, "nope", "reading").status_code == 404


def test_reading_then_writing_gives_each_band_and_a_partial_overall(client: TestClient):
    mock_id = new_mock(client)

    after_reading = sit_reading(client, mock_id)
    reading = modules_of(after_reading)["reading"]
    reading_band = reading["band"]
    assert reading["state"] == "graded" and reading_band is not None
    assert after_reading["overall"] == reading_band and after_reading["partial"] is True
    assert after_reading["next_module"]["key"] == "writing"

    final = sit_writing(client, mock_id)
    by_key = modules_of(final)
    assert by_key["writing"]["state"] == "graded" and by_key["writing"]["band"] == 6.5
    assert by_key["listening"]["band"] is None and by_key["speaking"]["band"] is None
    assert final["overall"] == round_half_band((reading_band + 6.5) / 2)
    assert final["partial"] is True
    assert "Listening and Speaking not assessed" in final["overall_note"]
    assert final["overall_source_url"] == SCORING_SOURCE_URL
    # everything sittable is done: the program names the lowest assessed skill
    lowest = min(("reading", reading_band), ("writing", 6.5), key=lambda kv: kv[1])[0]
    assert final["next_module"]["key"] == lowest


def test_the_module_pages_still_see_the_mock_attempts(client: TestClient):
    mock_id = new_mock(client)
    sit_reading(client, mock_id)
    assert len(client.get("/ielts/reading/attempts").json()) == 1


def test_mocks_are_listed_newest_first(client: TestClient, clock: Clock):
    first = new_mock(client)
    clock.now += 10
    second = new_mock(client)
    assert [m["id"] for m in client.get("/ielts/mock").json()] == [second, first]


# the coach


@pytest.fixture(autouse=True)
def gates():
    with (
        patch(f"{GATEWAY}.check_content_safety", new_callable=AsyncMock) as safety,
        patch(f"{GATEWAY}.check_completeness", new_callable=AsyncMock) as completeness,
        patch(
            "services.gateway.src.gateway.guardrails.output._call_leak_checker",
            new_callable=AsyncMock, return_value=False,
        ),
    ):
        safety.return_value = Verdict.allow()
        completeness.return_value = Verdict.allow()
        yield safety


def chat(client: TestClient, body: dict | None = None):
    seen: list[GenerateRequest] = []

    async def agent(payload, meta=None):
        seen.append(payload)
        yield b"Practise Writing next."

    with agent_replies(agent):
        resp = client.post("/chat", json={
            "candidate_id": "c1", "test_type": "ielts", "section": "mock",
            "message": "which skill should I practise next?", **(body or {}),
        })
    return resp, seen


def test_the_coach_gets_the_whole_mock_digest_with_the_partial_overall_and_next_module(
    client: TestClient,
):
    mock_id = new_mock(client)
    sit_reading(client, mock_id)
    final = sit_writing(client, mock_id)
    resp, seen = chat(client, {"attempt_id": mock_id})
    assert resp.text == "Practise Writing next."
    digest = seen[0].report_digest
    assert digest and len(digest) <= MAX_REPORT_DIGEST_CHARS and "<<<REPORT" in digest
    assert "Listening: not assessed" in digest and "Speaking: not assessed" in digest
    assert f"Overall band: {final['overall']} (PARTIAL" in digest
    recommended = final["next_module"]["label"]
    assert f"Recommended next module, worked out by the program: {recommended}" in digest
    assert "Reading by question type, weakest first: Matching headings" in digest
    assert "Task 1 letter: band 6.0" in digest and "Task 2 essay: band 7.0" in digest
    assert seen[0].attempt_id == mock_id
    assert LETTER not in digest and ESSAY_FULL not in digest


def test_the_latest_mock_with_a_band_is_coached_when_none_is_named(client: TestClient):
    older = new_mock(client)
    sit_reading(client, older)
    new_mock(client)  # newer, but nothing assessed in it
    _, seen = chat(client)
    assert seen and seen[0].attempt_id == older


def test_a_mock_with_no_band_is_answered_with_a_notice_and_no_agent_call(client: TestClient):
    new_mock(client)
    resp, seen = chat(client)
    assert resp.text == MOCK_NO_REPORT_NOTICE and seen == []


def test_an_unknown_mock_is_answered_with_a_notice(client: TestClient):
    resp, seen = chat(client, {"attempt_id": "nope"})
    assert resp.text == MOCK_UNKNOWN_ATTEMPT_NOTICE and seen == []


def test_solve_mode_is_answered_with_a_clarification(client: TestClient):
    sit_reading(client, new_mock(client))
    resp, seen = chat(client, {"mode": "solve"})
    assert resp.text == MOCK_SOLVE_MODE_NOTICE and seen == []
