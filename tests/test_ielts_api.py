"""
The /ielts API and attempt store: server-authoritative clock, autosave, resume, late submit,
background marking and retry. A scripted completer replaces the model.
"""
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.agent.src.agent.ielts import llm
from services.agent.src.agent.ielts.checks import words
from services.gateway.src.gateway import ielts_api
from services.gateway.src.gateway.ielts_store import (
    ALLOWED_S,
    GRACE_S,
    AttemptNotFoundError,
    AttemptStateError,
    AttemptStore,
)
from services.gateway.src.gateway.main import app
from shared.src.shared.budget import Ledger
from tests.test_ielts_marking import ESSAY, Scripted, feedback_reply, grade_reply


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path: Path, clock: Clock) -> AttemptStore:
    return AttemptStore(tmp_path / "ielts-local" / "attempts.sqlite3", clock)


@pytest.fixture
def completer() -> Scripted:
    return Scripted([grade_reply(6, 6, 6, 6)] * 30)


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.json", ceiling_usd=5.0)


@pytest.fixture
def client(store: AttemptStore, completer: Scripted, ledger: Ledger) -> Iterator[TestClient]:
    app.dependency_overrides[ielts_api.get_store] = lambda: store
    app.dependency_overrides[ielts_api.get_completer] = lambda: completer
    app.dependency_overrides[ielts_api.get_ledger] = lambda: ledger
    ielts_api._limiter._minute_window.clear()
    ielts_api._limiter._day_counts.clear()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def new_attempt(client: TestClient, prompt_id: str = "ielts_writing_prompt_1") -> str:
    resp = client.post("/ielts/attempts", json={"prompt_id": prompt_id})
    assert resp.status_code == 201
    return str(resp.json()["id"])


# ── store ────────────────────────────────────────────────────────────────────


def test_start_fixes_the_deadline_and_is_idempotent(store: AttemptStore, clock: Clock):
    attempt = store.create("p1", "Prompt text")
    assert attempt.status == "created" and attempt.deadline_at is None
    started = store.start(attempt.id)
    assert started.deadline_at == clock.now + ALLOWED_S
    clock.now += 100
    assert store.start(attempt.id).deadline_at == started.deadline_at  # clock never restarts


def test_autosave_works_until_deadline_plus_grace(store: AttemptStore, clock: Clock):
    attempt = store.start(store.create("p1", "P").id)
    clock.now += ALLOWED_S + GRACE_S
    assert store.save_draft(attempt.id, "last words").draft_text == "last words"
    clock.now += 1
    with pytest.raises(AttemptStateError, match="time is up"):
        store.save_draft(attempt.id, "too late")
    assert store.get(attempt.id).draft_text == "last words"


def test_submit_is_idempotent_and_late_after_the_grace_window(store: AttemptStore, clock: Clock):
    on_time = store.start(store.create("p1", "P").id)
    clock.now += ALLOWED_S + GRACE_S
    done = store.submit(on_time.id, "essay")
    assert not done.late and done.final_text == "essay" and done.status == "submitted"
    assert store.submit(on_time.id, "changed").final_text == "essay"

    late = store.start(store.create("p1", "P").id)
    store.save_draft(late.id, "saved words")
    clock.now += ALLOWED_S + GRACE_S + 1
    submitted = store.submit(late.id, "saved words and words written after time was up")
    assert submitted.late and submitted.final_text == "saved words"


def test_a_created_attempt_cannot_be_saved_or_submitted(store: AttemptStore):
    attempt = store.create("p1", "P")
    with pytest.raises(AttemptStateError):
        store.save_draft(attempt.id, "x")
    with pytest.raises(AttemptStateError):
        store.submit(attempt.id, "x")
    with pytest.raises(AttemptNotFoundError):
        store.get("missing")


def test_the_database_lives_where_it_is_told_to(store: AttemptStore, tmp_path: Path):
    store.create("p1", "P")
    assert (tmp_path / "ielts-local" / "attempts.sqlite3").exists()


# ── API ──────────────────────────────────────────────────────────────────────


def test_prompts_are_the_eight_original_task_2_prompts(client: TestClient):
    prompts = client.get("/ielts/prompts").json()
    assert len(prompts) == 8 and {p["essay_type"] for p in prompts} == {
        "opinion", "discuss_both_views", "advantages_disadvantages", "problem_solution"}
    assert all(p["text"] and p["essay_type_label"] for p in prompts)


def test_unknown_prompt_and_attempt_are_404(client: TestClient):
    assert client.post("/ielts/attempts", json={"prompt_id": "nope"}).status_code == 404
    assert client.get("/ielts/attempts/none").status_code == 404


def test_full_flow_start_autosave_resume_submit_report(
    client: TestClient, clock: Clock, completer: Scripted
):
    completer.feedback = feedback_reply()
    attempt_id = new_attempt(client)
    created = client.get(f"/ielts/attempts/{attempt_id}").json()
    assert created["status"] == "created" and created["remaining_s"] is None

    started = client.post(f"/ielts/attempts/{attempt_id}/start").json()
    assert started["status"] == "in_progress" and started["remaining_s"] == ALLOWED_S

    clock.now += 600
    saved = client.put(f"/ielts/attempts/{attempt_id}/draft", json={"text": "partial essay"})
    assert saved.status_code == 200 and saved.json()["word_count"] == 2

    clock.now += 300  # a refresh: the server's clock, not the browser's, says what is left
    resumed = client.get(f"/ielts/attempts/{attempt_id}").json()
    assert resumed["draft"] == "partial essay" and resumed["remaining_s"] == ALLOWED_S - 900

    done = client.post(f"/ielts/attempts/{attempt_id}/submit", json={"text": ESSAY})
    assert done.status_code == 200
    final = client.get(f"/ielts/attempts/{attempt_id}").json()
    assert final["status"] == "graded" and final["overall"] == 6.0 and not final["late"]
    report = client.get(f"/ielts/attempts/{attempt_id}/report").json()
    assert report["status"] == "graded" and report["timing"]["used_s"] == 900
    assert report["estimated_band"]["calibration"]["label"] == "Uncalibrated estimate"
    assert report["disclaimer"] == "Mock score: an estimate, not an official IELTS result."
    assert report["criteria"][0]["evidence"][0]["quote"] in ESSAY


def test_a_late_submission_marks_the_last_autosave_and_is_reported_as_late(
    client: TestClient, clock: Clock
):
    attempt_id = new_attempt(client)
    client.post(f"/ielts/attempts/{attempt_id}/start")
    client.put(f"/ielts/attempts/{attempt_id}/draft", json={"text": ESSAY})
    clock.now += ALLOWED_S + GRACE_S + 5
    assert client.put(f"/ielts/attempts/{attempt_id}/draft", json={"text": "x"}).status_code == 409
    late_text = ESSAY + "\n\nA paragraph written after time was up."
    submitted = client.post(f"/ielts/attempts/{attempt_id}/submit", json={"text": late_text})
    assert submitted.json()["draft"] == ESSAY
    report = client.get(f"/ielts/attempts/{attempt_id}/report").json()
    assert report["timing"]["late"] is True
    assert report["word_count"] == len(words(ESSAY))


def test_submitting_twice_marks_once(client: TestClient, completer: Scripted):
    attempt_id = new_attempt(client)
    client.post(f"/ielts/attempts/{attempt_id}/start")
    client.post(f"/ielts/attempts/{attempt_id}/submit", json={"text": ESSAY})
    calls = len(completer.calls)
    again = client.post(f"/ielts/attempts/{attempt_id}/submit", json={"text": "different"})
    assert again.status_code == 200 and len(completer.calls) == calls
    assert again.json()["draft"] == ESSAY


def test_model_outage_keeps_the_essay_and_regrade_recovers(client: TestClient, completer: Scripted):
    completer.grades = [llm.ModelUnavailableError("down")] * 3
    attempt_id = new_attempt(client)
    client.post(f"/ielts/attempts/{attempt_id}/start")
    client.post(f"/ielts/attempts/{attempt_id}/submit", json={"text": ESSAY})
    failed = client.get(f"/ielts/attempts/{attempt_id}").json()
    assert failed["status"] == "ungraded" and failed["retryable"] and failed["overall"] is None
    assert "unavailable" in failed["error"] and failed["draft"] == ESSAY
    assert client.get(f"/ielts/attempts/{attempt_id}/report").status_code == 409

    completer.grades = [grade_reply(6.5, 6.5, 6.5, 6.5)] * 3
    assert client.post(f"/ielts/attempts/{attempt_id}/regrade").status_code == 200
    assert client.get(f"/ielts/attempts/{attempt_id}").json()["overall"] == 6.5
    assert client.post(f"/ielts/attempts/{attempt_id}/regrade").status_code == 409


def test_an_unmarkable_essay_is_ungraded_with_a_reason_and_no_retry(client: TestClient):
    attempt_id = new_attempt(client)
    client.post(f"/ielts/attempts/{attempt_id}/start")
    client.post(f"/ielts/attempts/{attempt_id}/submit", json={"text": "Cooking is good."})
    view = client.get(f"/ielts/attempts/{attempt_id}").json()
    assert view["status"] == "ungraded" and not view["retryable"]
    report = client.get(f"/ielts/attempts/{attempt_id}/report").json()
    assert report["status"] == "ungraded" and "too short" in report["ungraded_reason"]
    assert report["estimated_band"] is None


def test_marking_interrupted_by_a_restart_is_offered_as_a_retry(
    client: TestClient, store: AttemptStore
):
    attempt_id = new_attempt(client)
    client.post(f"/ielts/attempts/{attempt_id}/start")
    store.submit(attempt_id, ESSAY)
    store.begin_grading(attempt_id)  # the process died here: nothing is in flight
    view = client.get(f"/ielts/attempts/{attempt_id}").json()
    assert view["status"] == "ungraded" and view["retryable"] and "interrupted" in view["error"]


def test_second_attempt_reports_the_change_since_the_first(
    client: TestClient, completer: Scripted, clock: Clock
):
    first = new_attempt(client)
    client.post(f"/ielts/attempts/{first}/start")
    client.post(f"/ielts/attempts/{first}/submit", json={"text": ESSAY})
    completer.grades = [grade_reply(7, 7, 7, 7)] * 3
    clock.now += 3600
    second = new_attempt(client)
    client.post(f"/ielts/attempts/{second}/start")
    client.post(f"/ielts/attempts/{second}/submit", json={"text": ESSAY})
    report = client.get(f"/ielts/attempts/{second}/report").json()
    assert report["vs_previous_attempt"] == {
        "overall_delta": 1.0, "criteria_deltas": {"TR": 1.0, "CC": 1.0, "LR": 1.0, "GRA": 1.0}}
    history = client.get("/ielts/attempts").json()
    assert [h["id"] for h in history] == [second, first]


def test_essay_text_is_capped_in_size(client: TestClient):
    attempt_id = new_attempt(client)
    client.post(f"/ielts/attempts/{attempt_id}/start")
    big = {"text": "x" * 20_001}
    assert client.put(f"/ielts/attempts/{attempt_id}/draft", json=big).status_code == 422


def test_rate_limit_stops_a_flood_of_new_attempts(client: TestClient):
    body = {"prompt_id": "ielts_writing_prompt_1"}
    codes = [client.post("/ielts/attempts", json=body).status_code for _ in range(12)]
    assert codes[:10] == [201] * 10 and 429 in codes[10:]


def test_chat_endpoint_is_untouched(client: TestClient):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert "/chat" in {r.path for r in app.routes}  # type: ignore[attr-defined]
    json.dumps(client.get("/openapi.json").json())
