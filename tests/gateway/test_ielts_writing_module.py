"""
The General Training Writing module: a Task 1 letter marked like an essay (Task Achievement in
place of Task Response), the 1:2 combined band, and the /ielts/writing API (one 60 minute clock,
two texts, two reports, one band). A scripted completer replaces the model.
"""
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.gateway.src.gateway import ielts_api
from services.gateway.src.gateway.ielts import llm
from services.gateway.src.gateway.ielts.digest import build_digest
from services.gateway.src.gateway.ielts.marking import mark_essay
from services.gateway.src.gateway.ielts.prompts import load_letter_prompts
from services.gateway.src.gateway.ielts.report_checks import check_report
from services.gateway.src.gateway.ielts.scoring import (
    CRITERIA_BY_TASK,
    TASK_SOURCE_URLS,
    writing_band,
)
from services.gateway.src.gateway.ielts_store import GRACE_S, WRITING_ALLOWED_S, AttemptStore
from services.gateway.src.gateway.limits import ielts_limiter
from services.gateway.src.gateway.main import app
from shared.src.shared.budget import Ledger
from tests.gateway.test_ielts_marking import ESSAY, feedback_reply, grade_reply

LETTER_PROMPT = (
    "You live in a flat and the lift has been out of order for three weeks. Write a letter to "
    "the building manager. In your letter:\n- describe the problem\n- say what you have done\n"
    "- say what you want the manager to do"
)
LETTER = (
    "Dear Mr Hall,\n\n"
    "I am writing to complain about the lift in our building, which has not worked for three "
    "weeks. I live on the eighth floor and my mother, who is seventy, can no longer visit me. "
    "Carrying heavy shopping up so many stairs is also very tiring after a long day at work.\n\n"
    "Last week I phoned the maintenance office twice and left a message each time, but nobody "
    "has called me back. I also spoke to two of my neighbours, and they have the same problem "
    "and are just as worried about it as I am.\n\n"
    "I would be grateful if you could send an engineer this week and tell us when the repair "
    "will be finished. If that is not possible, please arrange some help for older residents.\n\n"
    "Many of the older people here stay indoors all day because they cannot manage the stairs, "
    "and that is not fair to them.\n\n"
    "I look forward to hearing from you soon.\n\n"
    "Yours sincerely,\nAnna Lee"
)


ESSAY_FULL = ESSAY + (
    "\n\nSome people say that schools have too many subjects already. However, cooking teaches "
    "maths, health and teamwork together, so it saves time in the long run, and every pupil "
    "who learns it will cook at home for the rest of their life."
)


def letter_grade(ta: float, cc: float, lr: float, gra: float) -> str:
    return json.dumps({"TA": ta, "CC": cc, "LR": lr, "GRA": gra, "rationale": "r"})


def letter_feedback() -> str:
    quotes = {
        "TA": "I am writing to complain about the lift",
        "CC": "Last week I phoned the maintenance office",
        "LR": "I would be grateful if you could send an engineer",
        "GRA": "I look forward to hearing from you soon",
    }
    return json.dumps({
        "criteria": {
            c: {
                "verdict": f"{c} verdict", "evidence": [{"quote": q, "effect": "supports"}],
                "holding_back": "h", "to_reach_next_band": "n",
                "actions": [{"do": f"fix {c}", "why": "because"}],
            }
            for c, q in quotes.items()
        },
        "errors": [],
        "checks": {
            "covers_every_bullet": True, "tone_fits_reader": True,
            "clear_opening_and_closing": False,
        },
    })


class TaskAware:
    """Replies per task: the user message names the candidate's work a letter or an essay."""

    def __init__(self, t1: tuple[float, ...], t2: tuple[float, ...]) -> None:
        self.t1, self.t2 = t1, t2
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, role: str, messages: list[dict[str, str]], **kw: object) -> str:
        letter = "Candidate letter" in messages[-1]["content"]
        self.calls.append((role, "T1" if letter else "T2"))
        if role == llm.GRADER_ROLE:
            return letter_grade(*self.t1) if letter else grade_reply(*self.t2)
        return letter_feedback() if letter else feedback_reply()


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
def completer() -> TaskAware:
    return TaskAware((6, 6, 6, 6), (7, 7, 7, 7))


@pytest.fixture
def client(store: AttemptStore, completer: TaskAware, tmp_path: Path) -> Iterator[TestClient]:
    ledger = Ledger(tmp_path / "ledger.json", ceiling_usd=5.0)
    app.dependency_overrides[ielts_api.get_store] = lambda: store
    app.dependency_overrides[ielts_api.get_completer] = lambda: completer
    app.dependency_overrides[ielts_api.get_ledger] = lambda: ledger
    ielts_limiter().reset()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── the combined band ────────────────────────────────────────────────────────


@pytest.mark.parametrize(("t1", "t2", "expected"), [
    (6.0, 7.0, 6.5),  # (6 + 14) / 3 = 6.67
    (7.0, 6.0, 6.5),  # (7 + 12) / 3 = 6.33
    (5.0, 7.0, 6.5),  # (5 + 14) / 3 = 6.33
    (6.5, 6.5, 6.5),
    (4.0, 8.0, 6.5),  # (4 + 16) / 3 = 6.67
    (5.5, 6.5, 6.0),  # (5.5 + 13) / 3 = 6.17
    (8.0, 5.0, 6.0),  # (8 + 10) / 3 = 6.0
])
def test_task_2_counts_twice_as_much_as_task_1(t1: float, t2: float, expected: float):
    assert writing_band({"T1": t1, "T2": t2}) == expected


def test_the_weighting_cites_the_public_ielts_pages():
    assert all(u.startswith("https://ielts.org/") for u in TASK_SOURCE_URLS)


# ── the letter ───────────────────────────────────────────────────────────────


def test_there_are_original_letter_prompts_with_three_bullets():
    prompts = load_letter_prompts()
    assert len(prompts) >= 6 and len({p.id for p in prompts}) == len(prompts)
    assert {p.register for p in prompts} == {"formal", "semi_formal", "personal"}
    assert all(p.text.count("\n- ") == 3 for p in prompts)


async def test_a_letter_is_marked_on_task_achievement_with_the_letter_checks():
    completer = TaskAware((6, 6.5, 6, 6), (7, 7, 7, 7))
    report = await mark_essay(
        LETTER_PROMPT, LETTER, completer=completer, samples=1, trace_id="t", task="T1"
    )
    assert report.status == "graded" and report.task == "T1"
    assert [c.id for c in report.criteria] == list(CRITERIA_BY_TASK["T1"])
    assert report.criteria[0].name == "Task Achievement"
    assert report.estimated_band is not None and report.estimated_band.overall == 6.0
    assert check_report(report, LETTER) == []
    checks = {t.check: t.passed for t in report.task_checks}
    assert checks["at least 150 words"] is True
    assert checks["covers every bullet point of the task"] is True
    assert checks["clear opening and closing"] is False
    assert report.prompt_versions is not None
    digest = build_digest(report, LETTER, LETTER_PROMPT)
    assert "Task 1 (letter)" in digest and "Task Achievement" in digest


async def test_a_letter_under_100_words_is_capped_like_a_short_essay():
    short = "Dear Sir,\n\n" + " ".join(["The lift is broken and I need help now."] * 11)
    report = await mark_essay(
        LETTER_PROMPT, short, completer=TaskAware((8, 8, 8, 8), (7, 7, 7, 7)), samples=1,
        trace_id="t", task="T1",
    )
    assert report.status == "graded"
    ta = report.criteria[0]
    assert ta.id == "TA" and ta.band <= 5.0 and ta.capped_from == 8.0


async def test_task_2_is_marked_exactly_as_before():
    report = await mark_essay(
        "p", ESSAY, completer=TaskAware((6, 6, 6, 6), (7, 7, 7, 7)), samples=1, trace_id="t"
    )
    assert report.task == "T2" and [c.id for c in report.criteria][0] == "TR"


# ── the API ──────────────────────────────────────────────────────────────────


def new_attempt(client: TestClient) -> str:
    # the prompts the sample letter and essay answer (omitted ids pick a random prompt)
    resp = client.post("/ielts/writing/attempts", json={
        "t1_prompt_id": "ielts_letter_prompt_1", "t2_prompt_id": "ielts_writing_prompt_1",
    })
    assert resp.status_code == 201
    return str(resp.json()["id"])


def test_full_module_flow_one_clock_two_reports_one_band(
    client: TestClient, clock: Clock, completer: TaskAware
):
    attempt_id = new_attempt(client)
    created = client.get(f"/ielts/writing/attempts/{attempt_id}").json()
    assert created["status"] == "created" and created["remaining_s"] is None
    assert "\n- " in created["t1"]["prompt"] and created["t2"]["prompt"]

    started = client.post(f"/ielts/writing/attempts/{attempt_id}/start").json()
    assert started["status"] == "in_progress"
    assert started["remaining_s"] == WRITING_ALLOWED_S == 3600

    clock.now += 600
    saved = client.put(
        f"/ielts/writing/attempts/{attempt_id}/draft", json={"t1": "Dear Mr Hall", "t2": ""}
    )
    assert saved.status_code == 200 and saved.json()["t1"]["text"] == "Dear Mr Hall"

    clock.now += 1800  # refresh: the server's clock says what is left on the one 60 minute clock
    resumed = client.get(f"/ielts/writing/attempts/{attempt_id}").json()
    assert resumed["remaining_s"] == 3600 - 2400 and resumed["t1"]["word_count"] == 3

    done = client.post(
        f"/ielts/writing/attempts/{attempt_id}/submit", json={"t1": LETTER, "t2": ESSAY_FULL}
    )
    assert done.status_code == 200
    final = client.get(f"/ielts/writing/attempts/{attempt_id}").json()
    assert final["status"] == "graded" and not final["late"]
    assert final["t1"]["band"] == 6.0 and final["t2"]["band"] == 7.0
    assert final["writing_band"] == 6.5  # (6 + 2 x 7) / 3 = 6.67

    report = client.get(f"/ielts/writing/attempts/{attempt_id}/report").json()
    assert report["status"] == "graded" and report["writing_band"] == 6.5
    assert report["t1"]["task"] == "T1" and report["t2"]["task"] == "T2"
    assert report["t1"]["criteria"][0]["id"] == "TA" and report["t2"]["criteria"][0]["id"] == "TR"
    assert report["timing"]["used_s"] == 2400 and report["timing"]["allowed_s"] == 3600
    assert "twice as much" in report["weights_note"]
    assert report["disclaimer"] == "Mock score: an estimate, not an official IELTS result."
    assert report["t1"]["criteria"][0]["evidence"][0]["quote"] in LETTER
    assert {t for _, t in completer.calls} == {"T1", "T2"}


def test_a_missing_letter_gives_a_partial_report_and_no_combined_band(client: TestClient):
    attempt_id = new_attempt(client)
    client.post(f"/ielts/writing/attempts/{attempt_id}/start")
    client.post(f"/ielts/writing/attempts/{attempt_id}/submit", json={"t1": "", "t2": ESSAY_FULL})
    report = client.get(f"/ielts/writing/attempts/{attempt_id}/report").json()
    assert report["status"] == "partial" and report["writing_band"] is None
    assert report["t1"]["status"] == "ungraded" and report["t1"]["ungraded_reason"]
    assert report["t2"]["status"] == "graded"
    view = client.get(f"/ielts/writing/attempts/{attempt_id}").json()
    assert view["status"] == "ungraded" and view["t2"]["band"] == 7.0


def test_a_late_module_submission_marks_the_last_autosave(client: TestClient, clock: Clock):
    attempt_id = new_attempt(client)
    client.post(f"/ielts/writing/attempts/{attempt_id}/start")
    client.put(f"/ielts/writing/attempts/{attempt_id}/draft", json={"t1": LETTER, "t2": ESSAY_FULL})
    clock.now += WRITING_ALLOWED_S + GRACE_S + 1
    client.post(
        f"/ielts/writing/attempts/{attempt_id}/submit", json={"t1": "x", "t2": "written after time"}
    )
    view = client.get(f"/ielts/writing/attempts/{attempt_id}").json()
    assert view["late"] is True
    assert view["t1"]["text"] == LETTER and view["t2"]["text"] == ESSAY_FULL


def test_a_model_outage_keeps_both_texts_and_regrade_recovers(
    client: TestClient, completer: TaskAware
):
    async def down(role: str, messages: list[dict[str, str]], **kw: object) -> str:
        raise llm.ModelUnavailableError("down")

    app.dependency_overrides[ielts_api.get_completer] = lambda: down
    attempt_id = new_attempt(client)
    client.post(f"/ielts/writing/attempts/{attempt_id}/start")
    both = {"t1": LETTER, "t2": ESSAY_FULL}
    client.post(f"/ielts/writing/attempts/{attempt_id}/submit", json=both)
    failed = client.get(f"/ielts/writing/attempts/{attempt_id}").json()
    assert failed["status"] == "ungraded" and failed["retryable"] is True
    assert failed["t1"]["text"] == LETTER and failed["t2"]["text"] == ESSAY_FULL

    app.dependency_overrides[ielts_api.get_completer] = lambda: completer
    assert client.post(f"/ielts/writing/attempts/{attempt_id}/regrade").status_code == 200
    assert client.get(f"/ielts/writing/attempts/{attempt_id}").json()["writing_band"] == 6.5


def test_modules_do_not_see_each_others_attempts(client: TestClient):
    module_id = new_attempt(client)
    essay = client.post("/ielts/attempts", json={"prompt_id": "ielts_writing_prompt_1"}).json()
    assert client.get(f"/ielts/attempts/{module_id}").status_code == 404
    assert client.get(f"/ielts/writing/attempts/{essay['id']}").status_code == 404
    client.post(f"/ielts/attempts/{essay['id']}/start")
    assert [a["id"] for a in client.get("/ielts/attempts").json()] == [essay["id"]]
    assert [a["id"] for a in client.get("/ielts/writing/attempts").json()] == []  # not started


def test_the_writing_coach_reads_a_graded_module_with_both_tasks(
    client: TestClient, store: AttemptStore
):
    from services.gateway.src.gateway.ielts_coach import find_report

    attempt_id = new_attempt(client)
    client.post(f"/ielts/writing/attempts/{attempt_id}/start")
    both = {"t1": LETTER, "t2": ESSAY_FULL}
    client.post(f"/ielts/writing/attempts/{attempt_id}/submit", json=both)
    for found in (find_report(store, attempt_id), find_report(store, None)):
        assert found is not None and found.attempt_id == attempt_id
        assert "Combined Writing band (mock): 6.5" in found.digest
        assert "Task 1 (letter)" in found.digest and "Task 2" in found.digest
        assert LETTER not in found.digest and len(found.digest) <= 8_000
    assert find_report(store, attempt_id, "reading") is None
