"""
The quick sample test: the same Reading, Listening, Writing and full-mock path as the real thing,
on a handful of items and a short clock, always labelled as a sample and kept out of the real
history. Only the item subset and the clock differ, so these tests drive the real routes.
"""
# ruff: noqa: F811 - the imported pytest fixtures are used as test parameters
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.gateway.src.gateway.ielts import listening, reading
from services.gateway.src.gateway.ielts_store import AttemptStore, allowed_s_for
from tests.gateway.test_ielts_mock import (  # noqa: F401 - fixtures
    client,
    clock,
    modules_of,
    sample_prompts,
    start,
    store,
)
from tests.gateway.test_ielts_writing_module import ESSAY_FULL, LETTER, Clock

# the report shows 'in 3 minutes' at most, so every module's sample clock stays short
SAMPLE_TOTAL_CLOCK_S = 2 * 60 + 3 * 60 + 3 * 60


def sample_mock(client: TestClient) -> str:
    resp = client.post("/ielts/mock?sample=true")
    assert resp.status_code == 201
    assert resp.json()["sample"] is True
    return str(resp.json()["id"])


def keys_of(paper) -> dict[str, str]:
    return {q.id: q.key for q in paper.questions}


def test_the_sample_clocks_are_short_and_the_real_clocks_are_unchanged():
    assert allowed_s_for("ielts_reading", True) == 120
    assert allowed_s_for("ielts_reading", False) == 3600
    assert allowed_s_for("ielts_listening", False) == 1800
    assert allowed_s_for("ielts_writing", False) == 3600
    assert SAMPLE_TOTAL_CLOCK_S <= 8 * 60


def test_the_sample_sets_are_a_renumbered_subset_of_the_real_items():
    real_reading, real_listening = reading.get_set(), listening.get_set()
    sample_reading, sample_listening = reading.get_sample_set(), listening.get_sample_set()
    assert [q.number for q in sample_reading.questions] == [1, 2, 3, 4, 5]
    assert {q.id for q in sample_reading.questions} <= {q.id for q in real_reading.questions}
    assert {q.id for q in sample_listening.questions} <= {q.id for q in real_listening.questions}
    assert len(sample_listening.sections) == 1
    assert sample_reading.id != real_reading.id and sample_listening.id != real_listening.id
    assert len({q.task_type for q in sample_reading.questions}) == 4  # every task type


def test_a_sample_reading_attempt_runs_the_real_path_and_is_labelled(client: TestClient):
    created = client.post("/ielts/reading/attempts?sample=true").json()
    assert created["sample"] is True and created["allowed_s"] == 120
    started = client.post(f"/ielts/reading/attempts/{created['id']}/start").json()
    assert started["remaining_s"] == 120
    assert len(started["paper"]["questions"]) == 5
    answers = keys_of(reading.get_sample_set())
    done = client.post(f"/ielts/reading/attempts/{created['id']}/submit", json={"answers": answers})
    assert done.status_code == 200 and done.json()["raw"] == 5
    report = client.get(f"/ielts/reading/attempts/{created['id']}/report").json()
    assert report["sample"] is True and report["total"] == 5 and report["band"] == 9.0
    assert report["set_note"].startswith("SAMPLE")


def test_an_answer_to_a_question_outside_the_sample_is_dropped(client: TestClient):
    created = client.post("/ielts/reading/attempts?sample=true").json()
    client.post(f"/ielts/reading/attempts/{created['id']}/start")
    outside = next(
        q.id for q in reading.get_set().questions
        if q.id not in keys_of(reading.get_sample_set())
    )
    done = client.post(
        f"/ielts/reading/attempts/{created['id']}/submit", json={"answers": {outside: "x"}}
    ).json()
    assert done["answered"] == 0


def test_a_sample_listening_attempt_serves_only_its_recording(client: TestClient):
    created = client.post("/ielts/listening/attempts?sample=true").json()
    assert created["sample"] is True and created["allowed_s"] == 180
    started = client.post(f"/ielts/listening/attempts/{created['id']}/start").json()
    assert [s["id"] for s in started["paper"]["sections"]] == ["s1"]
    assert client.get(f"/ielts/listening/attempts/{created['id']}/audio/s1").status_code == 200
    assert client.get(f"/ielts/listening/attempts/{created['id']}/audio/s2").status_code == 404
    answers = keys_of(listening.get_sample_set())
    client.post(f"/ielts/listening/attempts/{created['id']}/submit", json={"answers": answers})
    report = client.get(f"/ielts/listening/attempts/{created['id']}/report").json()
    assert report["sample"] is True and report["total"] == 5 and report["raw"] == 5
    assert [t["id"] for t in report["transcripts"]] == ["s1"]


def test_a_sample_attempt_is_not_in_the_real_history(client: TestClient):
    client.post("/ielts/reading/attempts?sample=true")
    real = client.post("/ielts/reading/attempts").json()
    client.post(f"/ielts/reading/attempts/{real['id']}/start")
    client.post(f"/ielts/reading/attempts/{client.post('/ielts/reading/attempts?sample=true').json()['id']}/start")
    listed = client.get("/ielts/reading/attempts").json()
    assert [a["id"] for a in listed] == [real["id"]]
    assert real["sample"] is False and real["allowed_s"] == 3600


def test_a_sample_mock_is_not_in_the_real_mock_list(client: TestClient):
    sample_mock(client)
    assert client.get("/ielts/mock").json() == []


def test_the_sample_clock_runs_out_on_the_server(
    client: TestClient, store: AttemptStore, clock: Clock
):
    created = client.post("/ielts/reading/attempts?sample=true").json()
    client.post(f"/ielts/reading/attempts/{created['id']}/start")
    clock.now += 121 + 30  # past the 2 minute sample clock and the grace period
    done = client.post(
        f"/ielts/reading/attempts/{created['id']}/submit",
        json={"answers": keys_of(reading.get_sample_set())},
    ).json()
    assert done["late"] is True and done["raw"] == 0


def test_the_sample_full_mock_runs_end_to_end_to_an_overall_band(client: TestClient):
    mock_id = sample_mock(client)

    body = start(client, mock_id, "listening").json()
    assert body["sample"] is True and body["overall_note"].startswith("SAMPLE")
    lid = modules_of(body)["listening"]["attempt_id"]
    client.post(f"/ielts/listening/attempts/{lid}/start")
    client.post(
        f"/ielts/listening/attempts/{lid}/submit",
        json={"answers": keys_of(listening.get_sample_set())},
    )

    rid = modules_of(start(client, mock_id, "reading").json())["reading"]["attempt_id"]
    client.post(f"/ielts/reading/attempts/{rid}/start")
    client.post(
        f"/ielts/reading/attempts/{rid}/submit",
        json={"answers": keys_of(reading.get_sample_set())},
    )

    wid = modules_of(start(client, mock_id, "writing").json())["writing"]["attempt_id"]
    started = client.post(f"/ielts/writing/attempts/{wid}/start").json()
    assert started["sample"] is True and started["allowed_s"] == 180
    client.post(f"/ielts/writing/attempts/{wid}/submit", json={"t1": LETTER, "t2": ESSAY_FULL})

    body = client.get(f"/ielts/mock/{mock_id}").json()
    assert body["sample"] is True
    assert {k: m["state"] for k, m in modules_of(body).items() if k != "speaking"} == {
        "listening": "graded", "reading": "graded", "writing": "graded",
    }
    assert body["overall"] is not None and body["partial"] is True  # Speaking is not assessed
    assert body["overall_note"].startswith("SAMPLE")
    writing = client.get(f"/ielts/writing/attempts/{wid}/report").json()
    assert writing["sample"] is True


def test_an_old_database_gains_the_sample_columns(tmp_path: Path):
    import sqlite3

    path = tmp_path / "old.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE mocks (id TEXT PRIMARY KEY, created_at REAL NOT NULL)")
    db.execute("INSERT INTO mocks (id, created_at) VALUES ('m1', 1.0)")
    db.commit()
    db.close()
    store = AttemptStore(path)
    assert store.get_mock("m1").sample is False
    assert store.create_mock(sample=True).sample is True


@pytest.mark.parametrize("module", ["ielts_reading", "ielts_listening", "ielts_writing"])
def test_every_module_has_a_sample_clock(module: str):
    assert allowed_s_for(module, True) < allowed_s_for(module, False)
