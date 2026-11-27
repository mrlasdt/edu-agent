"""
The IELTS Listening mock: the committed scripts and audio, marking in code, the raw-to-band table,
the /ielts/listening API (server clock, no keys before submit, audio only once started, Range
requests) and the store's upgrade of an older mocks table. No model and no text-to-speech call.
"""
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts import build_listening_audio
from services.gateway.src.gateway import ielts_api
from services.gateway.src.gateway.ielts import listening
from services.gateway.src.gateway.ielts.models import ListeningReport, Timing
from services.gateway.src.gateway.ielts.reading import band_for, normalise
from services.gateway.src.gateway.ielts_store import (
    GRACE_S,
    LISTENING_ALLOWED_S,
    AttemptStore,
)
from services.gateway.src.gateway.limits import ielts_limiter
from services.gateway.src.gateway.main import app

PAPER = listening.get_set()
TIMING = Timing(allowed_s=LISTENING_ALLOWED_S, used_s=1500, late=False)


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path: Path, clock: Clock) -> Iterator[AttemptStore]:
    store = AttemptStore(tmp_path / "attempts.sqlite3", clock)
    app.dependency_overrides[ielts_api.get_store] = lambda: store
    ielts_limiter().reset()
    yield store
    app.dependency_overrides.pop(ielts_api.get_store, None)


@pytest.fixture
def client(store: AttemptStore) -> TestClient:
    return TestClient(app)


def all_correct() -> dict[str, str]:
    return {q.id: q.key for q in PAPER.questions}


def correct_except(wrong: int) -> dict[str, str]:
    """Every answer right except the first `wrong` questions."""
    answers = all_correct()
    for q in PAPER.questions[:wrong]:
        answers[q.id] = "zzz"
    return answers


# the committed content


def test_the_paper_has_four_sections_and_forty_questions_in_order():
    assert len(PAPER.sections) == 4 and len(PAPER.questions) == 40
    assert [q.number for q in PAPER.questions] == list(range(1, 41))
    assert [s.id for s in PAPER.sections] == ["s1", "s2", "s3", "s4"]
    assert len({q.id for q in PAPER.questions}) == 40
    assert PAPER.id.startswith("listening-")


def test_every_key_rests_on_a_sentence_that_is_in_the_transcript():
    text = {s.id: "\n".join(s.transcript) for s in PAPER.sections}
    for q in PAPER.questions:
        assert q.evidence in text[q.section_id], q.id
        if q.task_type not in ("multiple_choice", "matching"):
            assert any(
                normalise(v) in normalise(q.evidence)
                for v in (q.key, *q.accept)
            ), q.id
        else:
            assert q.key in q.choices, q.id


def test_every_recording_was_built_from_the_current_script():
    manifest = json.loads(listening.MANIFEST_PATH.read_text(encoding="utf-8"))
    for s in PAPER.sections:
        entry = manifest["sections"][s.id]
        assert entry["sha256"] == listening.fingerprint(PAPER.style, s), (
            f"{s.id}: the script changed; run scripts/build_listening_audio.py"
        )
        assert listening.audio_path(s.id) is not None
        assert listening.audio_duration(s.id) > 60
    assert build_listening_audio.stale(PAPER, manifest) == []


def test_editing_a_script_makes_its_recording_stale():
    manifest = json.loads(listening.MANIFEST_PATH.read_text(encoding="utf-8"))
    first = PAPER.sections[0]
    edited = listening.Section(
        first.id, first.number, first.title, first.context, first.speakers,
        (*first.transcript[:-1], first.transcript[-1] + " Changed."),
    )
    edited_paper = listening.ListeningSet(PAPER.id, PAPER.style, (edited, *PAPER.sections[1:]),
                                          PAPER.questions)
    assert [s.id for s in build_listening_audio.stale(edited_paper, manifest)] == [first.id]


# marking


@pytest.mark.parametrize("given", ["Mon", " MON. ", "mon"])
def test_a_completion_match_ignores_case_punctuation_and_spacing(given):
    q = next(q for q in PAPER.questions if q.task_type == "form_completion")
    assert listening.is_correct(
        listening.Question(**{**q.__dict__, "key": "Mon", "accept": ()}), given
    )


def test_a_blank_or_wrong_answer_is_not_correct():
    q = PAPER.questions[0]
    assert not listening.is_correct(q, "") and not listening.is_correct(q, "zzz")


def test_full_marks_and_no_marks_mark_to_the_ends_of_the_table():
    full = listening.mark(PAPER, all_correct(), TIMING)
    assert (full.raw, full.total, full.scaled_40, full.band) == (40, 40, 40, 9.0)
    none = listening.mark(PAPER, {}, TIMING)
    assert (none.raw, none.scaled_40, none.band) == (0, 0, 0.0)
    assert all(not r.correct and r.answer_label == "(no answer)" for r in none.questions)


@pytest.mark.parametrize(
    ("raw", "band"),
    [(39, 9.0), (38, 8.5), (35, 8.0), (34, 7.5), (30, 7.0), (29, 6.5), (23, 6.0), (22, 5.5),
     (16, 5.0), (15, 4.5), (10, 4.0), (6, 3.0), (1, 1.0), (0, 0.0)],
)
def test_the_band_table_follows_the_published_anchors(raw, band):
    assert band_for(raw, 40, listening.BAND_TABLE) == (raw, band)


@pytest.mark.parametrize(("raw", "lowest"), [(16, 5.0), (23, 6.0), (30, 7.0), (35, 8.0), (39, 9.0)])
def test_the_idp_anchor_marks_are_the_lowest_raw_for_their_band(raw, lowest):
    assert band_for(raw, 40, listening.BAND_TABLE)[1] == lowest
    assert band_for(raw - 1, 40, listening.BAND_TABLE)[1] < lowest


def test_the_report_names_its_sources_its_transcripts_and_its_caveats():
    report = listening.mark(PAPER, correct_except(3), TIMING)
    assert report.timing_source_url.startswith("https://ielts.org/")
    assert report.band_source_url.startswith("https://ielts.idp.com/")
    assert "about 9 minutes" in report.set_note and "author" in report.keys_note
    assert "interpolation" in report.band_note
    assert [t.id for t in report.transcripts] == ["s1", "s2", "s3", "s4"]
    assert report.transcripts[0].lines == list(PAPER.sections[0].transcript)


def test_a_miss_carries_the_key_and_its_evidence_and_by_type_counts_add_up():
    report = listening.mark(PAPER, correct_except(3), TIMING)
    assert report.raw == 37
    misses = [r for r in report.questions if not r.correct]
    assert [m.number for m in misses] == [1, 2, 3]
    assert all(m.evidence and m.key for m in misses)
    assert sum(t.correct for t in report.by_type) == 37
    assert sum(t.total for t in report.by_type) == 40


# the API


def sit(client: TestClient, answers: dict[str, str]) -> dict:
    attempt_id = client.post("/ielts/listening/attempts").json()["id"]
    client.post(f"/ielts/listening/attempts/{attempt_id}/start")
    resp = client.post(f"/ielts/listening/attempts/{attempt_id}/submit", json={"answers": answers})
    assert resp.status_code == 200
    return resp.json()


def test_info_states_the_clock_the_sources_and_the_caveats(client):
    info = client.get("/ielts/listening/info").json()
    assert info["allowed_s"] == 1800 and info["question_count"] == 40
    assert "ielts.org" in info["timing_source_url"] and "idp.com" in info["band_source_url"]
    assert "not been checked by a human" in info["keys_note"]


def test_the_clock_is_thirty_minutes_and_the_paper_has_no_keys_before_submit(client, clock):
    attempt_id = client.post("/ielts/listening/attempts").json()["id"]
    created = client.get(f"/ielts/listening/attempts/{attempt_id}").json()
    assert created["status"] == "created" and created["paper"] is None
    started = client.post(f"/ielts/listening/attempts/{attempt_id}/start").json()
    assert started["remaining_s"] == 1800
    clock.now += 90
    running = client.get(f"/ielts/listening/attempts/{attempt_id}").json()
    assert running["remaining_s"] == 1710
    paper = running["paper"]
    assert len(paper["questions"]) == 40 and len(paper["sections"]) == 4
    for served in paper["questions"]:
        assert set(served) == {
            "id", "number", "section_id", "task_type", "task_label", "instruction", "prompt",
            "choices",
        }
    for section in paper["sections"]:
        assert set(section) == {"id", "number", "title", "context", "audio_url", "audio_s"}
        assert "transcript" not in section
    assert client.get(f"/ielts/listening/attempts/{attempt_id}/report").status_code == 409


def test_the_audio_is_served_only_once_the_attempt_has_started(client):
    attempt_id = client.post("/ielts/listening/attempts").json()["id"]
    url = f"/ielts/listening/attempts/{attempt_id}/audio/s1"
    assert client.get(url).status_code == 409
    client.post(f"/ielts/listening/attempts/{attempt_id}/start")
    audio = client.get(url)
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/mpeg"
    assert audio.content == listening.audio_path("s1").read_bytes()
    assert client.get(f"/ielts/listening/attempts/{attempt_id}/audio/s9").status_code == 404
    assert client.get("/ielts/listening/attempts/nope/audio/s1").status_code == 404


def test_the_audio_supports_range_requests_so_the_player_can_seek(client):
    attempt_id = client.post("/ielts/listening/attempts").json()["id"]
    client.post(f"/ielts/listening/attempts/{attempt_id}/start")
    part = client.get(
        f"/ielts/listening/attempts/{attempt_id}/audio/s2", headers={"Range": "bytes=100-199"}
    )
    assert part.status_code == 206 and len(part.content) == 100
    assert part.content == listening.audio_path("s2").read_bytes()[100:200]


def test_the_audio_is_still_served_after_submit_but_never_calls_a_model(client):
    out = sit(client, all_correct())
    resp = client.get(f"/ielts/listening/attempts/{out['id']}/audio/s1")
    assert resp.status_code == 200


def test_autosave_resumes_and_ignores_unknown_question_ids(client):
    attempt_id = client.post("/ielts/listening/attempts").json()["id"]
    client.post(f"/ielts/listening/attempts/{attempt_id}/start")
    first = PAPER.questions[0].id
    saved = client.put(
        f"/ielts/listening/attempts/{attempt_id}/draft",
        json={"answers": {first: "Mon", "bogus": "x"}},
    ).json()
    assert saved["answers"] == {first: "Mon"} and saved["answered"] == 1
    assert client.get(f"/ielts/listening/attempts/{attempt_id}").json()["answers"] == {first: "Mon"}


def test_submit_marks_synchronously_and_the_report_shows_evidence_and_transcripts(client):
    out = sit(client, correct_except(5))
    assert out["status"] == "graded" and out["raw"] == 35 and out["band"] == 8.0
    assert out["paper"] is None
    report = ListeningReport.model_validate(
        client.get(f"/ielts/listening/attempts/{out['id']}/report").json()
    )
    assert report.scaled_40 == 35 and len(report.transcripts) == 4
    assert all(r.evidence for r in report.questions if not r.correct)


def test_a_repeated_submit_changes_nothing(client):
    out = sit(client, all_correct())
    again = client.post(
        f"/ielts/listening/attempts/{out['id']}/submit", json={"answers": {}}
    ).json()
    assert again["raw"] == 40 and again["band"] == 9.0


def test_a_late_submit_is_marked_on_the_last_autosave(client, clock):
    attempt_id = client.post("/ielts/listening/attempts").json()["id"]
    client.post(f"/ielts/listening/attempts/{attempt_id}/start")
    q = PAPER.questions[0]
    client.put(f"/ielts/listening/attempts/{attempt_id}/draft", json={"answers": {q.id: q.key}})
    clock.now += LISTENING_ALLOWED_S + GRACE_S + 5
    out = client.post(
        f"/ielts/listening/attempts/{attempt_id}/submit", json={"answers": all_correct()}
    ).json()
    assert out["late"] is True and out["raw"] == 1


def test_a_writing_attempt_is_not_a_listening_attempt(client, store):
    writing = store.create("p", "prompt")
    assert client.get(f"/ielts/listening/attempts/{writing.id}").status_code == 404
    assert client.get("/ielts/listening/attempts").json() == []


def test_a_listening_attempt_is_unknown_to_the_other_modules(client):
    attempt_id = sit(client, all_correct())["id"]
    for url in (
        f"/ielts/attempts/{attempt_id}", f"/ielts/reading/attempts/{attempt_id}",
        f"/ielts/writing/attempts/{attempt_id}",
    ):
        assert client.get(url).status_code == 404, url


def test_listening_attempts_are_listed(client):
    sit(client, all_correct())
    assert len(client.get("/ielts/listening/attempts").json()) == 1


# the store


def test_an_older_mocks_table_gains_the_listening_column_and_keeps_its_rows(tmp_path: Path):
    path = tmp_path / "old.sqlite3"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE mocks (id TEXT PRIMARY KEY, created_at REAL NOT NULL, "
        "reading_attempt_id TEXT, writing_attempt_id TEXT)"
    )
    db.execute("INSERT INTO mocks VALUES ('m1', 1.0, 'r1', 'w1')")
    db.commit()
    db.close()

    store = AttemptStore(path)
    mock = store.get_mock("m1")
    assert mock.attempt_ids["reading"] == "r1" and mock.attempt_ids["writing"] == "w1"
    assert mock.attempt_ids["listening"] is None
    store.link_mock_attempt("m1", "listening", "l1")
    assert store.get_mock("m1").attempt_ids["listening"] == "l1"
    AttemptStore(path)  # opening it again is harmless
