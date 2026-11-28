"""
The IELTS Speaking mock: the upload and submit flow, marking with a scripted audio model, the
partial band (Pronunciation not assessed), the guards, the sample set, and the Gemini client
against a mock transport.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from services.gateway.src.gateway import ielts_api
from services.gateway.src.gateway.ielts import speaking
from services.gateway.src.gateway.ielts.llm import ModelUnavailableError
from services.gateway.src.gateway.ielts.speaking_llm import AudioClip, GeminiAudioAssessor
from services.gateway.src.gateway.ielts_speaking_api import get_assessor, get_audio_dir
from services.gateway.src.gateway.ielts_store import AttemptStore
from services.gateway.src.gateway.limits import ielts_limiter
from services.gateway.src.gateway.main import app
from shared.src.shared.budget import Ledger
from tests.gateway.speaking_fakes import SPEAKING_BAND, TRANSCRIPTS, FakeAssessor, reply
from tests.gateway.test_ielts_writing_module import Clock

WAV = b"RIFFfakeaudio"


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def assessor() -> FakeAssessor:
    return FakeAssessor()


@pytest.fixture
def audio_dir(tmp_path: Path) -> Path:
    return tmp_path / "audio"


@pytest.fixture
def client(
    clock: Clock, assessor: FakeAssessor, audio_dir: Path, tmp_path: Path,
) -> Iterator[TestClient]:
    store = AttemptStore(tmp_path / "ielts-local" / "attempts.sqlite3", clock)
    app.dependency_overrides[ielts_api.get_store] = lambda: store
    app.dependency_overrides[ielts_api.get_ledger] = lambda: Ledger(
        tmp_path / "ledger.json", ceiling_usd=5.0,
    )
    app.dependency_overrides[get_assessor] = lambda: assessor
    app.dependency_overrides[get_audio_dir] = lambda: audio_dir
    ielts_limiter().reset()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def make(client: TestClient, sample: bool = False) -> str:
    resp = client.post("/ielts/speaking/attempts", params={"sample": str(sample).lower()})
    assert resp.status_code == 201
    attempt_id = str(resp.json()["id"])
    assert client.post(f"/ielts/speaking/attempts/{attempt_id}/start").status_code == 200
    return attempt_id


def upload(
    client: TestClient, attempt_id: str, part: int, duration: str = "20", mime: str = "audio/wav",
):
    return client.put(
        f"/ielts/speaking/attempts/{attempt_id}/parts/{part}/audio", content=WAV,
        headers={"content-type": mime, "x-duration-s": duration},
    )


def record_all(client: TestClient, attempt_id: str) -> None:
    for part in (1, 2, 3):
        assert upload(client, attempt_id, part).status_code == 200


def report_of(client: TestClient, attempt_id: str) -> dict:
    resp = client.get(f"/ielts/speaking/attempts/{attempt_id}/report")
    assert resp.status_code == 200, resp.text
    return resp.json()


# the flow and the partial band


def test_three_recordings_give_a_partial_band_with_pronunciation_not_assessed(
    client: TestClient, assessor: FakeAssessor, audio_dir: Path,
):
    attempt_id = make(client)
    record_all(client, attempt_id)
    done = client.post(f"/ielts/speaking/attempts/{attempt_id}/submit")
    assert done.status_code == 200
    body = client.get(f"/ielts/speaking/attempts/{attempt_id}").json()
    assert body["status"] == "graded" and body["band"] == SPEAKING_BAND and body["partial"] is True

    report = report_of(client, attempt_id)
    assert report["band"] == SPEAKING_BAND and report["partial"] is True
    assert report["not_assessed"] == ["P"]
    by_id = {c["id"]: c for c in report["criteria"]}
    assert [by_id[k]["band"] for k in ("FC", "LR", "GRA")] == [7.0, 6.5, 6.0]
    assert by_id["P"]["assessed"] is False and by_id["P"]["band"] is None
    assert by_id["P"]["reason"]
    assert "Pronunciation not assessed" in report["partial_note"]
    assert report["descriptors_source_url"] == speaking.DESCRIPTORS_URL
    assert [p["transcript"] for p in report["parts"]] == list(TRANSCRIPTS.values())
    assert report["parts"][0]["fillers"] == 1  # "um"
    assert report["priorities"][0]["criterion"] == "GRA"
    # the model heard all three recordings, labelled by part
    assert [c.label.split(" ")[1] for c in assessor.calls[0]] == ["1", "2", "3"]
    # the voice is deleted once the report is stored
    assert not (audio_dir / attempt_id).exists()


def test_part_two_is_counted_against_its_two_minutes(client: TestClient):
    attempt_id = make(client)
    upload(client, attempt_id, 1)
    upload(client, attempt_id, 2, duration="100")
    upload(client, attempt_id, 3)
    client.post(f"/ielts/speaking/attempts/{attempt_id}/submit")
    part2 = report_of(client, attempt_id)["parts"][1]
    assert part2["allowed_s"] == 120 and part2["duration_s"] == 100.0
    assert part2["words"] == 18 and part2["words_per_minute"] == 10.8


def test_a_recording_that_speaks_to_the_marker_is_capped(
    client: TestClient, assessor: FakeAssessor,
):
    assessor.texts = [reply(addressed_to_marker=True)]
    attempt_id = make(client)
    record_all(client, attempt_id)
    client.post(f"/ielts/speaking/attempts/{attempt_id}/submit")
    report = report_of(client, attempt_id)
    assert report["band"] == 5.0 and report["flags"]


def test_too_little_speech_gives_no_band(client: TestClient, assessor: FakeAssessor):
    assessor.texts = [reply(heard_speech=False, parts=[], criteria={})]
    attempt_id = make(client)
    record_all(client, attempt_id)
    client.post(f"/ielts/speaking/attempts/{attempt_id}/submit")
    report = report_of(client, attempt_id)
    assert report["status"] == "no_speech" and report["band"] is None
    body = client.get(f"/ielts/speaking/attempts/{attempt_id}").json()
    assert body["status"] == "graded" and body["band"] is None and body["error"] is None


def test_a_late_re_recording_keeps_the_stored_clip(
    client: TestClient, clock: Clock, audio_dir: Path,
):
    attempt_id = make(client)
    record_all(client, attempt_id)
    clock.now += 60 * 60
    again = upload(client, attempt_id, 1, mime="audio/webm")
    assert again.status_code == 409
    assert sorted(p.name for p in (audio_dir / attempt_id).iterdir()) == [
        "part1.wav", "part2.wav", "part3.wav",
    ]
    done = client.post(f"/ielts/speaking/attempts/{attempt_id}/submit")
    assert done.status_code == 200
    assert client.get(f"/ielts/speaking/attempts/{attempt_id}").json()["status"] == "graded"


def test_an_unreadable_reply_is_retried_once_then_retryable_with_the_audio_kept(
    client: TestClient, assessor: FakeAssessor, audio_dir: Path,
):
    assessor.texts = ["not json"]
    attempt_id = make(client)
    record_all(client, attempt_id)
    client.post(f"/ielts/speaking/attempts/{attempt_id}/submit")
    body = client.get(f"/ielts/speaking/attempts/{attempt_id}").json()
    assert body["status"] == "ungraded" and body["retryable"] is True
    assert len(assessor.calls) == 2
    assert (audio_dir / attempt_id / "part1.wav").exists()

    assessor.texts = [reply()]
    assessor.calls.clear()
    retry = client.post(f"/ielts/speaking/attempts/{attempt_id}/regrade")
    assert retry.status_code == 200
    assert client.get(f"/ielts/speaking/attempts/{attempt_id}").json()["status"] == "graded"
    assert not (audio_dir / attempt_id).exists()


def test_a_model_outage_is_retryable(client: TestClient, assessor: FakeAssessor):
    assessor.error = ModelUnavailableError("down")
    attempt_id = make(client)
    record_all(client, attempt_id)
    client.post(f"/ielts/speaking/attempts/{attempt_id}/submit")
    body = client.get(f"/ielts/speaking/attempts/{attempt_id}").json()
    assert body["status"] == "ungraded" and body["retryable"] is True
    assert "try again" in body["error"]


# the guards


def test_submit_needs_every_part(client: TestClient):
    attempt_id = make(client)
    upload(client, attempt_id, 1)
    resp = client.post(f"/ielts/speaking/attempts/{attempt_id}/submit")
    assert resp.status_code == 409 and "part 2, 3" in resp.json()["detail"]


def test_upload_guards(client: TestClient):
    attempt_id = make(client)
    assert upload(client, attempt_id, 4).status_code == 404
    assert upload(client, attempt_id, 1, mime="text/plain").status_code == 415
    assert upload(client, attempt_id, 1, duration="0").status_code == 422
    assert upload(client, attempt_id, 1, duration="9999").status_code == 422
    assert upload(client, attempt_id, 2, duration="151").status_code == 422  # over 120 + 30
    empty = client.put(
        f"/ielts/speaking/attempts/{attempt_id}/parts/1/audio", content=b"",
        headers={"content-type": "audio/wav", "x-duration-s": "5"},
    )
    assert empty.status_code == 422
    missing = client.put(
        f"/ielts/speaking/attempts/{attempt_id}/parts/1/audio", content=WAV,
        headers={"content-type": "audio/wav"},
    )
    assert missing.status_code == 422


def test_a_second_upload_replaces_the_part(client: TestClient, audio_dir: Path):
    attempt_id = make(client)
    upload(client, attempt_id, 1)
    upload(client, attempt_id, 1, mime="audio/webm")
    assert [p.name for p in (audio_dir / attempt_id).iterdir()] == ["part1.webm"]


def test_uploads_are_refused_before_start_and_after_submit(client: TestClient):
    created = client.post("/ielts/speaking/attempts").json()["id"]
    assert upload(client, created, 1).status_code == 409
    attempt_id = make(client)
    record_all(client, attempt_id)
    client.post(f"/ielts/speaking/attempts/{attempt_id}/submit")
    assert upload(client, attempt_id, 1).status_code == 409


def test_a_second_submit_does_not_mark_again(client: TestClient, assessor: FakeAssessor):
    attempt_id = make(client)
    record_all(client, attempt_id)
    client.post(f"/ielts/speaking/attempts/{attempt_id}/submit")
    client.post(f"/ielts/speaking/attempts/{attempt_id}/submit")
    assert len(assessor.calls) == 1


def test_other_modules_attempts_are_not_found_here(client: TestClient):
    reading = client.post("/ielts/reading/attempts").json()["id"]
    assert client.get(f"/ielts/speaking/attempts/{reading}").status_code == 404


def test_the_report_is_not_there_before_marking(client: TestClient):
    attempt_id = make(client)
    assert client.get(f"/ielts/speaking/attempts/{attempt_id}/report").status_code == 409


# the sample set


def test_the_quick_sample_has_short_timings_and_is_labelled(client: TestClient):
    attempt_id = make(client, sample=True)
    body = client.get(f"/ielts/speaking/attempts/{attempt_id}").json()
    assert body["set_id"] == "sp-sample"
    part2 = body["paper"]["parts"][1]
    assert part2["prep_s"] == 5 and part2["talk_s"] == 20
    assert upload(client, attempt_id, 1, duration="41").status_code == 422
    record_all(client, attempt_id)
    client.post(f"/ielts/speaking/attempts/{attempt_id}/submit")
    report = report_of(client, attempt_id)
    assert report["sample"] is True and report["parts"][1]["allowed_s"] == 20


def test_info_names_the_unassessed_criterion_and_the_sample(client: TestClient):
    info = client.get("/ielts/speaking/info").json()
    assert "P" in info["not_assessed"]
    assert info["sample"]["sample"] is True and info["part2_prep_s"] == 60
    assert info["part2_talk_s"] == 120 and info["model"]


def test_full_sets_are_original_and_full_length():
    assert len(speaking.SETS) >= 2
    for item in speaking.SETS:
        assert not item.sample and item.prep_s == 60 and item.talk_s == 120
        assert len(item.part1) >= 3 and len(item.part3) >= 3
    assert speaking.pick_set() in speaking.SETS


# counting in code


def test_features_are_counted_in_code():
    assert speaking.count_words("It's a well-known place, isn't it?") == 6
    assert speaking.count_fillers("Um, well, er, I think hmm so uhh yes") == 4
    assert speaking.count_fillers("The summer was great") == 0
    part = speaking.part_features(1, "one two three four five six", 3.0)
    assert part.words_per_minute is None  # too short a recording to give a pace


# the Gemini client


def gemini_client(handler, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    sleeps: list[float] = []

    async def sleep(s: float) -> None:
        sleeps.append(s)

    ledger = Ledger(tmp_path / "ledger.json", ceiling_usd=5.0)
    assessor = GeminiAudioAssessor(
        ledger=ledger, transport=httpx.MockTransport(handler), api_key="test-key", sleep=sleep,
    )
    return assessor, ledger, sleeps


CLIPS = [AudioClip("PART 1 recording", WAV, "audio/wav", 5.0)]


def ok_body(text: str = "{}") -> dict:
    return {
        "candidates": [{"content": {"parts": [{"thoughtSignature": "x"}, {"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 200},
    }


@pytest.mark.asyncio
async def test_gemini_sends_inline_audio_and_reads_only_the_text_parts(tmp_path, monkeypatch):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ok_body('{"a": 1}'))

    assessor, ledger, _ = gemini_client(handler, tmp_path, monkeypatch)
    answer = await assessor("system text", CLIPS, run_id="r1", temperature=0.2)
    assert answer.text == '{"a": 1}' and answer.model
    request = seen[0]
    assert request.headers["x-goog-api-key"] == "test-key"
    assert ":generateContent" in str(request.url)
    body = json.loads(request.content)
    parts = body["contents"][0]["parts"]
    assert any("inlineData" in p and p["inlineData"]["mimeType"] == "audio/wav" for p in parts)
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert ledger.spent() > 0


@pytest.mark.asyncio
async def test_gemini_retries_a_busy_model_then_falls_back(tmp_path, monkeypatch):
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = str(request.url).split("/models/")[1].split(":")[0]
        models.append(model)
        if len(set(models)) == 1:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json=ok_body())

    assessor, _, sleeps = gemini_client(handler, tmp_path, monkeypatch)
    answer = await assessor("s", CLIPS, run_id="r2", temperature=0.2)
    assert answer.model == models[-1] and models[0] != models[-1]
    assert len(sleeps) >= 1


@pytest.mark.asyncio
async def test_gemini_gives_up_when_every_model_is_down(tmp_path, monkeypatch):
    assessor, _, _ = gemini_client(
        lambda r: httpx.Response(503, json={"error": "busy"}), tmp_path, monkeypatch,
    )
    with pytest.raises(ModelUnavailableError):
        await assessor("s", CLIPS, run_id="r3", temperature=0.2)


@pytest.mark.asyncio
async def test_gemini_without_a_key_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assessor = GeminiAudioAssessor(
        ledger=Ledger(tmp_path / "ledger.json", ceiling_usd=5.0),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=ok_body())),
    )
    with pytest.raises(ModelUnavailableError):
        await assessor("s", CLIPS, run_id="r4", temperature=0.2)
