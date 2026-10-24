"""
Importers for local third-party data, tested on tiny synthetic fixtures only: the writing9
download goes through httpx's mock transport and the Cambridge book importer reads text the
test writes itself. No network, no OCR binary, no real book or essay text.
"""
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.exam_set.importers import cambridge, writing9
from evals.exam_set.local_store import load_manifest, read_items
from evals.exam_set.schema import LabelTier

# ── writing9 ─────────────────────────────────────────────────────────────────


def w9_row(i: int, band: float = 6.0, **kw: Any) -> dict[str, Any]:
    row = {
        "id": f"id{i}", "question": f"Synthetic question {i}?", "essay": f"Synthetic essay {i}.",
        "task_type": "essay", "question_type": "Opinion", "word_count": 250,
        "band_overall": band, "band_task": band, "band_coherence": band,
        "band_lexical": int(band), "band_grammar": int(band),
    }
    row.update(kw)
    return row


def fake_hub(rows: list[dict[str, Any]], calls: list[dict[str, str]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        q = dict(request.url.params)
        calls.append(q)
        offset, length = int(q["offset"]), int(q["length"])
        page = rows[offset: offset + length]
        return httpx.Response(
            200, json={"rows": [{"row": r} for r in page], "num_rows_total": len(rows)}
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_page_offsets_spread_across_the_split():
    offsets = writing9.page_offsets(total=100_000, limit=100, page_size=25)
    assert len(offsets) == 4 and offsets[0] == 0 and offsets[-1] > 50_000
    assert writing9.page_offsets(total=30, limit=1000, page_size=25) == [0, 5]


def test_row_to_item_maps_labels_and_marks_the_tier():
    item = writing9.row_to_item(w9_row(1, 6.5), "train")
    assert item is not None
    assert item.id == "w9_id1" and item.skill == "writing_t2" and item.split == "train"
    assert item.labels is not None and item.labels.overall == 6.5
    assert item.labels.criteria == {"TR": 6.5, "CC": 6.5, "LR": 6, "GRA": 6}
    assert item.label_provenance.tier == LabelTier.site_or_model_weak
    assert item.source.kind == "third_party_local" and not item.committable()


@pytest.mark.parametrize(
    "bad", [{"band_overall": None}, {"band_task": 6.3}, {"essay": ""}, {"band_grammar": None}]
)
def test_row_to_item_rejects_unusable_rows(bad: dict[str, Any]):
    assert writing9.row_to_item(w9_row(1, **bad), "train") is None


def test_download_dedupes_filters_and_caps_per_band(tmp_path: Path):
    rows = (
        [w9_row(i, 4.0) for i in range(6)]
        + [w9_row(6, 7.0), w9_row(0, 4.0)]  # duplicate id0
        + [w9_row(7, 7.0, task_type="chart"), w9_row(8, 7.0, band_overall=None)]
    )
    calls: list[dict[str, str]] = []
    with fake_hub(rows, calls) as client:
        items, stats = writing9.download_items(
            client, limit=50, page_size=5, per_band_cap=3, pause_s=0
        )
    assert [i.id for i in items] == ["w9_id0", "w9_id1", "w9_id2", "w9_id6"]
    assert stats.skipped_duplicate == 1 and stats.skipped_not_task2 == 1
    assert stats.skipped_invalid == 1 and stats.skipped_band_cap == 3
    assert all(c["dataset"] == writing9.DATASET for c in calls)


def test_fetch_rows_backs_off_on_rate_limit_then_succeeds():
    attempts: list[int] = []
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "7"})
        if len(attempts) == 2:
            return httpx.Response(429)
        return httpx.Response(200, json={"rows": [{"row": w9_row(1)}], "num_rows_total": 5})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        rows, total = writing9.fetch_rows(client, "train", 0, 1, sleep=delays.append)
    assert total == 5 and len(rows) == 1 and delays == [7.0, 4.0]


def test_fetch_rows_gives_up_after_repeated_rate_limits():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            writing9.fetch_rows(client, "train", 0, 1, sleep=lambda s: None)


def test_download_rejects_the_synthetic_split():
    with fake_hub([], []) as client, pytest.raises(ValueError):
        writing9.download_items(client, limit=1, split="band9_augmented")


def test_import_writes_only_to_the_local_dir_and_records_manifest(tmp_path: Path):
    with fake_hub([w9_row(i, 5.0 + (i % 3)) for i in range(9)], []) as client:
        record, stats = writing9.import_writing9(tmp_path, client, limit=9, pause_s=0)
    assert stats.kept == 9 and record.count == 9
    assert {p.name for p in tmp_path.iterdir()} == {"manifest.json", writing9.FILE_NAME}
    (entry,) = load_manifest(tmp_path)["sources"]
    assert entry["source_id"] == "writing9" and entry["label_tier"] == "site_or_model_weak"
    assert "unknown" in entry["licence_status"]
    assert "NOT agreement with an examiner" in entry["caveat"]
    assert len(read_items(tmp_path / writing9.FILE_NAME)) == 9


# ── Cambridge books ──────────────────────────────────────────────────────────

SYNTHETIC_BOOK = """\
Synthetic Practice Book A

TEST 1

WRITING TASK 1
You should spend about 20 minutes on this task.
The synthetic chart shows imaginary figures for three invented towns.
Summarise the information. Write at least 150 words.

Sample answer
The chart shows made-up figures. Town A rose while Town B fell. (Band 7)

Examiner comments
This is synthetic text. Band 7.

WRITING TASK 2
You should spend about 40 minutes on this task.
Write about the following topic:
Some invented people think purple is the best colour. Discuss both views.
Give reasons for your answer. Write at least 250 words.

Sample answer - Band 6.5
Purple is a colour. Some people like it and some do not. In my view both sides
have a point.

However, taste differs.

Examiner comments: The essay is invented for a test.
Task Response: 7
Coherence and Cohesion: 6
Lexical Resource: 6.5
Grammatical Range and Accuracy: 6.5

Sample answer
This second script has no published band and must be skipped.

TEST 2

WRITING TASK 2
Write about the following topic:
Invented topic two. Write at least 250 words.

Model answer
A short invented script.

Examiner's comments
The examiner would give this script a band score of 8.
"""


def test_parse_extracts_prompts_samples_bands_and_comments():
    result = cambridge.parse_book_text(SYNTHETIC_BOOK, "Synthetic Practice Book A")
    assert result.tasks_seen == 3 and result.skipped_no_band == 1
    t1, t2, t2b = result.items
    assert [i.id for i in (t1, t2, t2b)] == [
        "cam_synthetic_practice_book_a_t1_w1_1",
        "cam_synthetic_practice_book_a_t1_w2_1",
        "cam_synthetic_practice_book_a_t2_w2_1",
    ]
    assert t1.skill == "writing_t1" and t1.extras["task1_visual_not_extracted"] is True
    assert t2.skill == "writing_t2"
    assert t2.prompt.startswith("Write about the following topic:")
    assert "spend about" not in t2.prompt and t2.prompt.endswith("at least 250 words.")
    assert t2.response is not None and t2.response.endswith("However, taste differs.")
    assert t2.labels is not None and t2.labels.overall == 6.5
    assert t2.labels.criteria == {"TR": 7, "CC": 6, "LR": 6.5, "GRA": 6.5}
    assert t2.examiner_comment is not None and "invented for a test" in t2.examiner_comment
    assert t2b.labels is not None and t2b.labels.overall == 8.0 and t2b.labels.criteria == {}
    for item in result.items:
        assert item.label_provenance.tier == LabelTier.examiner
        assert item.source.kind == "third_party_local"


def test_parse_ignores_text_before_the_first_task_and_unlabelled_samples():
    text = "Intro text\n\nWRITING TASK 2\nPrompt. Write at least 250 words.\nSample answer\nBody.\n"
    result = cambridge.parse_book_text(text, "B")
    assert result.items == [] and result.skipped_no_band == 1


def test_import_folder_reads_text_runs_ocr_for_images_and_writes_locally(tmp_path: Path):
    books, local = tmp_path / "books", tmp_path / "local"
    books.mkdir()
    (books / "book_a.txt").write_text(SYNTHETIC_BOOK)
    (books / "scan.png").write_bytes(b"fake-image")
    (books / "notes.docx").write_bytes(b"ignored")
    seen: list[bytes] = []

    def fake_ocr(image: bytes) -> str:
        seen.append(image)
        return (
            "WRITING TASK 2\nOcr prompt. Write at least 250 words.\n"
            "Sample answer (Band 5)\nOcr essay.\n"
        )

    record, totals = cambridge.import_folder(books, local, ocr=fake_ocr)
    assert seen == [b"fake-image"]
    assert totals["files"] == 2 and totals["unsupported"] == 1 and totals["items"] == 4
    assert record.label_tier == "examiner" and record.count == 4
    assert (local / cambridge.EXTRACT_DIR / "scan.txt").read_text().startswith("WRITING TASK 2")
    assert {i.id for i in read_items(local / cambridge.FILE_NAME)} >= {"cam_scan_w2_1"}
    (entry,) = load_manifest(local)["sources"]
    assert "never committed" in entry["licence_status"]


def test_pdf_pages_without_a_text_layer_fall_back_to_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Page:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class Reader:
        def __init__(self, path: str) -> None:
            self.pages = [Page("A page with a perfectly good text layer, long enough."), Page("")]

    monkeypatch.setattr("pypdf.PdfReader", Reader)
    monkeypatch.setattr(cambridge, "_page_images", lambda page: [b"scan"])
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF")
    text = cambridge.extract_text(pdf, ocr=lambda image: f"ocr:{image.decode()}")
    assert text == "A page with a perfectly good text layer, long enough.\n\nocr:scan"


def test_ocr_unavailable_has_an_actionable_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(cambridge.OcrUnavailableError, match="tesseract"):
        cambridge.tesseract_ocr(b"x")


def test_unsupported_file_type_is_rejected(tmp_path: Path):
    path = tmp_path / "x.docx"
    path.write_bytes(b"x")
    with pytest.raises(ValueError):
        cambridge.extract_text(path)


def test_manifest_json_is_valid_after_cambridge_import(tmp_path: Path):
    books = tmp_path / "b"
    books.mkdir()
    (books / "x.txt").write_text(SYNTHETIC_BOOK)
    cambridge.import_folder(books, tmp_path / "l")
    json.loads((tmp_path / "l" / "manifest.json").read_text())
