"""The local blind-scoring pool: shuffled, opaque, key kept apart, scores imported honestly."""
import csv
import json
from pathlib import Path

import pytest

from evals.exam_set import blind_pool
from evals.exam_set.schema import (
    ExamItem,
    LabelProvenance,
    Labels,
    LabelTier,
    Source,
)

PROMPTS = [(f"p{i}", f"Question {i}: discuss.") for i in range(8)]


def teacher(i: int) -> ExamItem:
    return ExamItem(
        id=f"ib_{i}", exam="ielts", skill="writing_t2", task_type="essay_score",
        prompt=f"Teacher question {i}", response=f"Teacher essay {i} " * 40,
        labels=Labels(overall=5.0, criteria={"TR": 5.0, "CC": 5.0, "LR": 5.0, "GRA": 5.0}),
        label_provenance=LabelProvenance(tier=LabelTier.teacher_graded),
        source=Source(kind="third_party_local", license="x"),
    )


async def fake_complete(messages: list[dict[str, str]]) -> str:
    return "A drafted essay about the topic. " * 20


async def test_drafts_spread_over_every_target_band():
    drafts = await blind_pool.draft_essays(PROMPTS, 18, fake_complete)
    assert sorted({d.target for d in drafts}) == list(blind_pool.TARGET_BANDS)
    assert len(drafts) == 18


def test_a_small_instructor_sample_still_covers_the_range_of_bands():
    bands = [4.0] * 8 + [5.0] * 5 + [6.0] * 4 + [7.0, 8.0]
    items = [
        teacher(i).model_copy(update={"labels": Labels(
            overall=b, criteria={"TR": b, "CC": b, "LR": b, "GRA": b})})
        for i, b in enumerate(bands)
    ]
    picked = blind_pool.spread_by_band(items, 5)
    assert len(picked) == 5 and len({i.id for i in picked}) == 5
    overall = [i.labels.overall for i in picked if i.labels]
    assert overall[0] == 4.0 and overall[-1] == 8.0 and overall == sorted(overall)
    assert len(blind_pool.spread_by_band(items, 99)) == len(items)
    assert blind_pool.spread_by_band(items, 0) == []
    assert len(blind_pool.spread_by_band(items, 1)) == 1


async def test_too_short_drafts_are_dropped():
    async def short(messages: list[dict[str, str]]) -> str:
        return "No."

    assert await blind_pool.draft_essays(PROMPTS, 3, short) == []


def test_the_draft_request_carries_the_level_but_never_calls_the_text_a_sample_to_the_reader():
    messages = blind_pool.draft_request("Q?", 5.5)
    assert "band 5.5" in messages[1]["content"] and "Never mention" in messages[0]["content"]


async def test_pool_is_shuffled_blind_and_the_key_is_separate(tmp_path: Path):
    drafts = await blind_pool.draft_essays(PROMPTS, 9, fake_complete)
    counts = blind_pool.build_pool(tmp_path, drafts, [teacher(1), teacher(2)], seed=7)
    assert counts == {"essays": 11, "llm_drafts": 9, "teacher_graded": 2}

    blind = (tmp_path / blind_pool.ESSAYS_JSONL).read_text() + (
        tmp_path / blind_pool.ESSAYS_MD).read_text()
    for leak in ("llm_draft", "teacher_graded", "target_band", "ib_1", "band 4.0"):
        assert leak not in blind
    key = [json.loads(x) for x in (tmp_path / blind_pool.KEY_FILE).read_text().splitlines()]
    assert {k["origin"] for k in key} == {"llm_draft", "teacher_graded"}
    ids = [json.loads(x)["id"] for x in (tmp_path / blind_pool.ESSAYS_JSONL).read_text()
           .splitlines()]
    assert ids == [k["id"] for k in key] and len(set(ids)) == 11
    # the shuffle interleaves the sources instead of listing drafts then teacher essays
    assert [k["origin"] for k in key] != ["llm_draft"] * 9 + ["teacher_graded"] * 2


async def test_rebuilding_a_scored_pool_is_refused_and_changes_nothing(tmp_path: Path):
    blind_pool.build_pool(tmp_path, [], [teacher(1), teacher(2)])
    fill(tmp_path, {0: ("6", "6", "6", "6")})
    before = {p.name: p.read_text() for p in tmp_path.iterdir()}

    async def other_essay(messages: list[dict[str, str]]) -> str:
        return "A different drafted essay on the question. " * 20

    drafts = await blind_pool.draft_essays(PROMPTS, 2, other_essay)
    with pytest.raises(blind_pool.PoolAlreadyScoredError):
        blind_pool.build_pool(tmp_path, drafts, [teacher(1), teacher(2)])
    assert {p.name: p.read_text() for p in tmp_path.iterdir()} == before
    items, _ = blind_pool.import_scores(tmp_path, "d")
    essays = {json.loads(x)["id"]: json.loads(x)["essay"]
              for x in (tmp_path / blind_pool.ESSAYS_JSONL).read_text().splitlines()}
    assert [i.response for i in items] == [essays[i.id.removeprefix("ielts_blind_")]
                                           for i in items]
    assert len(items) == 1 and str(items[0].response).startswith("Teacher essay")


def test_rebuilding_an_unscored_pool_gives_a_sheet_for_the_new_essays(tmp_path: Path):
    blind_pool.build_pool(tmp_path, [], [teacher(1)])
    blind_pool.build_pool(tmp_path, [], [teacher(i) for i in range(3)])
    sheet_ids = [r["id"] for r in csv.DictReader((tmp_path / blind_pool.SCORE_SHEET).open())]
    pool_ids = [json.loads(x)["id"] for x in (tmp_path / blind_pool.ESSAYS_JSONL).read_text()
                .splitlines()]
    assert sheet_ids == pool_ids and len(pool_ids) == 3


def fill(tmp_path: Path, scores: dict[int, tuple[str, str, str, str]]) -> None:
    sheet = tmp_path / blind_pool.SCORE_SHEET
    rows = list(csv.reader(sheet.open()))
    for index, values in scores.items():
        rows[index + 1][1:5] = list(values)
    with sheet.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def test_import_scores_makes_captain_blind_items_with_a_stable_split(tmp_path: Path):
    blind_pool.build_pool(tmp_path, [], [teacher(i) for i in range(4)])
    fill(tmp_path, {0: ("6", "6.5", "6", "6"), 1: ("5", "5", "5.5", "5")})
    items, empty = blind_pool.import_scores(tmp_path, "2026-10-29")
    assert len(items) == 2 and empty == 2
    first = items[0]
    assert first.labels is not None and first.labels.overall == 6.0  # mean 6.125
    assert first.label_provenance.tier == LabelTier.captain_blind
    assert first.label_provenance.blind_to_target is True
    assert first.split in {"dev", "test"}
    again, _ = blind_pool.import_scores(tmp_path, "2026-10-30")
    assert [i.split for i in again] == [i.split for i in items]


def test_import_rejects_partial_rows_and_non_half_bands(tmp_path: Path):
    blind_pool.build_pool(tmp_path, [], [teacher(1)])
    fill(tmp_path, {0: ("6", "", "6", "6")})
    with pytest.raises(blind_pool.ScoreSheetError, match="all four"):
        blind_pool.import_scores(tmp_path, "d")
    fill(tmp_path, {0: ("6", "6.3", "6", "6")})
    with pytest.raises(blind_pool.ScoreSheetError, match="half band"):
        blind_pool.import_scores(tmp_path, "d")
    fill(tmp_path, {0: ("6", "ten", "6", "6")})
    with pytest.raises(blind_pool.ScoreSheetError, match="not a number"):
        blind_pool.import_scores(tmp_path, "d")


def test_two_pools_never_share_ids_even_with_the_same_seed(tmp_path: Path):
    first, second = tmp_path / "a", tmp_path / "b"
    blind_pool.build_pool(first, [], [teacher(i) for i in range(5)])
    blind_pool.build_pool(second, [], [teacher(i) for i in range(5)])

    def ids(d: Path) -> set[str]:
        return {json.loads(x)["id"] for x in (d / blind_pool.ESSAYS_JSONL).read_text()
                .splitlines()}

    assert not ids(first) & ids(second)


def import_pools(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], pools: list[tuple[list[ExamItem], str]]
) -> list[dict]:
    """Build, fully score and import each pool in turn; returns what each import printed."""
    from evals.exam_set.__main__ import main

    printed = []
    for n, (essays, band) in enumerate(pools):
        pool = tmp_path / f"pool{n}"
        blind_pool.build_pool(pool, [], essays)
        fill(pool, {row: (band,) * 4 for row in range(len(essays))})
        assert main(["--local-dir", str(tmp_path / "local"), "import-blind-scores",
                     "--blind-dir", str(pool)]) == 0
        printed.append(json.loads(capsys.readouterr().out))
    return printed


def imported_bands(tmp_path: Path) -> dict[str, float | None]:
    """Overall band per imported essay, keyed by the essay text."""
    from evals.exam_set.local_store import read_items

    items = read_items(tmp_path / "local" / blind_pool.IMPORTED_FILE)
    assert len({i.id for i in items}) == len(items)
    return {str(i.response): i.labels.overall if i.labels else None for i in items}


def test_importing_a_second_pool_keeps_the_first_pools_scores(tmp_path: Path, capsys):
    import_pools(tmp_path, capsys, [([teacher(1)], "6"), ([teacher(2)], "7")])
    assert imported_bands(tmp_path) == {
        teacher(1).response: 6.0, teacher(2).response: 7.0}


def test_an_essay_in_two_pools_is_imported_once_with_its_first_score(tmp_path: Path, capsys):
    reflowed = teacher(1).model_copy(update={"response": "\n  " + str(teacher(1).response)})
    first, second = import_pools(
        tmp_path, capsys, [([teacher(1)], "6"), ([reflowed, teacher(2)], "7")])
    assert first["skipped_duplicates"] == 0
    assert second["skipped_duplicates"] == 1 and second["kept_from_earlier_pools"] == 1
    assert imported_bands(tmp_path) == {
        teacher(1).response: 6.0, teacher(2).response: 7.0}


def test_reimporting_a_pool_updates_its_scores_instead_of_skipping_them(
    tmp_path: Path, capsys
):
    import_pools(tmp_path, capsys, [([teacher(1)], "6")])
    from evals.exam_set.__main__ import main

    pool = tmp_path / "pool0"
    fill(pool, {0: ("6.5",) * 4})
    assert main(["--local-dir", str(tmp_path / "local"), "import-blind-scores",
                 "--blind-dir", str(pool)]) == 0
    assert json.loads(capsys.readouterr().out)["skipped_duplicates"] == 0
    assert imported_bands(tmp_path) == {teacher(1).response: 6.5}
