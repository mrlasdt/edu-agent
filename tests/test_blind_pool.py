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


def test_rebuilding_never_overwrites_the_owners_scores(tmp_path: Path):
    blind_pool.build_pool(tmp_path, [], [teacher(1)])
    sheet = tmp_path / blind_pool.SCORE_SHEET
    sheet.write_text(sheet.read_text().replace(",,,,", ",6,6,6,6", 1))
    before = sheet.read_text()
    blind_pool.build_pool(tmp_path, [], [teacher(1)])
    assert sheet.read_text() == before


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
