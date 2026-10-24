"""
Importer for the writing9 IELTS Task 2 essay corpus on Hugging Face
(`ndtran0101/writing9-ielts-essays`).

Downloaded at run time into the gitignored local directory; nothing from it is committed.
Licence status is "unknown" (scraped user submissions; the card asks for credit to
writing9.com and says to follow the source site's terms), so this is for private statistics.

Label tier is `site_or_model_weak`: the bands are the site's scores, not examiners'. Use the
data for large-n statistics, judge calibration and agreement metrics reported with that
caveat. Never present agreement with it as agreement with an examiner.

Rows are fetched through the datasets-server `/rows` API (no extra dependencies). The train
split is stored in band-sorted blocks, so the importer samples small pages at evenly spaced
offsets rather than reading the head, and can cap rows per overall band.

`chillies/IELTS-writing-task-2-evaluation` was checked and deliberately not imported: one
free-text band string per row (250 distinct values, entries like "<4"), no per-criterion
labels, no stated source or scorer, visible duplicates and templated evaluations.
"""
from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from evals.exam_set.local_store import SourceRecord, register_source
from evals.exam_set.schema import (
    TIER_CAVEATS,
    Design,
    ExamItem,
    LabelProvenance,
    Labels,
    LabelTier,
    Source,
)

DATASET = "ndtran0101/writing9-ielts-essays"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
SOURCE_ID = "writing9"
SOURCE_URL = f"https://huggingface.co/datasets/{DATASET}"
LICENCE_STATUS = (
    "unknown; scraped from writing9.com user submissions; card asks for credit to writing9.com "
    "and to follow the source site's terms; personal local use only, never committed"
)
FILE_NAME = "ielts_writing9.jsonl"
SPLITS = ("train", "test")
_COLUMN_TO_CRITERION = {
    "band_task": "TR",
    "band_coherence": "CC",
    "band_lexical": "LR",
    "band_grammar": "GRA",
}
MAX_PAGE = 100
MAX_RETRIES = 6
DEFAULT_PAUSE_S = 0.5


@dataclass
class ImportStats:
    fetched: int = 0
    kept: int = 0
    skipped_not_task2: int = 0
    skipped_invalid: int = 0
    skipped_duplicate: int = 0
    skipped_band_cap: int = 0


def page_offsets(total: int, limit: int, page_size: int) -> list[int]:
    """Evenly spaced page offsets covering `limit` rows across `total` rows."""
    pages = max(1, -(-limit // page_size))
    pages = min(pages, max(1, -(-total // page_size)))
    stride = max(page_size, total // pages)
    return [min(i * stride, max(0, total - page_size)) for i in range(pages)]


def _half_band(value: Any) -> float | None:
    try:
        band = float(value)
    except (TypeError, ValueError):
        return None
    return band if 0.0 <= band <= 9.0 and (band * 2) % 1 == 0 else None


def row_to_item(row: dict[str, Any], split: str) -> ExamItem | None:
    """Map one writing9 row to an ExamItem, or None when it is unusable as an IELTS item."""
    essay = str(row.get("essay") or "").strip()
    prompt = str(row.get("question") or "").strip()
    overall = _half_band(row.get("band_overall"))
    criteria = {
        crit: band
        for col, crit in _COLUMN_TO_CRITERION.items()
        if (band := _half_band(row.get(col))) is not None
    }
    if not essay or not prompt or overall is None or len(criteria) != len(_COLUMN_TO_CRITERION):
        return None
    return ExamItem(
        id=f"w9_{row['id']}",
        exam="ielts",
        skill="writing_t2",
        task_type="essay_score",
        prompt=prompt,
        response=essay,
        labels=Labels(overall=overall, criteria=criteria),
        label_provenance=LabelProvenance(
            tier=LabelTier.site_or_model_weak,
            note=TIER_CAVEATS[LabelTier.site_or_model_weak],
        ),
        source=Source(
            kind="third_party_local",
            license="unknown",
            source_id=SOURCE_ID,
            url=SOURCE_URL,
            attribution="writing9.com via Hugging Face " + DATASET,
        ),
        design=Design(generator="scraped"),
        split="test" if split == "test" else "train",
        extras={"question_type": row.get("question_type"), "word_count": row.get("word_count")},
    )


def fetch_rows(
    client: httpx.Client,
    split: str,
    offset: int,
    length: int,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[dict[str, Any]], int]:
    """One `/rows` page. The public API rate-limits (429), so back off and retry."""
    params: dict[str, str | int] = {
        "dataset": DATASET, "config": "default", "split": split,
        "offset": offset, "length": length,
    }
    for attempt in range(MAX_RETRIES + 1):
        resp = client.get(ROWS_URL, params=params)
        if resp.status_code == 429 and attempt < MAX_RETRIES:
            retry_after = resp.headers.get("retry-after", "")
            sleep(float(retry_after) if retry_after.isdigit() else min(60.0, 2.0 * 2**attempt))
            continue
        resp.raise_for_status()
        body = resp.json()
        return [r["row"] for r in body["rows"]], int(body["num_rows_total"])
    raise AssertionError("unreachable")


def download_items(
    client: httpx.Client,
    *,
    limit: int,
    split: str = "train",
    page_size: int = 25,
    per_band_cap: int | None = None,
    pause_s: float = DEFAULT_PAUSE_S,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[ExamItem], ImportStats]:
    """Sample about `limit` Task 2 items from `split`, deduplicated by id."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS} (band9_augmented is synthetic)")
    page_size = min(page_size, MAX_PAGE)
    stats = ImportStats()
    _, total = fetch_rows(client, split, 0, 1, sleep)
    seen: set[str] = set()
    per_band: Counter[float] = Counter()
    items: list[ExamItem] = []
    for offset in page_offsets(total, limit, page_size):
        rows, _ = fetch_rows(client, split, offset, page_size, sleep)
        for row in rows:
            stats.fetched += 1
            if len(items) >= limit:
                break
            if row.get("task_type") != "essay":
                stats.skipped_not_task2 += 1
                continue
            item = row_to_item(row, split)
            if item is None:
                stats.skipped_invalid += 1
                continue
            if item.id in seen:
                stats.skipped_duplicate += 1
                continue
            assert item.labels is not None and item.labels.overall is not None
            if per_band_cap is not None and per_band[item.labels.overall] >= per_band_cap:
                stats.skipped_band_cap += 1
                continue
            seen.add(item.id)
            per_band[item.labels.overall] += 1
            items.append(item)
        if pause_s:
            sleep(pause_s)
    stats.kept = len(items)
    return items, stats


def import_writing9(
    directory: Path,
    client: httpx.Client,
    *,
    limit: int = 1200,
    split: str = "train",
    per_band_cap: int | None = None,
    pause_s: float = DEFAULT_PAUSE_S,
) -> tuple[SourceRecord, ImportStats]:
    items, stats = download_items(
        client, limit=limit, split=split, per_band_cap=per_band_cap, pause_s=pause_s
    )
    record = register_source(
        directory,
        source_id=SOURCE_ID,
        name="writing9 IELTS Task 2 essays (sample)",
        url=SOURCE_URL,
        licence_status=LICENCE_STATUS,
        items=items,
        file_name=FILE_NAME,
        caveat=TIER_CAVEATS[LabelTier.site_or_model_weak],
        notes=(
            f"sampled {stats.kept} of the {split} split via datasets-server at evenly spaced "
            "offsets; Task 2 essays only; band9_augmented (synthetic) excluded"
        ),
    )
    return record, stats
