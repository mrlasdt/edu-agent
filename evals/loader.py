from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from evals.exam_set.local_store import local_dir, read_items
from evals.exam_set.schema import ExamItem

_GOLDEN_DIR = Path(__file__).parent / "golden"


def load_golden_set(suite: str) -> list[dict[str, Any]]:
    """Load all items from the golden set JSONL for the given suite ('quant' or 'aw')."""
    path = _GOLDEN_DIR / f"gre_{suite}.jsonl"
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def sample_golden_set(
    items: list[dict[str, Any]],
    sample_rate: float = 1.0,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Return a deterministic random sample of items at the given rate."""
    if sample_rate >= 1.0:
        return list(items)
    rng = random.Random(seed)
    k = max(1, round(len(items) * sample_rate))
    return rng.sample(items, min(k, len(items)))


# Committed IELTS/SAT families (original or CC BY items only). The GRE files above are untouched.
EXAM_FAMILIES = ("sat_math", "ielts_reading")


def load_exam_items(
    *,
    exam: str | None = None,
    skill: str | None = None,
    include_local: bool = False,
    local_directory: Path | str | None = None,
) -> list[ExamItem]:
    """
    Load IELTS/SAT golden items. Committed families are always included; items from the
    gitignored local directory (third-party data, see evals/exam_set/local_store.py) only
    when `include_local` is set, because their labels are weaker and their text is not ours.
    """
    items: list[ExamItem] = []
    for family in EXAM_FAMILIES:
        path = _GOLDEN_DIR / f"{family}.jsonl"
        if path.exists():
            items.extend(read_items(path))
    if include_local:
        directory = local_dir(local_directory)
        if directory.is_dir():
            for path in sorted(directory.glob("*.jsonl")):
                items.extend(read_items(path))
    return [
        i for i in items
        if (exam is None or i.exam == exam) and (skill is None or i.skill == skill)
    ]
