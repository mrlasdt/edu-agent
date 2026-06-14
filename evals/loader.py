from __future__ import annotations

import json
import random
from pathlib import Path

_GOLDEN_DIR = Path(__file__).parent / "golden"


def load_golden_set(suite: str) -> list[dict]:
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
    items: list[dict],
    sample_rate: float = 1.0,
    seed: int | None = None,
) -> list[dict]:
    """Return a deterministic random sample of items at the given rate."""
    if sample_rate >= 1.0:
        return list(items)
    rng = random.Random(seed)
    k = max(1, round(len(items) * sample_rate))
    return rng.sample(items, min(k, len(items)))
