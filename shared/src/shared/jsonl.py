"""Reading JSON-lines files: the one reader the gateway's prompt set and the evals share."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """The rows of a JSON-lines file, one JSON object per line; blank lines are skipped."""
    # split("\n"), not splitlines(): a row may hold a raw U+2028 inside a string, which is valid
    # JSON and must not end the row.
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]
