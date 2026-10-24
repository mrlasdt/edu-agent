"""
On-disk response cache for real-model eval calls.

Keyed by a hash of every input that can change a response (model, messages, sampling
parameters), so a rerun of the same plan re-pays nothing and re-hits no rate limit. Only
successful responses are stored. The directory (default `.eval-cache/`) is gitignored and
holds model outputs, so it must never be committed or shared.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path(".eval-cache")


def cache_key(**inputs: Any) -> str:
    blob = json.dumps(inputs, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            value: dict[str, Any] = json.loads(self._path(key).read_text())
        except (OSError, ValueError):
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(self._path(key))
