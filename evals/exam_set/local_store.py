"""
Gitignored local directory for third-party golden data, plus its manifest.

edu-agent is a public repo: scraped or copyrighted text is written only under
`evals/golden/local/` (ignored by git) and each source is recorded in
`manifest.json` there: where it came from, its licence status, its label tier and counts.
Committed code holds importers, schema, templates and tiny synthetic test fixtures only.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evals.exam_set.schema import ExamItem

DEFAULT_LOCAL_DIR = Path(__file__).resolve().parents[1] / "golden" / "local"
LOCAL_DIR_ENV = "EDU_GOLDEN_LOCAL_DIR"
MANIFEST_NAME = "manifest.json"


def local_dir(override: Path | str | None = None) -> Path:
    if override is not None:
        return Path(override)
    env = os.environ.get(LOCAL_DIR_ENV)
    return Path(env) if env else DEFAULT_LOCAL_DIR


@dataclass
class SourceRecord:
    source_id: str
    name: str
    url: str
    licence_status: str
    label_tier: str
    file: str
    count: int
    sha256: str
    imported_at: str
    caveat: str = ""
    counts_by_overall_band: dict[str, int] = field(default_factory=dict)
    notes: str = ""


def write_items(path: Path, items: Iterable[ExamItem]) -> tuple[int, str]:
    """Write items as JSONL; returns (count, sha256 of the file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [item.model_dump_json(exclude_none=True) for item in items]
    body = "\n".join(lines) + ("\n" if lines else "")
    tmp = path.with_name(path.name + ".tmp")  # replace atomically: readers never see a half file
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
    return len(lines), hashlib.sha256(body.encode("utf-8")).hexdigest()


def read_items(path: Path) -> list[ExamItem]:
    out: list[ExamItem] = []
    # split on "\n" only: essays may hold U+2028/U+0085, which str.splitlines would cut at
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            out.append(ExamItem.model_validate_json(line))
    return out


def band_histogram(items: Iterable[ExamItem]) -> dict[str, int]:
    counts = Counter(
        f"{i.labels.overall:g}" for i in items if i.labels and i.labels.overall is not None
    )
    return dict(sorted(counts.items(), key=lambda kv: float(kv[0])))


def load_manifest(directory: Path) -> dict[str, Any]:
    path = directory / MANIFEST_NAME
    if not path.exists():
        return {"note": "local third-party data; gitignored, never commit", "sources": []}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def register_source(
    directory: Path,
    *,
    source_id: str,
    name: str,
    url: str,
    licence_status: str,
    items: list[ExamItem],
    file_name: str,
    caveat: str = "",
    notes: str = "",
) -> SourceRecord:
    """Write `items` to `<directory>/<file_name>` and upsert the manifest entry for the source."""
    count, digest = write_items(directory / file_name, items)
    tiers = {i.label_provenance.tier.value for i in items}
    record = SourceRecord(
        source_id=source_id,
        name=name,
        url=url,
        licence_status=licence_status,
        label_tier=",".join(sorted(tiers)) or "none",
        file=file_name,
        count=count,
        sha256=digest,
        imported_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        caveat=caveat,
        counts_by_overall_band=band_histogram(items),
        notes=notes,
    )
    manifest = load_manifest(directory)
    manifest["sources"] = [s for s in manifest["sources"] if s["source_id"] != source_id]
    manifest["sources"].append(asdict(record))
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return record
