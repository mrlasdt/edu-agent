"""
Importer for the captain's own Cambridge IELTS practice-book material (personal use).

Input is a local folder of `.txt`/`.md` text, text PDFs, or scans (scanned PDFs and
`.png`/`.jpg`/`.tif` images), which are read with local OCR (the `tesseract` binary; nothing
is uploaded anywhere). The extracted text is written next to the items so OCR slips can be
fixed by hand and the folder re-imported as plain text.

It extracts Writing tasks with the books' sample answers, their published band scores and the
examiner comments, as label tier `examiner` (the book's published band, not a re-rating). The
text is copyrighted: items go to the gitignored local directory only.

Recognised layout (headings are matched case-insensitively, one per line):

  WRITING TASK 1 | WRITING TASK 2     starts a task; the prompt runs to "Write at least N words"
  Sample answer | Model answer | Candidate's answer | Sample script ...   (a band may follow,
                                      e.g. "Sample answer - Band 7" or "(Band 6.5)")
  Examiner comments | Examiner's comments    comment text; may hold "Band 7" or per-criterion
                                      lines such as "Lexical Resource: 7"
  TEST 3                              optional, recorded in the item id when seen

A sample answer with no recoverable band is skipped and counted: it cannot be labelled.
This parser is built and tested on synthetic fixtures only. Real book layouts vary, so run it
on one book first and read `<local>/cambridge_extracted/` plus the printed counts before
trusting it; fix the extracted text rather than the labels.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

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

SOURCE_ID = "cambridge_ielts_books"
FILE_NAME = "ielts_cambridge_books.jsonl"
EXTRACT_DIR = "cambridge_extracted"
LICENCE_STATUS = (
    "copyrighted (Cambridge University Press & Assessment); the captain's own copies, personal "
    "local use only, never committed"
)
TEXT_SUFFIXES = {".txt", ".md"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
OcrFn = Callable[[bytes], str]
MIN_PAGE_CHARS = 40  # a PDF page with less text than this is treated as a scan


class OcrUnavailableError(RuntimeError):
    pass


def tesseract_ocr(image: bytes) -> str:
    """Local OCR through the `tesseract` CLI."""
    binary = shutil.which("tesseract")
    if binary is None:
        raise OcrUnavailableError(
            "scanned input needs local OCR: install tesseract (e.g. `brew install tesseract`) "
            "or supply text/text-PDF files"
        )
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "page"
        src.write_bytes(image)
        done = subprocess.run(  # noqa: S603 - fixed argv, local file
            [binary, str(src), "stdout", "-l", "eng"],
            capture_output=True, text=True, check=True, timeout=120,
        )
    return done.stdout


def _page_images(page: object) -> list[bytes]:
    """Embedded images of a pypdf page (a scanned page is usually one full-page image)."""
    images = getattr(page, "images", [])
    return [bytes(img.data) for img in images]


def extract_text(path: Path, ocr: OcrFn = tesseract_ocr) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in IMAGE_SUFFIXES:
        return ocr(path.read_bytes())
    if suffix == ".pdf":
        from pypdf import PdfReader

        pages: list[str] = []
        for page in PdfReader(str(path)).pages:
            text = (page.extract_text() or "").strip()
            if len(text) < MIN_PAGE_CHARS:
                scanned = "\n".join(ocr(img) for img in _page_images(page))
                text = scanned.strip() or text
            pages.append(text)
        return "\n\n".join(pages)
    raise ValueError(f"unsupported file type: {path.name}")


# ── parsing ──────────────────────────────────────────────────────────────────

_TASK = re.compile(r"^\W*writing\s+task\s+([12])\b", re.I)
_TEST = re.compile(r"^\W*test\s+(\d+)\W*$", re.I)
_SAMPLE = re.compile(
    r"^\W*(?:sample|model)\s+(?:answer|script|response)s?\b|"
    r"^\W*candidate'?s?\s+(?:answer|script|response)\b",
    re.I,
)
_COMMENT = re.compile(r"^\W*examiner'?s?\s+comments?\b", re.I)
_WORDS = re.compile(r"write\s+at\s+least\s+\d+\s+words", re.I)
_BOILERPLATE = re.compile(r"^\W*you\s+should\s+spend\s+about\s+\d+\s+minutes", re.I)
_BAND = re.compile(r"band(?:\s+score)?(?:\s+of)?\s*[:\-]?\s*(\d(?:\.[05])?)\b", re.I)
_CRITERION_LINE = re.compile(
    r"^\W*(task\s+(?:response|achievement)|coherence\s+and\s+cohesion|lexical\s+resource|"
    r"grammatical\s+range\s+and\s+accuracy)\s*[:\-]?\s*(?:band\s+)?(\d(?:\.[05])?)\b",
    re.I | re.M,
)
_CRITERION_KEY = {"task": "TR", "coherence": "CC", "lexical": "LR", "grammatical": "GRA"}


def _band(text: str) -> float | None:
    for match in _BAND.finditer(text):
        value = float(match.group(1))
        if 0.0 <= value <= 9.0 and (value * 2) % 1 == 0:
            return value
    return None


def _criteria(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for match in _CRITERION_LINE.finditer(text):
        key = _CRITERION_KEY[match.group(1).split()[0].lower()]
        value = float(match.group(2))
        if 0.0 <= value <= 9.0 and (value * 2) % 1 == 0:
            out[key] = value
    return out


@dataclass
class _Sample:
    heading: str
    lines: list[str] = field(default_factory=list)
    comment: list[str] = field(default_factory=list)


@dataclass
class _Task:
    number: int
    test: int | None
    prompt: list[str] = field(default_factory=list)
    prompt_done: bool = False
    samples: list[_Sample] = field(default_factory=list)


@dataclass
class ParseResult:
    items: list[ExamItem] = field(default_factory=list)
    skipped_no_band: int = 0
    tasks_seen: int = 0


def _tasks(text: str) -> list[_Task]:
    tasks: list[_Task] = []
    test: int | None = None
    current: _Task | None = None
    sample: _Sample | None = None
    in_comment = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if (m := _TEST.match(line.strip())) is not None:
            test, current, sample = int(m.group(1)), None, None
            continue
        if (m := _TASK.match(line)) is not None:
            current = _Task(number=int(m.group(1)), test=test)
            tasks.append(current)
            sample, in_comment = None, False
            continue
        if current is None:
            continue
        if _SAMPLE.match(line):
            current.prompt_done = True
            sample = _Sample(heading=line)
            current.samples.append(sample)
            in_comment = False
            continue
        if _COMMENT.match(line) and sample is not None:
            in_comment = True
            sample.comment.append(re.sub(_COMMENT, "", line, count=1).strip(" :-"))
            continue
        if sample is not None:
            (sample.comment if in_comment else sample.lines).append(line)
        elif not current.prompt_done:
            current.prompt.append(line)
            if _WORDS.search(line):
                current.prompt_done = True
    return tasks


def _clean(lines: list[str]) -> str:
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _prompt_text(lines: list[str]) -> str:
    return _clean([ln for ln in lines if not _BOILERPLATE.match(ln)])


def parse_book_text(text: str, book: str) -> ParseResult:
    """Extract labelled writing items from one book's text. `book` names the book in ids."""
    result = ParseResult()
    slug = re.sub(r"[^a-z0-9]+", "_", book.lower()).strip("_")
    for task in _tasks(text):
        result.tasks_seen += 1
        prompt = _prompt_text(task.prompt)
        for n, sample in enumerate(task.samples, start=1):
            essay = _clean(sample.lines)
            comment = _clean(sample.comment)
            overall = _band(sample.heading) or _band(comment)
            if not prompt or not essay or overall is None:
                result.skipped_no_band += 1
                continue
            criteria = _criteria(comment)
            where = f"t{task.test}_" if task.test is not None else ""
            result.items.append(
                ExamItem(
                    id=f"cam_{slug}_{where}w{task.number}_{n}",
                    exam="ielts",
                    skill="writing_t2" if task.number == 2 else "writing_t1",
                    task_type="essay_score",
                    prompt=prompt,
                    response=essay,
                    labels=Labels(
                        overall=overall,
                        criteria=criteria if len(criteria) == 4 else {},
                    ),
                    label_provenance=LabelProvenance(
                        tier=LabelTier.examiner,
                        note=TIER_CAVEATS[LabelTier.examiner],
                    ),
                    source=Source(
                        kind="third_party_local",
                        license="copyrighted; owner's personal copy",
                        source_id=SOURCE_ID,
                        attribution=f"Cambridge IELTS practice book: {book}",
                    ),
                    design=Design(generator="book_extract"),
                    split="calibration",
                    examiner_comment=comment or None,
                    extras={
                        "book": book,
                        **({"task1_visual_not_extracted": True} if task.number == 1 else {}),
                    },
                )
            )
    return result


def import_folder(
    folder: Path, directory: Path, ocr: OcrFn = tesseract_ocr
) -> tuple[SourceRecord, dict[str, int]]:
    """Read every supported file in `folder` and register the extracted items locally."""
    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
    items: list[ExamItem] = []
    totals = {"files": 0, "tasks_seen": 0, "skipped_no_band": 0, "items": 0, "unsupported": 0}
    extract_dir = directory / EXTRACT_DIR
    extract_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES | IMAGE_SUFFIXES | {".pdf"}:
            totals["unsupported"] += 1
            continue
        text = extract_text(path, ocr)
        (extract_dir / f"{path.stem}.txt").write_text(text, encoding="utf-8")
        parsed = parse_book_text(text, path.stem)
        items.extend(parsed.items)
        totals["files"] += 1
        totals["tasks_seen"] += parsed.tasks_seen
        totals["skipped_no_band"] += parsed.skipped_no_band
    totals["items"] = len(items)
    record = register_source(
        directory,
        source_id=SOURCE_ID,
        name="Cambridge IELTS practice books (captain's copies)",
        url="local folder (not published)",
        licence_status=LICENCE_STATUS,
        items=items,
        file_name=FILE_NAME,
        caveat=TIER_CAVEATS[LabelTier.examiner],
        notes=(
            f"{totals['files']} files, {totals['tasks_seen']} writing tasks seen, "
            f"{totals['skipped_no_band']} samples skipped (no recoverable band)"
        ),
    )
    return record, totals
