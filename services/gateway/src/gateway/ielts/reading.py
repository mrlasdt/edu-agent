"""
The IELTS Reading mock: the committed items, marking in code, and the raw-score-to-band table.

No model is called anywhere here. The items are the repository's own original passages (the
golden set `ielts_reading.jsonl`, keys written by an agent and NOT checked by a human, label tier
`author_key`), each with the sentence that supports its key. An answer is marked by a normalised
comparison with the key (case, punctuation, hyphens and extra spaces do not matter), and the
supporting sentence is shown after submission.

Two facts about the real test are used, each from a public page:

- Timing: General Training Reading is 60 minutes, three sections, 40 questions.
  https://ielts.org/take-a-test/test-types/ielts-general-training-test/ielts-general-training-format-reading
- Raw score to band: IELTS publishes no single official table (the exact marks differ a little
  between test versions). The anchors for bands 4 to 8 come from IDP, an IELTS owner, for General
  Training Reading: 15, 23, 30, 34 and 38 correct out of 40.
  https://ielts.idp.com/bangladesh/results/scores/reading
  The half bands between them are interpolated by us and the table says so (`BAND_TABLE_NOTE`).

This set has 41 questions on five short passages, not a full 40-question, three-section paper.
So a raw score out of 41 is first scaled to the 40-mark table, and the report says so.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from services.gateway.src.gateway.ielts.models import (
    QuestionResult,
    ReadingReport,
    Timing,
    TypeScore,
)
from shared.src.shared.jsonl import read_jsonl

READING_PATH = Path(__file__).resolve().parent / "data" / "ielts_reading.jsonl"

ALLOWED_S = 60 * 60
TIMING_SOURCE_URL = (
    "https://ielts.org/take-a-test/test-types/ielts-general-training-test/"
    "ielts-general-training-format-reading"
)
BAND_SOURCE_URL = "https://ielts.idp.com/bangladesh/results/scores/reading"
BAND_TABLE_BASIS = 40  # the table is for a 40-mark paper

# Lowest raw score (out of 40) that earns each band, highest band first. The bands 4, 5, 6, 7 and
# 8 rows are IDP's published General Training figures; every other row is interpolated.
BAND_TABLE: tuple[tuple[float, int], ...] = (
    (9.0, 40), (8.5, 39), (8.0, 38), (7.5, 36), (7.0, 34), (6.5, 32), (6.0, 30), (5.5, 27),
    (5.0, 23), (4.5, 19), (4.0, 15), (3.5, 12), (3.0, 9), (2.5, 6), (2.0, 3), (1.0, 1), (0.0, 0),
)
BAND_TABLE_NOTE = (
    "Band from a raw-to-band table for General Training Reading (marks out of 40). The marks for "
    "bands 4, 5, 6, 7 and 8 follow IDP's published figures; the half bands between them are our "
    "interpolation, and real tests vary by a mark or so. An estimate, not an official result."
)
SET_NOTE = (
    "This mock has 41 questions on five short passages, not a full 40-question, three-section "
    "paper, so your score is scaled to 40 before it is converted to a band."
)
KEYS_NOTE = (
    "The answer keys are written by the project's authors and have not been checked by a human, "
    "so a True / False / Not Given or other key may be arguable. Each key shows the sentence it "
    "rests on, so you can judge for yourself."
)

TASK_LABELS = {
    "tfng": "True / False / Not Given",
    "sentence_completion": "Sentence completion",
    "matching_headings": "Matching headings",
    "multiple_choice_single": "Multiple choice",
}
TFNG_CHOICES = {"TRUE": "True", "FALSE": "False", "NOT GIVEN": "Not given"}


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    paragraphs: tuple[str, ...]


@dataclass(frozen=True)
class Question:
    id: str
    number: int
    passage_id: str
    task_type: str
    prompt: str
    instruction: str
    choices: dict[str, str]  # empty for completion; the TFNG tokens for tfng
    key: str
    evidence: str
    paragraph: int | None = None


@dataclass(frozen=True)
class ReadingSet:
    id: str  # changes whenever an item changes, so an old attempt names the paper it sat
    passages: tuple[Passage, ...]
    questions: tuple[Question, ...]


def _title(passage_id: str) -> str:
    return passage_id.replace("_", " ").capitalize()


def load_set(path: Path = READING_PATH) -> ReadingSet:
    rows = read_jsonl(path)
    passages: dict[str, Passage] = {}
    questions: list[Question] = []
    for row in rows:
        extras = row.get("extras", {})
        pid = str(extras["passage_id"])
        if pid not in passages:
            paragraphs = tuple(p for p in str(row["passage"]).split("\n\n") if p.strip())
            passages[pid] = Passage(pid, _title(pid), paragraphs)
        task = str(row["task_type"])
        choices = dict(row.get("choices") or {})
        if task == "tfng":
            choices = dict(TFNG_CHOICES)
        questions.append(Question(
            id=str(row["id"]), number=len(questions) + 1, passage_id=pid, task_type=task,
            prompt=str(row["prompt"]), instruction=str(extras.get("instruction", "")),
            choices=choices, key=str(row["ground_truth"]), evidence=str(row.get("evidence") or ""),
            paragraph=extras.get("paragraph"),
        ))
    digest = json.dumps(
        [(q.id, q.prompt, q.key, sorted(q.choices.items())) for q in questions]
    ).encode()
    set_id = "reading-" + hashlib.sha256(digest).hexdigest()[:10]
    return ReadingSet(set_id, tuple(passages.values()), tuple(questions))


@lru_cache
def get_set() -> ReadingSet:
    return load_set()


_NOT_WORD = re.compile(r"[^\w\s]")


def normalise(text: str) -> str:
    """Case, accents' compatibility forms, punctuation, hyphens and spacing do not matter."""
    flat = unicodedata.normalize("NFKC", text).casefold().replace("-", " ").replace("_", " ")
    return " ".join(_NOT_WORD.sub(" ", flat).split())


def is_correct(question: Question, answer: str) -> bool:
    given = normalise(answer)
    return bool(given) and given == normalise(question.key)


def band_for(
    raw: int, total: int, table: tuple[tuple[float, int], ...] = BAND_TABLE
) -> tuple[int, float]:
    """(the raw score scaled to the 40-mark table, the band it earns)."""
    scaled = 0 if total <= 0 else min(
        BAND_TABLE_BASIS, int(raw * BAND_TABLE_BASIS / total + 0.5)
    )
    for band, lowest in table:
        if scaled >= lowest:
            return scaled, band
    return scaled, 0.0


def answer_label(question: Question, value: str) -> str:
    """An answer as the learner sees it: a choice letter with its text, else the text itself."""
    if not value:
        return "(no answer)"
    for letter, text in question.choices.items():
        if normalise(letter) == normalise(value) and question.task_type != "tfng":
            return f"{letter}: {text}"
    return value


def read_answers(text: str) -> dict[str, str]:
    """The stored answers (a JSON object of question id to answer); anything else is no answers."""
    try:
        data = json.loads(text) if text else {}
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}



def mark(
    paper: ReadingSet, answers: dict[str, str], timing: Timing
) -> ReadingReport:
    """Mark the answers (question id to answer) against the keys. Pure code, no model."""
    titles = {p.id: p.title for p in paper.passages}
    results: list[QuestionResult] = []
    tallies: dict[str, list[int]] = {}
    for q in paper.questions:
        given = answers.get(q.id, "").strip()
        ok = is_correct(q, given)
        results.append(QuestionResult(
            number=q.number, id=q.id, passage=titles[q.passage_id], task_type=q.task_type,
            task_label=TASK_LABELS.get(q.task_type, q.task_type), prompt=q.prompt,
            paragraph=q.paragraph, answer=given, answer_label=answer_label(q, given), key=q.key,
            key_label=answer_label(q, q.key), correct=ok, evidence=q.evidence,
        ))
        tally = tallies.setdefault(q.task_type, [0, 0])
        tally[0] += int(ok)
        tally[1] += 1
    raw = sum(r.correct for r in results)
    scaled, band = band_for(raw, len(results))
    return ReadingReport(
        set_id=paper.id, raw=raw, total=len(results), scaled_40=scaled, band=band, timing=timing,
        by_type=[
            TypeScore(task_type=t, label=TASK_LABELS.get(t, t), correct=c, total=n)
            for t, (c, n) in tallies.items()
        ],
        questions=results, band_note=BAND_TABLE_NOTE, set_note=SET_NOTE, keys_note=KEYS_NOTE,
        band_source_url=BAND_SOURCE_URL, timing_source_url=TIMING_SOURCE_URL,
    )
