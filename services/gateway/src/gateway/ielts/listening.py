"""
The IELTS Listening mock: the committed scripts and questions, marking in code, and the
raw-score-to-band table.

No model is called when a learner sits the test. The four recordings are the repository's own
original scripts (`data/ielts_listening.json`, written by an agent and NOT checked by a human; no
real IELTS material), turned into speech ONCE by `scripts/build_listening_audio.py` with Gemini
text-to-speech and stored as mp3 files next to the scripts. An attempt only plays those files. A
manifest records a fingerprint of the script each file was made from, so a test fails if a script
is edited without rebuilding its audio.

An answer is marked by a normalised comparison with the key or one of its listed alternatives
(case, punctuation, hyphens and extra spaces do not matter; spelling does, as in the real test).
The sentence that supports each key is shown, with the full transcript, after submission.

Two facts about the real test are used, each from a public page:

- Timing and format: Listening is about 30 minutes, four parts, 40 questions, and each recording
  is played once. Our paper keeps the 30-minute clock but its recordings are shorter.
  https://ielts.org/take-a-test/test-types/ielts-general-training-test/ielts-general-training-format-listening
- Raw score to band: IELTS publishes no single official table (the exact marks differ a little
  between test versions). The anchors for bands 5 to 9 come from IDP, an IELTS owner: the lowest
  raw score for each band is 16 (band 5), 23 (6), 30 (7), 35 (8) and 39 (9) out of 40.
  https://ielts.idp.com/bangladesh/results/scores/listening
  The half bands between them and the rows below band 5 are interpolated by us and the table
  says so (`BAND_TABLE_NOTE`).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from services.gateway.src.gateway.ielts.models import (
    ListeningReport,
    QuestionResult,
    SectionTranscript,
    Timing,
    TypeScore,
)
from services.gateway.src.gateway.ielts.reading import band_for, normalise, read_answers

__all__ = [
    "AUDIO_DIR", "BAND_TABLE", "MANIFEST_PATH", "ListeningSet", "Question", "Section",
    "audio_duration", "audio_path", "fingerprint", "get_set", "load_set", "mark", "read_answers",
    "tts_text",
]

DATA_DIR = Path(__file__).resolve().parent / "data"
LISTENING_PATH = DATA_DIR / "ielts_listening.json"
AUDIO_DIR = DATA_DIR / "listening_audio"
MANIFEST_PATH = DATA_DIR / "listening_audio.json"

ALLOWED_S = 30 * 60
TIMING_SOURCE_URL = (
    "https://ielts.org/take-a-test/test-types/ielts-general-training-test/"
    "ielts-general-training-format-listening"
)
BAND_SOURCE_URL = "https://ielts.idp.com/bangladesh/results/scores/listening"
BAND_TABLE_BASIS = 40

# Lowest raw score (out of 40) that earns each band, highest band first. The bands 5, 6, 7, 8 and
# 9 rows are IDP's published figures; every other row is interpolated.
BAND_TABLE: tuple[tuple[float, int], ...] = (
    (9.0, 39), (8.5, 37), (8.0, 35), (7.5, 32), (7.0, 30), (6.5, 26), (6.0, 23), (5.5, 18),
    (5.0, 16), (4.5, 13), (4.0, 10), (3.5, 8), (3.0, 6), (2.5, 4), (2.0, 3), (1.5, 2), (1.0, 1),
    (0.0, 0),
)
BAND_TABLE_NOTE = (
    "Band from a raw-to-band table for Listening (marks out of 40). The marks for bands 5, 6, 7, "
    "8 and 9 follow IDP's published figures; the half bands between them and the rows below band "
    "5 are our interpolation, and real tests vary by a mark or so. An estimate, not an official "
    "result."
)
SET_NOTE = (
    "This paper has four recordings and 40 questions, like the real test, but our recordings are "
    "shorter (about 9 minutes of audio in all) and speech is synthetic, so it is easier to "
    "follow than a real recording. In the real test each recording plays once; here you can "
    "replay it. The 30-minute clock is the real test's."
)
KEYS_NOTE = (
    "The scripts, questions and answer keys are written by the project's authors and have not "
    "been checked by a human, and the voices are text-to-speech. Each key shows the sentence it "
    "rests on, and the full transcript is shown after you submit, so you can judge for yourself."
)

# The quick sample test: the first recording only, with the questions heard early in it and one
# multiple-choice. Same items, keys, audio, marking and band table as the full paper.
SAMPLE_SECTION = "s1"
SAMPLE_QUESTION_IDS = (
    "ielts_listening_s1_q01", "ielts_listening_s1_q02", "ielts_listening_s1_q03",
    "ielts_listening_s1_q04", "ielts_listening_s1_q09",
)
SAMPLE_NOTE = (
    "SAMPLE: a quick test of 5 questions on the first recording, for checking the flow in a few "
    "minutes. The band below scales 5 questions to a 40-mark table, so it is NOT comparable to "
    "a real mock score. Take the full Listening mock for a real estimate."
)

TASK_LABELS = {
    "form_completion": "Form completion",
    "multiple_choice": "Multiple choice",
    "matching": "Matching",
    "sentence_completion": "Sentence completion",
    "note_completion": "Note completion",
}


@dataclass(frozen=True)
class Question:
    id: str
    number: int
    section_id: str
    task_type: str
    prompt: str
    instruction: str
    choices: dict[str, str]  # empty for completion
    key: str
    accept: tuple[str, ...]  # other spellings that are also right
    evidence: str


@dataclass(frozen=True)
class Speaker:
    name: str
    voice: str


@dataclass(frozen=True)
class Section:
    id: str
    number: int
    title: str
    context: str
    speakers: tuple[Speaker, ...]
    transcript: tuple[str, ...]


@dataclass(frozen=True)
class ListeningSet:
    id: str  # changes whenever a question or a script changes
    style: str
    sections: tuple[Section, ...]
    questions: tuple[Question, ...]


def tts_text(style: str, section: Section) -> str:
    """The exact text sent to the speech model for a section (also what its fingerprint covers)."""
    joiner = "\n" if len(section.speakers) > 1 else "\n\n"
    return style + "\n\n" + joiner.join(section.transcript)


def fingerprint(style: str, section: Section) -> str:
    """Identifies the script and voices a recording was made from."""
    payload = json.dumps(
        [tts_text(style, section), [(s.name, s.voice) for s in section.speakers]]
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_set(path: Path = LISTENING_PATH) -> ListeningSet:
    data = json.loads(path.read_text(encoding="utf-8"))
    style = str(data["style"])
    sections: list[Section] = []
    questions: list[Question] = []
    for number, raw in enumerate(data["sections"], start=1):
        sid = str(raw["id"])
        sections.append(Section(
            id=sid, number=number, title=str(raw["title"]), context=str(raw["context"]),
            speakers=tuple(Speaker(str(s["name"]), str(s["voice"])) for s in raw["speakers"]),
            transcript=tuple(str(line) for line in raw["transcript"]),
        ))
        for q in raw["questions"]:
            questions.append(Question(
                id=str(q["id"]), number=len(questions) + 1, section_id=sid,
                task_type=str(q["task_type"]), prompt=str(q["prompt"]),
                instruction=str(q.get("instruction", "")), choices=dict(q.get("choices") or {}),
                key=str(q["key"]), accept=tuple(str(a) for a in q.get("accept") or ()),
                evidence=str(q["evidence"]),
            ))
    digest = json.dumps([
        [(q.id, q.prompt, q.key, q.accept, sorted(q.choices.items())) for q in questions],
        [fingerprint(style, s) for s in sections],
    ]).encode()
    return ListeningSet(
        "listening-" + hashlib.sha256(digest).hexdigest()[:10], style, tuple(sections),
        tuple(questions),
    )


@lru_cache
def get_set() -> ListeningSet:
    return load_set()


def sample_of(paper: ListeningSet) -> ListeningSet:
    """The quick sample test: a subset of the paper's own items, renumbered from 1."""
    wanted = [q for q in paper.questions if q.id in SAMPLE_QUESTION_IDS]
    questions = tuple(replace(q, number=i) for i, q in enumerate(wanted, start=1))
    sections = tuple(s for s in paper.sections if s.id == SAMPLE_SECTION)
    return ListeningSet(paper.id + "-sample", paper.style, sections, questions)


@lru_cache
def get_sample_set() -> ListeningSet:
    return sample_of(get_set())


def is_correct(question: Question, answer: str) -> bool:
    given = normalise(answer)
    return bool(given) and given in {normalise(k) for k in (question.key, *question.accept)}


def _label(question: Question, value: str) -> str:
    """An answer as the learner sees it: a choice letter with its text, else the text itself."""
    if not value:
        return "(no answer)"
    for letter, text in question.choices.items():
        if normalise(letter) == normalise(value):
            return f"{letter}: {text}"
    return value


def mark(
    paper: ListeningSet, answers: dict[str, str], timing: Timing, sample: bool = False
) -> ListeningReport:
    """Mark the answers (question id to answer) against the keys. Pure code, no model."""
    titles = {s.id: s.title for s in paper.sections}
    results: list[QuestionResult] = []
    tallies: dict[str, list[int]] = {}
    for q in paper.questions:
        given = answers.get(q.id, "").strip()
        ok = is_correct(q, given)
        results.append(QuestionResult(
            number=q.number, id=q.id, passage=titles[q.section_id], task_type=q.task_type,
            task_label=TASK_LABELS.get(q.task_type, q.task_type), prompt=q.prompt,
            answer=given, answer_label=_label(q, given), key=q.key, key_label=_label(q, q.key),
            correct=ok, evidence=q.evidence,
        ))
        tally = tallies.setdefault(q.task_type, [0, 0])
        tally[0] += int(ok)
        tally[1] += 1
    raw = sum(r.correct for r in results)
    scaled, band = band_for(raw, len(results), BAND_TABLE)
    return ListeningReport(
        set_id=paper.id, raw=raw, total=len(results), scaled_40=scaled, band=band, timing=timing,
        by_type=[
            TypeScore(task_type=t, label=TASK_LABELS.get(t, t), correct=c, total=n)
            for t, (c, n) in tallies.items()
        ],
        questions=results,
        transcripts=[
            SectionTranscript(id=s.id, title=s.title, lines=list(s.transcript))
            for s in paper.sections
        ],
        band_note=BAND_TABLE_NOTE, set_note=SAMPLE_NOTE if sample else SET_NOTE,
        keys_note=KEYS_NOTE, band_source_url=BAND_SOURCE_URL, timing_source_url=TIMING_SOURCE_URL,
        sample=sample,
    )


def _manifest_entry(section_id: str) -> dict[str, Any] | None:
    try:
        entry = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["sections"][section_id]
    except (OSError, ValueError, KeyError):
        return None
    return entry if isinstance(entry, dict) else None


def audio_path(section_id: str) -> Path | None:
    """The built recording for a section, or None when it was never built."""
    entry = _manifest_entry(section_id)
    if entry is None:
        return None
    path = AUDIO_DIR / str(entry.get("file", ""))
    return path if path.is_file() else None


def audio_duration(section_id: str) -> float | None:
    entry = _manifest_entry(section_id)
    seconds = entry.get("duration_s") if entry else None
    return float(seconds) if isinstance(seconds, (int, float)) else None
