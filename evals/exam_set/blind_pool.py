"""
The local blind-scoring pool for the IELTS-style Writing mock.

The grader's accuracy can only be certified against bands a person gave without seeing the
grader's output (ADR-0014, report section 4.2). This builds the material for that: essays
drafted by a model at a spread of target bands plus the instructor-graded essays already in
the local golden directory, shuffled under opaque ids. The owner scores `essays.md` into
`scores.csv`; `import-blind-scores` turns the sheet into `captain_blind` golden items.

Everything lives under `ielts-local/blind/` (gitignored). `key.jsonl` records where each essay
came from and must not be opened until scoring is finished - the file name says so, and the
essay file carries nothing but opaque ids, prompts and text.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from evals.exam_set.schema import (
    CRITERIA,
    Design,
    ExamItem,
    LabelProvenance,
    Labels,
    LabelTier,
    Source,
)
from services.agent.src.agent.ielts.scoring import round_overall

BLIND_DIR = Path("ielts-local/blind")
ESSAYS_MD = "essays.md"
ESSAYS_JSONL = "essays.jsonl"
KEY_FILE = "key-do-not-open-until-scored.jsonl"
SCORE_SHEET = "scores.csv"
IMPORTED_FILE = "ielts_captain_blind.jsonl"  # written into the local golden directory
TARGET_BANDS = (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0)

# What an essay at each level typically looks like, so the drafts spread across the scale
# instead of all sounding like a fluent model. The owner's scores, not these targets, are labels.
_PROFILES: dict[float, str] = {
    4.0: "Barely answers the question; ideas are few and repeated; frequent basic grammar "
         "errors that often block meaning; very limited vocabulary with wrong word choices; "
         "paragraphing is unclear. About 180 words.",
    4.5: "Partly answers the question; position is unclear; ideas are undeveloped; many "
         "grammar errors, some blocking meaning; limited vocabulary; mechanical linking. "
         "About 200 words.",
    5.0: "Addresses the task only in part; a position appears but is not always clear; ideas "
         "are simple and thinly supported; frequent errors in grammar and spelling that rarely "
         "block meaning; basic vocabulary; linking words overused. About 230 words.",
    5.5: "Addresses the task but unevenly; some ideas lack support; mix of simple and a few "
         "complex sentences with regular errors; vocabulary adequate but repetitive; "
         "paragraphs present but linking is mechanical. About 250 words.",
    6.0: "Answers all parts, though some ideas are underdeveloped; position generally clear; "
         "some complex sentences with errors that seldom block meaning; adequate vocabulary "
         "with occasional wrong collocations; clear paragraphs. About 265 words.",
    6.5: "Answers all parts with a clear position; ideas extended but sometimes not fully "
         "supported; complex sentences with some errors; reasonable vocabulary range with "
         "occasional awkward word choice; logical organisation. About 275 words.",
    7.0: "Fully answers the task with a clear, developed position; well-supported main ideas "
         "with a little over-generalising; frequent error-free sentences, a few slips; good "
         "vocabulary with some uncommon items used slightly unnaturally; clear progression. "
         "About 285 words.",
    7.5: "Well-developed, relevant ideas; position consistent; mostly error-free sentences "
         "with a range of structures; wide vocabulary used naturally with rare slips; "
         "cohesion handled well with only small lapses. About 290 words.",
    8.0: "Sufficiently developed answer to all parts; very few errors, mostly slips; wide, "
         "natural vocabulary; skilful paragraphing and cohesion; only minor lapses in "
         "precision. About 300 words.",
}


def draft_request(prompt: str, target: float) -> list[dict[str, str]]:
    """Messages asking a model for one learner essay at a target band."""
    return [
        {"role": "system", "content": (
            "You write realistic practice responses for an IELTS-style Writing Task 2, as a "
            "learner at a stated level would write them, errors included. Output only the "
            "essay text: no title, no band, no commentary. Separate paragraphs with a blank "
            "line. Never mention that the text is a sample.")},
        {"role": "user", "content": (
            f"Question:\n{prompt}\n\nLevel profile (band {target}):\n{_PROFILES[target]}\n\n"
            "Write the essay now. Make the errors natural for this level; do not label them.")},
    ]


@dataclass
class Draft:
    prompt_id: str
    prompt: str
    target: float
    essay: str


Completion = Callable[[list[dict[str, str]]], Awaitable[str]]


async def draft_essays(
    prompts: Sequence[tuple[str, str]], n: int, complete: Completion
) -> list[Draft]:
    """`n` drafts spread evenly over TARGET_BANDS, cycling through the prompts."""
    plan = [
        (TARGET_BANDS[i % len(TARGET_BANDS)],
         prompts[(i * 3 + i // len(TARGET_BANDS)) % len(prompts)])
        for i in range(n)
    ]
    drafts: list[Draft] = []
    for target, (prompt_id, prompt) in plan:
        essay = (await complete(draft_request(prompt, target))).strip()
        if len(essay.split()) >= 50:
            drafts.append(Draft(prompt_id, prompt, target, essay))
    return drafts


def _opaque_id(seed: str, index: int) -> str:
    return "b" + hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()[:7]


def build_pool(
    out_dir: Path, drafts: Sequence[Draft], teacher_items: Sequence[ExamItem], *, seed: int = 42
) -> dict[str, int]:
    """Write the shuffled blind pool: essays (md + jsonl), the key, and an empty score sheet."""
    entries: list[tuple[str, str, dict[str, object]]] = []
    for d in drafts:
        entries.append((d.prompt, d.essay, {
            "origin": "llm_draft", "target_band": d.target, "prompt_id": d.prompt_id}))
    for item in teacher_items:
        assert item.response is not None and item.labels is not None
        entries.append((item.prompt, item.response, {
            "origin": "teacher_graded", "source_item": item.id,
            "teacher_overall": item.labels.overall, "teacher_criteria": item.labels.criteria}))
    random.Random(seed).shuffle(entries)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [(_opaque_id(str(seed), i), p, e, k) for i, (p, e, k) in enumerate(entries)]
    (out_dir / ESSAYS_JSONL).write_text("".join(
        json.dumps({"id": i, "prompt": p, "essay": e}) + "\n" for i, p, e, _ in rows))
    (out_dir / KEY_FILE).write_text("".join(
        json.dumps({"id": i, **k}) + "\n" for i, _, _, k in rows))
    (out_dir / ESSAYS_MD).write_text(_render_markdown(rows))
    sheet = out_dir / SCORE_SHEET
    if not sheet.exists():  # never overwrite the owner's scores
        with sheet.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", *CRITERIA, "notes"])
            writer.writerows([[i, "", "", "", "", ""] for i, *_ in rows])
    return {"essays": len(rows), "llm_drafts": len(drafts), "teacher_graded": len(teacher_items)}


def _render_markdown(rows: Sequence[tuple[str, str, str, dict[str, object]]]) -> str:
    out = [
        "# Blind scoring pool",
        "",
        "Score each essay below on the four IELTS Writing Task 2 criteria (half bands, 0 to 9)",
        f"and enter them in `{SCORE_SHEET}`. Do not open `{KEY_FILE}` and do not run the",
        "grader on these essays until you have finished: it would remove the blindness.",
        "Score from the official band descriptors, one criterion at a time.",
        "",
    ]
    for n, (essay_id, prompt, essay, _) in enumerate(rows, 1):
        out += [f"## {n}. `{essay_id}`", "", f"**Question:** {prompt}", "", essay, "", "---", ""]
    return "\n".join(out)


class ScoreSheetError(ValueError):
    pass


def _half_band(raw: str, essay_id: str, criterion: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ScoreSheetError(f"{essay_id}: {criterion} is not a number: {raw!r}") from exc
    if not 0.0 <= value <= 9.0 or (value * 2) % 1 != 0:
        raise ScoreSheetError(f"{essay_id}: {criterion} must be a half band 0..9, got {raw!r}")
    return value


def _split(essay_id: str) -> str:
    """Stable dev/test split by id hash: roughly half each, fixed before the grader is tuned."""
    return "test" if int(hashlib.sha256(essay_id.encode()).hexdigest(), 16) % 2 else "dev"


def import_scores(blind_dir: Path, today: str) -> tuple[list[ExamItem], int]:
    """Items for every fully scored row of the sheet, and how many rows are still empty."""
    essays = {
        r["id"]: r for r in (json.loads(x) for x in
                             (blind_dir / ESSAYS_JSONL).read_text().splitlines() if x.strip())
    }
    items: list[ExamItem] = []
    empty = 0
    with (blind_dir / SCORE_SHEET).open(newline="") as f:
        for row in csv.DictReader(f):
            essay_id = row["id"]
            raw = [(row.get(c) or "").strip() for c in CRITERIA]
            if not any(raw):
                empty += 1
                continue
            if not all(raw):
                raise ScoreSheetError(f"{essay_id}: all four criteria are needed (TR CC LR GRA)")
            if essay_id not in essays:
                raise ScoreSheetError(f"{essay_id}: not in the pool")
            bands = {c: _half_band(r, essay_id, c) for c, r in zip(CRITERIA, raw, strict=True)}
            overall = round_overall(bands)
            items.append(ExamItem(
                id=f"ielts_blind_{essay_id}", exam="ielts", skill="writing_t2",
                task_type="essay_score", prompt=essays[essay_id]["prompt"],
                response=essays[essay_id]["essay"],
                labels=Labels(overall=overall, criteria=bands),
                label_provenance=LabelProvenance(
                    tier=LabelTier.captain_blind, n_raters=1, blind_to_target=True, date=today,
                    note="scored by the project owner from the band descriptors before seeing "
                         "any grader output; single rater, not an examiner",
                ),
                source=Source(kind="third_party_local", license="local use; pool may hold "
                              "third-party essays"),
                design=Design(generator="blind_pool"),
                split=_split(essay_id),  # type: ignore[arg-type]
            ))
    return items, empty
