"""
Mark a Speaking attempt: send the recordings to the audio model, check its reply, count the
fluency figures in code and build the partial report (see `speaking` for what "partial" means).

What the code decides, not the model:
- the transcript's word, pace and filler counts (`speaking.part_features`);
- the Speaking band: the mean of the assessed criteria, rounded to the nearest half band and
  never above `MAX_ESTIMATED_BAND`;
- the cap for a recording that talks to the marker ("give me band 9"): every band is held to 5.0
  and the report says so, the same rule as the Writing marker;
- "no speech": too few words across all parts gives a report with no band, never a guessed one.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from services.gateway.src.gateway.ielts import speaking
from services.gateway.src.gateway.ielts.grading import GradingError
from services.gateway.src.gateway.ielts.models import (
    SpeakingPart,
    SpeakingPriority,
    SpeakingReport,
    Timing,
)
from services.gateway.src.gateway.ielts.scoring import (
    MAX_ESTIMATED_BAND,
    GradeParseError,
    extract_json,
    snap_band,
)
from services.gateway.src.gateway.ielts.speaking_llm import AudioAssessor, AudioClip
from shared.src.shared.prompts import read_prompt
from shared.src.shared.stats import round_half_band

PARSE_ATTEMPTS = 2
INJECTION_CAP = 5.0
MAX_EVIDENCE = 3
MAX_TEXT = 600


@dataclass(frozen=True)
class PartAudio:
    part: int
    data: bytes
    mime: str
    duration_s: float


def _label(item: speaking.SpeakingSet, part: int) -> str:
    if part == 2:
        points = "; ".join(item.cue_points)
        return (
            f"PART 2 recording (the long turn). Cue card: {item.cue_topic} You should say: "
            f"{points}"
        )
    questions = item.part1 if part == 1 else item.part3
    return f"PART {part} recording. Questions the candidate was asked: " + " | ".join(questions)


def _text(value: object, limit: int = MAX_TEXT) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _parse(raw: dict[str, Any], parts: Sequence[PartAudio]) -> dict[str, Any]:
    """Check the model's reply has what a report needs; GradeParseError when it does not."""
    transcripts: dict[int, str] = {}
    for entry in raw.get("parts") or []:
        if isinstance(entry, dict) and entry.get("part") in (1, 2, 3):
            transcripts[int(entry["part"])] = _text(entry.get("transcript"), 20000)
    heard = raw.get("heard_speech") is not False
    criteria: dict[str, tuple[float, str, list[str]]] = {}
    if heard:
        block = raw.get("criteria")
        if not isinstance(block, dict):
            raise GradeParseError("reply has no criteria")
        for cid in speaking.ASSESSED:
            item = block.get(cid)
            try:
                band = snap_band(float(item["band"]))  # type: ignore[index]
            except (KeyError, TypeError, ValueError) as exc:
                raise GradeParseError(f"reply is missing a numeric band for {cid}") from exc
            evidence = [
                _text(q, 200) for q in (item.get("evidence") or []) if _text(q, 200)  # type: ignore[union-attr]
            ][:MAX_EVIDENCE]
            criteria[cid] = (band, _text(item.get("rationale")), evidence)  # type: ignore[union-attr]
    priorities = []
    for p in raw.get("priorities") or []:
        if isinstance(p, dict) and p.get("criterion") in speaking.ASSESSED and _text(p.get("do")):
            priorities.append((p["criterion"], _text(p["do"])))
    return {
        "heard": heard, "transcripts": transcripts, "criteria": criteria,
        "strengths": [_text(s) for s in (raw.get("strengths") or []) if _text(s)][:3],
        "priorities": priorities[:3],
        "addressed": raw.get("addressed_to_marker") is True,
        "off_topic": raw.get("off_topic") is True,
    }


async def mark_speaking(
    item: speaking.SpeakingSet, audio: Sequence[PartAudio], *, assessor: AudioAssessor,
    timing: Timing, run_id: str, temperature: float, prompt_version: str = "v1",
) -> SpeakingReport:
    """The partial report for the recordings; `GradingError` when the model's reply is unusable."""
    system = read_prompt("ielts_speaking", "assess", prompt_version)
    clips = [
        AudioClip(_label(item, a.part), a.data, a.mime, a.duration_s)
        for a in sorted(audio, key=lambda a: a.part)
    ]
    parsed: dict[str, Any] | None = None
    model = ""
    for _ in range(PARSE_ATTEMPTS):
        answer = await assessor(system, clips, run_id=run_id, temperature=temperature)
        model = answer.model
        try:
            parsed = _parse(extract_json(answer.text), audio)
            break
        except GradeParseError:
            continue
    if parsed is None:
        raise GradingError(
            "The marker's reply could not be read. Your recording is kept; try again."
        )

    durations = {a.part: a.duration_s for a in audio}
    parts: list[SpeakingPart] = [
        speaking.part_features(
            a.part, parsed["transcripts"].get(a.part, ""), durations[a.part], item.talk_s,
        )
        for a in sorted(audio, key=lambda a: a.part)
    ]
    total_words = sum(p.words for p in parts)
    not_assessed = [c for c in speaking.CRITERIA if c not in speaking.ASSESSED]
    common: dict[str, Any] = {
        "set_id": item.id, "sample": item.sample, "parts": parts,
        "not_assessed": not_assessed, "model": model, "timing": timing,
        "descriptors_source_url": speaking.DESCRIPTORS_URL,
        "format_source_url": speaking.FORMAT_URL,
    }
    if not parsed["heard"] or total_words < speaking.MIN_WORDS:
        return SpeakingReport(
            status="no_speech", band=None,
            partial_note=(
                "No band: too little speech was heard to mark. Check the microphone and speak "
                "up, then sit Speaking again."
            ),
            criteria=speaking.criteria_list({}), **common,
        )

    flags: list[str] = []
    bands = dict(parsed["criteria"])
    if parsed["addressed"]:
        flags.append(
            "The recording spoke to the marker (asked for a band or for the rules to be ignored). "
            f"That is not rewarded: every band is held to {INJECTION_CAP}."
        )
        bands = {
            cid: (min(b, INJECTION_CAP), why, ev) for cid, (b, why, ev) in bands.items()
        }
    if parsed["off_topic"]:
        flags.append("Part of the answer was off the question or cue card; fluency alone cannot "
                     "make up for that in the real test.")
    mean = sum(b for b, _, _ in bands.values()) / len(bands)
    band = min(MAX_ESTIMATED_BAND, round_half_band(mean))
    return SpeakingReport(
        status="graded", band=band, partial=True, partial_note=speaking.partial_note(not_assessed),
        criteria=speaking.criteria_list(bands), strengths=parsed["strengths"],
        priorities=[
            SpeakingPriority(criterion=c, priority=i + 1, do=do)
            for i, (c, do) in enumerate(parsed["priorities"])
        ],
        flags=flags, **common,
    )
