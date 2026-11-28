"""
The IELTS Speaking mock: original prompts for Parts 1 to 3, the fluency figures counted in code,
and the partial report.

Two public pages are used, and only for what they say:

- The marking criteria. Speaking is marked on four equally weighted criteria: Fluency and
  Coherence, Lexical Resource, Grammatical Range and Accuracy, and Pronunciation.
  https://ielts.org/-/media/pdfs/speaking-band-descriptors.ashx  (`DESCRIPTORS_URL`)
- The format: three parts, the Part 2 long turn is two minutes after one minute to prepare.
  https://ielts.org/take-a-test/test-types/ielts-academic-test/ielts-academic-format-speaking
  (`FORMAT_URL`)

The prompts below are the repository's own original questions, written by an agent and NOT
checked by a human; no real IELTS material.

What is "partial". The learner's recording goes to a Gemini model that hears audio. It returns a
transcript and bands for the three criteria that can be judged from what was said: Fluency and
Coherence, Lexical Resource, Grammatical Range and Accuracy. Pronunciation is NOT assessed: a
model's judgement of accent and sound has not been checked against examiner ratings, so a number
would be a guess dressed as a score. It is shown as "not assessed", the Speaking band is the mean
of the other three (rounded to the nearest half band like every band here) and is marked partial,
and the full mock's overall is marked partial while it contains it. When Pronunciation can be
assessed, `UNASSESSED` shrinks and `partial` follows from it.

Counted in code, never asked of the model: words, words per minute (against the length the
browser measured) and filler sounds ("um", "uh", "er"), per part, plus how much of the two
minutes Part 2 used.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from services.gateway.src.gateway.ielts.models import (
    SpeakingCriterion,
    SpeakingPart,
)

DESCRIPTORS_URL = "https://ielts.org/-/media/pdfs/speaking-band-descriptors.ashx"
FORMAT_URL = (
    "https://ielts.org/take-a-test/test-types/ielts-academic-test/ielts-academic-format-speaking"
)

CRITERIA: dict[str, str] = {
    "FC": "Fluency and Coherence",
    "LR": "Lexical Resource",
    "GRA": "Grammatical Range and Accuracy",
    "P": "Pronunciation",
}
ASSESSED: tuple[str, ...] = ("FC", "LR", "GRA")
UNASSESSED: dict[str, str] = {
    "P": (
        "A model's judgement of accent and sound has not been checked against examiner ratings, "
        "so this mock gives no Pronunciation band rather than a guess."
    ),
}

PART2_PREP_S = 60
PART2_TALK_S = 120
# A set's `max_clip_s` is what the browser may send per part: a little over the real limits, so a
# clip cut off by a slow stop is kept, not refused. Part 1 and Part 3 are one recording each (about
# four to five minutes in the real test).
MAX_CLIP_BYTES = 10 * 1024 * 1024
ACCEPTED_MIME = ("audio/webm", "audio/ogg", "audio/mp4", "audio/wav", "audio/x-wav", "audio/mpeg")
# Fewer words than this across all parts is "no speech": nothing to mark.
MIN_WORDS = 15

PART_TITLES = {
    1: "Part 1: Introduction and interview", 2: "Part 2: Long turn", 3: "Part 3: Discussion",
}


@dataclass(frozen=True)
class SpeakingSet:
    id: str
    part1: tuple[str, ...]
    cue_topic: str
    cue_points: tuple[str, ...]
    cue_close: str
    part3: tuple[str, ...]
    prep_s: int = PART2_PREP_S
    talk_s: int = PART2_TALK_S
    max_clip_s: tuple[float, float, float] = (330.0, PART2_TALK_S + 30.0, 330.0)
    sample: bool = False

    @property
    def title(self) -> str:
        return "Quick sample" if self.sample else "Full test"


# A short set for trying the whole flow in a few minutes: one question per part and short
# timings, marked by the same code and model as the full sets. Its score says nothing about a
# real test (a few sentences cannot show a band), and the report is labelled as a sample.
SAMPLE_SET = SpeakingSet(
    id="sp-sample",
    part1=("What is your name, and where do you live?",),
    cue_topic="Describe a meal you enjoyed recently.",
    cue_points=("what you ate", "who you ate it with", "and say why you enjoyed it."),
    cue_close="You will have a few seconds to think, then talk for up to twenty seconds.",
    part3=("Why do you think people like to share meals with others?",),
    prep_s=5, talk_s=20, max_clip_s=(40.0, 30.0, 40.0), sample=True,
)

SETS: tuple[SpeakingSet, ...] = (
    SpeakingSet(
        id="sp-01",
        part1=(
            "Where is your hometown, and what do you like about living there?",
            "What do you usually do on a free weekend?",
            "Do you prefer to cook at home or eat out? Why?",
            "How do you usually travel around your town or city?",
        ),
        cue_topic="Describe a place near water that you enjoyed visiting.",
        cue_points=(
            "where it was and how you got there",
            "who you went with",
            "what you did there",
            "and explain why you remember it well.",
        ),
        cue_close="You will have one minute to think, then talk for up to two minutes.",
        part3=(
            "Why do many people feel calmer near water?",
            "How have holiday habits changed in your country over the last few decades?",
            "Should governments protect natural places even if it limits building?",
            "Do you think tourism is more helpful or harmful to small places? Why?",
        ),
    ),
    SpeakingSet(
        id="sp-02",
        part1=(
            "What do you do: do you work or are you a student?",
            "What kind of music do you enjoy listening to, and when?",
            "How often do you use public transport?",
            "Is there a small habit you would like to change in your daily routine?",
        ),
        cue_topic="Describe a skill you learned that you are proud of.",
        cue_points=(
            "what the skill is",
            "how and when you learned it",
            "who or what helped you",
            "and explain why you are proud of it.",
        ),
        cue_close="You will have one minute to think, then talk for up to two minutes.",
        part3=(
            "Is it easier to learn new skills when you are young? Why or why not?",
            "What makes a good teacher, in your opinion?",
            "Will machines make some human skills unnecessary in the future?",
            "Should schools spend more time on practical skills than on exams?",
        ),
    ),
)


def get_set(set_id: str) -> SpeakingSet:
    for item in (*SETS, SAMPLE_SET):
        if item.id == set_id:
            return item
    raise KeyError(set_id)


def pick_set(rng: random.Random | None = None) -> SpeakingSet:
    return (rng or random).choice(SETS)


def public_set(item: SpeakingSet) -> dict[str, object]:
    """What the browser is shown: the questions and the cue card, with the part timings."""
    return {
        "id": item.id, "sample": item.sample, "title": item.title,
        "max_clip_s": list(item.max_clip_s),
        "parts": [
            {"part": 1, "title": PART_TITLES[1], "questions": list(item.part1)},
            {
                "part": 2, "title": PART_TITLES[2], "topic": item.cue_topic,
                "points": list(item.cue_points), "note": item.cue_close,
                "prep_s": item.prep_s, "talk_s": item.talk_s,
            },
            {"part": 3, "title": PART_TITLES[3], "questions": list(item.part3)},
        ],
    }


# ---- what is counted in code ------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
# Hesitation sounds only. "like" and "you know" are real words and are not counted.
_FILLER = re.compile(r"\b(?:u+h+m*|u+m+|e+r+m*|a+h+m+|h+m+)\b", re.IGNORECASE)


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def count_fillers(text: str) -> int:
    return len(_FILLER.findall(text))


def part_features(
    part: int, transcript: str, duration_s: float, talk_s: int = PART2_TALK_S,
) -> SpeakingPart:
    words = count_words(transcript)
    wpm = round(words / (duration_s / 60), 1) if duration_s >= 5 and words else None
    return SpeakingPart(
        part=part,  # type: ignore[arg-type]
        title=PART_TITLES[part], transcript=transcript.strip(), duration_s=round(duration_s, 1),
        allowed_s=talk_s if part == 2 else None, words=words, words_per_minute=wpm,
        fillers=count_fillers(transcript),
    )


def criteria_list(bands: dict[str, tuple[float, str, list[str]]]) -> list[SpeakingCriterion]:
    """The four criteria in order: the assessed ones with their band, the rest "not assessed"."""
    out: list[SpeakingCriterion] = []
    for cid, name in CRITERIA.items():
        if cid in bands:
            band, rationale, evidence = bands[cid]
            out.append(SpeakingCriterion(
                id=cid,  # type: ignore[arg-type]
                name=name, assessed=True, band=band, rationale=rationale, evidence=evidence,
            ))
        else:
            out.append(SpeakingCriterion(
                id=cid,  # type: ignore[arg-type]
                name=name, assessed=False, reason=UNASSESSED.get(cid, "Not assessed."),
            ))
    return out


def partial_note(not_assessed: list[str]) -> str:
    names = ", ".join(CRITERIA[c] for c in not_assessed)
    return (
        f"Partial band: {names} not assessed, so this is the mean of the other criteria and not "
        "a full Speaking band."
    )
