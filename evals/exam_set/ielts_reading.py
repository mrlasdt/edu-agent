"""
A small set of original IELTS-style Reading items on one original passage.

The passage is written for this repository (not derived from any official test and not copied
from any source), so it is committed. Keys are author keys: written by the implementing agent
and NOT checked by a human, so the True/False/Not Given items in particular may be arguable.
Every key cites its supporting sentence, and tests check mechanically that the evidence occurs
in the passage and that each completion answer occurs in its evidence within the word limit.

Regenerate the committed file with `python -m evals.exam_set generate-reading`.
"""
from __future__ import annotations

from typing import Any

from evals.exam_set.schema import Design, ExamItem, LabelProvenance, LabelTier, Source

FILE_NAME = "ielts_reading.jsonl"
PASSAGE_ID = "green_roofs"
WORD_LIMIT = 2

PASSAGE = (
    "Cities that grow upward often run short of ground-level space for parks, so some "
    "planners have started to look at rooftops instead. A green roof is a roof covered with "
    "soil and plants, and the idea is much older than most people assume: families in "
    "northern Europe laid turf over their wooden houses centuries ago to keep out the cold. "
    "Modern versions are thinner and lighter, built in layers that include a waterproof "
    "membrane, a drainage sheet and a shallow growing medium.\n\n"
    "The benefits are easiest to see during summer. A conventional dark roof can become far "
    "hotter than the surrounding air, whereas a planted roof stays cooler because the plants "
    "shade the surface and release water vapour. Buildings below therefore need less air "
    "conditioning, and the roof membrane is protected from sunlight, which can double its "
    "working life. Green roofs also hold back rainwater. In one study of a mid-sized city, "
    "planted roofs delayed the peak of runoff after a storm by about twenty minutes, giving "
    "drains more time to cope.\n\n"
    "There are drawbacks. A green roof costs more to install than a standard one, and the "
    "structure beneath must be strong enough to carry the extra weight of wet soil. "
    "Maintenance is also needed, since weeds and dry spells can damage thin plant layers. "
    "Supporters argue that these costs are recovered through lower energy bills, although the "
    "researchers who carried out the city study did not measure savings over more than five "
    "years. Critics point out that a roof garden open to the public raises insurance "
    "questions that a sealed roof does not."
)

TFNG_INSTRUCTION = (
    "Do the following statements agree with the information in the passage? Answer TRUE if "
    "the statement agrees, FALSE if it contradicts the passage, NOT GIVEN if there is no "
    "information on this."
)
COMPLETION_INSTRUCTION = "Complete the sentence using NO MORE THAN TWO WORDS from the passage."

# (id suffix, task type, statement, key, evidence sentence, note)
_ROWS: list[tuple[str, str, str, str, str | None, str]] = [
    ("tfng_1", "tfng",
     "In the past, some families in northern Europe covered their houses with turf for "
     "protection against the cold.",
     "TRUE",
     "families in northern Europe laid turf over their wooden houses centuries ago to keep "
     "out the cold",
     ""),
    ("tfng_2", "tfng",
     "A planted roof is always cooler than a conventional roof in winter.",
     "NOT GIVEN",
     None,
     "the passage discusses summer cooling only; nothing is said about winter"),
    ("tfng_3", "tfng",
     "The city study found that planted roofs made the peak of storm runoff arrive earlier.",
     "FALSE",
     "planted roofs delayed the peak of runoff after a storm by about twenty minutes",
     ""),
    ("fill_1", "sentence_completion",
     "Modern green roofs are built in layers that include a waterproof membrane, a drainage "
     "sheet and a shallow ______.",
     "growing medium",
     "a waterproof membrane, a drainage sheet and a shallow growing medium",
     ""),
    ("fill_2", "sentence_completion",
     "Protection from sunlight can ______ the working life of the roof membrane.",
     "double",
     "the roof membrane is protected from sunlight, which can double its working life",
     ""),
    ("fill_3", "sentence_completion",
     "The structure beneath a green roof must be able to carry the weight of wet ______.",
     "soil",
     "the structure beneath must be strong enough to carry the extra weight of wet soil",
     ""),
]


def generate_all() -> list[ExamItem]:
    items: list[ExamItem] = []
    for suffix, task_type, statement, key, evidence, note in _ROWS:
        instruction = TFNG_INSTRUCTION if task_type == "tfng" else COMPLETION_INSTRUCTION
        extras: dict[str, Any] = {"passage_id": PASSAGE_ID, "instruction": instruction}
        if note:
            extras["key_note"] = note
        items.append(
            ExamItem(
                id=f"ielts_reading_{PASSAGE_ID}_{suffix}",
                exam="ielts",
                skill="reading",
                task_type=task_type,
                prompt=statement,
                passage=PASSAGE,
                ground_truth=key,
                evidence=evidence,
                label_provenance=LabelProvenance(
                    tier=LabelTier.author_key,
                    date="2026-10-24",
                    note="written by an AI agent; not checked by a human",
                ),
                source=Source(kind="original", license="original to this repository (MIT)"),
                design=Design(generator="llm_drafted"),
                extras=extras,
            )
        )
    return items
