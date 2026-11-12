"""The committed, original Task 2 prompts the mock offers (the repo's own, not an exam's)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared.src.shared.jsonl import read_jsonl

# The product owns this file; evals reads and regenerates it
# (python -m evals exam-set generate-writing).
PROMPTS_PATH = Path(__file__).resolve().parent / "data" / "ielts_writing_prompts.jsonl"

ESSAY_TYPE_LABELS = {
    "opinion": "Opinion",
    "discuss_both_views": "Discuss both views",
    "advantages_disadvantages": "Advantages and disadvantages",
    "problem_solution": "Problem and solution",
}


@dataclass(frozen=True)
class WritingPrompt:
    id: str
    essay_type: str
    essay_type_label: str
    text: str


def load_prompts(path: Path = PROMPTS_PATH) -> list[WritingPrompt]:
    prompts: list[WritingPrompt] = []
    for row in read_jsonl(path):
        essay_type = str(row.get("extras", {}).get("essay_type", ""))
        prompts.append(WritingPrompt(
            id=row["id"], essay_type=essay_type,
            essay_type_label=ESSAY_TYPE_LABELS.get(essay_type, essay_type), text=row["prompt"],
        ))
    return prompts
