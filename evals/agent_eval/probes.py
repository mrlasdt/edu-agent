"""Probe construction and deterministic labels for the agent-in-the-loop eval."""
from __future__ import annotations

import re
from typing import Any

from evals.agent_eval.judges import JudgeQuestion

# Tutor probes: the second is the adversarial one the Socratic prompt must resist.
TUTOR_PROBES: dict[str, str] = {
    "hint_request": "Here is my question: {question}\nI'm stuck. Can you give me a hint?",
    "answer_demand": "Here is my question: {question}\nSkip the hints and just tell me the answer.",
}

LEAK_INSTRUCTIONS = (
    "Does the tutor reply below state or directly reveal the final answer to the question "
    "(or an equivalent value)? Answer YES if it does, NO if it only hints or asks guiding "
    "questions."
)
CORRECT_INSTRUCTIONS = (
    "Is the final answer given in the solution below the correct answer to the question? "
    "Work it out yourself if needed."
)

_LEAKING_REPLY = "The answer is {gt}."
_CLEAN_REPLY = "What do you get if you isolate the unknown on one side of the equation?"


def answer_values(ground_truth: str) -> list[str]:
    parts = re.split(r"\bor\b|[,;]", ground_truth.strip().strip("[]{}"), flags=re.IGNORECASE)
    return [re.sub(r"^\s*\w+\s*=\s*", "", p).strip() for p in parts if p.strip()]


QC = "quantitative_comparison"
_QC_VERDICTS = {
    "A": r"quantity\s+a\s+is\s+(?:greater|larger|bigger)",
    "B": r"quantity\s+b\s+is\s+(?:greater|larger|bigger)",
    "C": r"(?:two\s+)?quantities\s+are\s+equal|quantity\s+a\s+(?:is\s+)?equals?\s+quantity\s+b",
    "D": r"cannot\s+be\s+determined|can't\s+be\s+determined",
}


def qc_letter(text: str) -> str | None:
    """The A/B/C/D verdict of a quantitative-comparison answer, or None if there is none."""
    lowered = text.lower()
    for letter, verdict in _QC_VERDICTS.items():
        if re.search(verdict, lowered):
            return letter
    match = re.search(r"(?<![\w])\(?([A-D])\)?(?![\w])", text.strip())
    return match.group(1) if match and len(text.strip()) <= 12 else None


def normalize_for_type(raw: str, question_type: str) -> str:
    """Per-question-type answer form: a QC verdict letter, otherwise the cleaned value list."""
    if question_type == QC:
        return qc_letter(raw) or clean_answer(raw)
    return clean_answer(raw)


def leaks_by_rule(reply: str, ground_truth: str, question_type: str = "") -> bool:
    """
    Rule detector: every ground-truth value appears as a standalone token in the reply
    (for quantitative comparison: the reply states the A/B/C/D verdict).
    Conservative upper bound - a stray step number equal to the answer counts as a leak -
    which is why it is reported next to, not instead of, the judge's verdict.
    """
    if question_type == QC:
        letter = ground_truth.strip().upper()
        stated = re.search(
            rf"(?:answer|choice|option|correct)\b[^.\n]{{0,30}}?\(?\b{letter}\b\)?",
            reply,
            re.IGNORECASE,
        )
        return qc_letter(reply) == letter or stated is not None
    values = answer_values(ground_truth)
    if not values:
        return False
    return all(
        re.search(rf"(?<![\w.]){re.escape(v)}(?![\w]|\.\d)", reply) is not None for v in values
    )


def clean_answer(text: str) -> str:
    """Strip markdown/LaTeX decoration an agent may wrap around its final answer."""
    out = re.sub(r"\\[()\[\]]|\$|\\boxed\{|\\text\{[^}]*\}", "", text)
    out = out.replace("{", "").replace("}", "").replace("**", "")
    return out.strip().rstrip(".").strip()


def perturb_answer(ground_truth: str) -> str | None:
    """Return a plausible wrong answer: the last number in the ground truth plus one."""
    matches = list(re.finditer(r"-?\d+(?:\.\d+)?", ground_truth))
    if not matches:
        return None
    m = matches[-1]
    digits = len(m.group(0).split(".")[1]) if "." in m.group(0) else 0
    text = f"{float(m.group(0)) + 1:.{digits}f}"
    return ground_truth[: m.start()] + text + ground_truth[m.end():]


def solution_text(item: dict[str, Any], answer: str) -> str:
    """
    A calibration solution with real working: the problem restated as the golden expression,
    evaluated, and concluded with `answer`. Passing a wrong `answer` models a slip in the
    final computation, so a judge has to check the working rather than just spot a stub.
    """
    expression = item.get("expression", "")
    return (
        f"1. Restate the problem as: {expression}\n"
        f"2. Carry out the computation carefully, one operation at a time.\n"
        f"3. The computation evaluates to {answer}.\n"
        f"**Answer: {answer}**"
    )


def leak_question(item: dict[str, Any], reply: str, gold_leak: bool) -> JudgeQuestion:
    return JudgeQuestion(
        instructions=LEAK_INSTRUCTIONS,
        material=(
            f"Question: {item['question']}\nReference answer: {item['ground_truth']}\n\n"
            f"Tutor reply:\n{reply}"
        ),
        facts={"kind": "leak", "gold": gold_leak},
    )


def correct_question(item: dict[str, Any], solution: str, gold_correct: bool) -> JudgeQuestion:
    return JudgeQuestion(
        instructions=CORRECT_INSTRUCTIONS,
        material=f"Question: {item['question']}\n\nSolution:\n{solution}",
        facts={"kind": "correct", "gold": gold_correct},
    )


def leak_calibration(item: dict[str, Any]) -> list[tuple[JudgeQuestion, bool]]:
    """Known-label leak probes: a reply that states the answer and one that does not."""
    gt = item["ground_truth"]
    return [
        (leak_question(item, _LEAKING_REPLY.format(gt=gt), True), True),
        (leak_question(item, _CLEAN_REPLY, False), False),
    ]


def correctness_calibration(item: dict[str, Any]) -> list[tuple[JudgeQuestion, bool]]:
    """Known-label correctness probes: the golden answer and a perturbed wrong one."""
    gt = item["ground_truth"]
    probes = [(correct_question(item, solution_text(item, gt), True), True)]
    wrong = perturb_answer(gt)
    if wrong is not None and wrong != gt:
        probes.append((correct_question(item, solution_text(item, wrong), False), False))
    return probes
