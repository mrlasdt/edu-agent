"""
Deterministic fake model for the question-generation pipeline (offline mode and tests).

It plays all three pipeline roles (planner, generator experts, critic) and the baseline. The
generator is *seeded with flaws*: depending on the request id, its first attempt has a wrong
key, a leaked answer, a duplicated option, or is not JSON at all; when the prompt carries
revision feedback it returns a corrected item, except for one "stubborn" id that never
improves. That makes the before/after comparison exercise the real pipeline code
(routing, tools, revise loop, escalation) without a network. The resulting numbers describe
the fake, not any model.
"""
from __future__ import annotations

import json
import re
from typing import Any

from services.agent.src.agent.pipeline.graph import rule_based_type
from services.agent.src.agent.pipeline.schema import QuestionType

# Index into this by (digits of the request id) % 7.
FLAW_CYCLE = [None, "wrong_key", "leak", "bad_distractor", "not_json", None, "stubborn"]


def _index(request_id: str) -> int:
    digits = re.sub(r"\D", "", request_id)
    return int(digits) if digits else 0


def _good_item(qtype: QuestionType, idx: int) -> dict[str, Any]:
    base: dict[str, Any] = {
        "question_type": qtype.value, "choices": [], "expression": "",
        "quantity_a": "", "quantity_b": "", "explanation": "Work it out step by step.",
    }
    if qtype == QuestionType.numeric_entry:
        a = 3 + idx % 5
        b = a + 1 + idx % 3
        return base | {
            "stem": f"Find all real solutions of x^2 - {a + b}x + {a * b} = 0. "
                    "Enter them separated by commas.",
            "expression": f"x**2 - {a + b}*x + {a * b} = 0",
            "answer": f"{a},{b}",
        }
    if qtype == QuestionType.multiple_choice_single:
        p, q, r = 2 + idx % 4, 3 + idx % 3, 2 + idx % 5
        value = p + q * r
        opts = [value, value + 1, value - 1, value + q, value * 2]
        shift = idx % len(opts)
        return base | {
            "stem": f"What is the value of {p} + {q} * {r}?",
            "choices": [str(o) for o in opts[shift:] + opts[:shift]],
            "expression": f"{p} + {q}*{r}",
            "answer": str(value),
        }
    if qtype == QuestionType.multiple_choice_multiple:
        n = 3 + idx % 6
        return base | {
            "stem": f"Which of the following values satisfy x^2 = {n * n}? Select all that apply.",
            "choices": [str(n), str(-n), str(n + 1), str(-(n + 1)), "0"],
            "expression": f"x**2 - {n * n} = 0",
            "answer": f"{n},{-n}",
        }
    a, b = 2 + idx % 3, 2 + idx % 2
    greater = "A" if a**2 > b**3 else "B" if a**2 < b**3 else "C"
    return base | {
        "stem": f"Quantity A: {a}^2\nQuantity B: {b}^3",
        "quantity_a": f"{a}**2", "quantity_b": f"{b}**3", "answer": greater,
    }


def _flawed(item: dict[str, Any], flaw: str) -> dict[str, Any]:
    qtype = QuestionType(item["question_type"])
    out = dict(item)
    if flaw == "leak":
        out["stem"] = f"{item['stem']} (The correct answer is {item['answer']}.)"
    elif flaw == "bad_distractor" and out["choices"]:
        out["choices"] = [out["choices"][0], f"{out['choices'][0]}.0", *out["choices"][2:]]
    elif qtype == QuestionType.quantitative_comparison:  # wrong_key / stubborn / fallback
        out["answer"] = "B" if item["answer"] == "A" else "A"
    elif qtype == QuestionType.multiple_choice_multiple:
        n = abs(int(item["answer"].split(",")[0]))
        out["answer"] = f"{n},{-(n + 1)}"
    elif qtype == QuestionType.multiple_choice_single:
        out["answer"] = str(int(item["answer"]) + 1)
    else:
        a, b = item["answer"].split(",")
        out["answer"] = f"{a},{int(b) + 1}"
    return out


def _generate(payload: dict[str, Any]) -> str:
    idx = _index(str(payload.get("request_id", "")))
    qtype = QuestionType(payload["plan"]["question_type"])
    item = _good_item(qtype, idx)
    flaw = FLAW_CYCLE[idx % len(FLAW_CYCLE)]
    revising = bool(payload.get("feedback"))
    if flaw is None or (revising and flaw != "stubborn"):
        return json.dumps(item)
    if flaw == "not_json":
        return "Sure! Here is a great GRE question about that topic. Let me know what you think."
    return json.dumps(_flawed(item, "wrong_key" if flaw == "stubborn" else flaw))


def _plan(request: dict[str, Any]) -> str:
    qtype = request.get("question_type") or rule_based_type(str(request["topic"])).value
    return json.dumps({
        "question_type": qtype,
        "learning_objective": f"Practice {request['topic']}.",
        "constraints": ["integer data", "exactly one defensible answer"],
    })


def respond(model: str, messages: list[dict[str, Any]]) -> str | None:
    """Reply for a pipeline prompt, or None when the prompt is not a pipeline prompt."""
    system = str(messages[0]["content"])
    if "planner of a GRE" in system:
        return _plan(json.loads(messages[-1]["content"]))
    if "critic in a GRE" in system:
        return json.dumps({"ambiguous": False, "leaks_answer": False, "issues": []})
    if "expert in a GRE" in system or "GRE Quantitative Reasoning question writer" in system:
        return _generate(json.loads(messages[-1]["content"]))
    return None
