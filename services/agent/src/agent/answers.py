"""
Final-answer extraction and normalisation for Quant solve output.

The solve prompt asks for `**Answer: [value]**`, but models also write
`Final Answer: The values of x that satisfy the equation are 2 and -2`. Extraction finds the
answer line; normalisation reduces prose and LaTeX to the value list the math verifier and the
graders can parse ("2, -2").
"""
from __future__ import annotations

import re

_BOLD_ANSWER = re.compile(r"\*\*\s*(?:final\s+)?answer\s*:\s*(.+?)\*\*", re.IGNORECASE)
_ANSWER_LINE = re.compile(
    r"^[\s*#>_-]*(?:final\s+)?answer\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE
)
_WORDS = re.compile(r"[A-Za-z]{3,}\s+[A-Za-z]{3,}")
_PROSE_TAIL = re.compile(r"\b(?:is|are|=)\s+(?!.*\b(?:is|are)\b)(.+)$", re.IGNORECASE)
_FRAC = re.compile(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}")


def extract_final_answer(text: str) -> str | None:
    """The answer: the `**Answer: x**` tag, else an `Answer:` line, else the last line."""
    bold = _BOLD_ANSWER.findall(text)
    if bold:
        return str(bold[-1]).strip()
    lines = _ANSWER_LINE.findall(text)
    if lines:
        return _strip_markup(str(lines[-1])) or None
    nonblank = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return nonblank[-1] if nonblank else None


def _strip_markup(text: str) -> str:
    out = _FRAC.sub(r"(\1)/(\2)", text)
    out = re.sub(r"\\[()\[\]]|\$|\\boxed\{|\\text\{([^}]*)\}", r"\1", out)
    out = out.replace("{", "").replace("}", "").replace("**", "")
    return out.strip().rstrip(".").strip()


def normalize_answer(raw: str) -> str:
    """
    Reduce an extracted answer to a bare value list.

    Prose such as "The values of x ... are 2 and -2" keeps only what follows the last
    "is"/"are", but only when that tail contains a digit, so a verdict like
    "Quantity A is greater" is left intact. "and"/"or" between values become commas.
    """
    out = _strip_markup(raw)
    tail = _PROSE_TAIL.search(out) if re.search(r"[A-Za-z]{3,}", out) else None
    if tail:
        value = tail.group(1)
        if re.search(r"\d", value) and not _WORDS.search(value):
            out = value.strip().rstrip(".").strip()
    return re.sub(r"\s*(?:,\s*)?\b(?:and|or)\b\s*", ", ", out).strip(", ")
