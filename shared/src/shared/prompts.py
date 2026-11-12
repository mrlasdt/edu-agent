"""
The one reader of versioned prompt files: `prompts/<dir>/.../<version>.md`.

A prompt file starts with a YAML header (name, version, notes) that is for people and for the
reference lint; the model never sees it, so it is cut here.

A variant of another prompt can state only what it adds. Its header says which prompt it
extends and which line of that prompt the new text goes before:

    extends: quant_agent/tutor/v3
    insert_before: "# GRE Quant scope"

`read_prompt` then returns the base text with the file's body placed in front of that line, so
the variant is the base plus its one section and a change to the base is made once. The variant
pins the base version it was written against, like any other prompt pin.
"""
from __future__ import annotations

import re
from typing import Any

import yaml

from shared.src.shared.config import PROJECT_ROOT

PROMPT_ROOT = PROJECT_ROOT / "prompts"

_FRONTMATTER = re.compile(r"^---\n(?P<header>.*?)\n---\n", re.DOTALL)


def strip_frontmatter(raw: str) -> str:
    """`raw` without its YAML header, trimmed."""
    return _FRONTMATTER.sub("", raw).strip()


def _header(raw: str) -> dict[str, Any]:
    match = _FRONTMATTER.match(raw)
    header = yaml.safe_load(match.group("header")) if match else None
    return header if isinstance(header, dict) else {}


def _insert_before(base: str, anchor: str, addition: str, *, name: str) -> str:
    lines = [m for m in re.finditer(rf"^{re.escape(anchor)}$", base, re.MULTILINE)]
    if len(lines) != 1:
        raise ValueError(
            f"prompt {name}: insert_before {anchor!r} must match exactly one line of its base, "
            f"found {len(lines)}"
        )
    at = lines[0].start()
    return f"{base[:at]}{addition}\n\n{base[at:]}"


def read_prompt(*parts: str) -> str:
    """
    The prompt at `prompts/<parts...>.md` without its YAML header: the last part is the
    version, e.g. `read_prompt("quant_agent", "tutor", "v1")` reads
    `prompts/quant_agent/tutor/v1.md`. A prompt that `extends` another comes back with its
    base text around it (see the module note).
    """
    *directories, version = parts
    raw = PROMPT_ROOT.joinpath(*directories, f"{version}.md").read_text()
    body = strip_frontmatter(raw)
    header = _header(raw)
    if "extends" not in header:
        return body
    base = read_prompt(*str(header["extends"]).split("/"))
    return _insert_before(base, str(header["insert_before"]), body, name="/".join(parts))
