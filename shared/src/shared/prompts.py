"""
The one reader of versioned prompt files: `prompts/<dir>/.../<version>.md`.

A prompt file starts with a YAML header (name, version, notes) that is for people and for the
reference lint; the model never sees it, so it is cut here.
"""
from __future__ import annotations

import re

from shared.src.shared.config import PROJECT_ROOT

PROMPT_ROOT = PROJECT_ROOT / "prompts"

_FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


def strip_frontmatter(raw: str) -> str:
    """`raw` without its YAML header, trimmed."""
    return _FRONTMATTER.sub("", raw).strip()


def read_prompt(*parts: str) -> str:
    """
    The prompt at `prompts/<parts...>.md` without its YAML header: the last part is the
    version, e.g. `read_prompt("quant_agent", "tutor", "v1")` reads
    `prompts/quant_agent/tutor/v1.md`.
    """
    *directories, version = parts
    return strip_frontmatter(PROMPT_ROOT.joinpath(*directories, f"{version}.md").read_text())
