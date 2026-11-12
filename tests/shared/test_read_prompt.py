"""
`shared.prompts.read_prompt`: the one reader of prompt files, including a variant that extends
another prompt and states only the section it adds.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from shared.src.shared import prompts
from shared.src.shared.prompts import read_prompt


@pytest.fixture
def prompt_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(prompts, "PROMPT_ROOT", tmp_path)
    return tmp_path


def _write(root: Path, name: str, text: str) -> None:
    path = root / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


BASE = "---\nversion: 3\nnotes: base\n---\n\nIntro\n\n# Scope\nGRE only\n\n# Tone\nKind\n"


def test_a_plain_prompt_is_its_body_without_the_header(prompt_root):
    _write(prompt_root, "agent/tutor/v3", BASE)
    assert read_prompt("agent", "tutor", "v3") == "Intro\n\n# Scope\nGRE only\n\n# Tone\nKind"


def test_a_variant_is_its_base_with_its_section_placed_before_the_named_line(prompt_root):
    _write(prompt_root, "agent/tutor/v3", BASE)
    _write(
        prompt_root, "agent_tools/tutor/v1",
        '---\nversion: 1\nextends: agent/tutor/v3\ninsert_before: "# Scope"\n---\n\n'
        "# Tools\nUse them.\n",
    )
    assert read_prompt("agent_tools", "tutor", "v1") == (
        "Intro\n\n# Tools\nUse them.\n\n# Scope\nGRE only\n\n# Tone\nKind"
    )


def test_a_variant_follows_a_change_to_the_base_it_extends(prompt_root):
    _write(prompt_root, "agent/tutor/v3", BASE)
    _write(
        prompt_root, "agent_tools/tutor/v1",
        '---\nextends: agent/tutor/v3\ninsert_before: "# Tone"\n---\n# Tools\nUse them.\n',
    )
    _write(prompt_root, "agent/tutor/v3", BASE.replace("GRE only", "GRE and GMAT"))
    assert "GRE and GMAT" in read_prompt("agent_tools", "tutor", "v1")


@pytest.mark.parametrize("anchor", ["# Missing", "#"])
def test_an_anchor_that_does_not_match_exactly_one_line_is_an_error(prompt_root, anchor):
    _write(prompt_root, "agent/tutor/v3", BASE + "\n#\n#\n")
    _write(
        prompt_root, "agent_tools/tutor/v1",
        f'---\nextends: agent/tutor/v3\ninsert_before: "{anchor}"\n---\nX\n',
    )
    with pytest.raises(ValueError, match="insert_before"):
        read_prompt("agent_tools", "tutor", "v1")


def test_the_math_tools_prompts_are_the_pinned_base_plus_the_math_tools_section():
    for mode, base_version in (("tutor", "v3"), ("solve", "v2")):
        base = read_prompt("quant_agent", mode, base_version)
        variant = read_prompt("quant_agent_math_tools", mode, "v1")
        head, tail = base.split("# GRE Quant scope")
        assert variant.startswith(head + "# Math tools\n")
        assert variant.endswith("# GRE Quant scope" + tail)
        assert variant.count("# Math tools") == 1
