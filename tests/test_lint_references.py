"""
The reference lint (scripts/lint_references.py): names in code, configs and prompt files agree.

Each test builds a small repository tree in tmp_path, breaks one reference, and checks the lint
reports it by file and line. The last tests run the lint on this repository and exercise the run
time half: `prompt_version` raises for an unknown pair, and the two defects the lint was written
for (the completeness check and the style critic) now report a version that names a real file.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.lint_references import lint
from shared.src.shared.config import ModelConfig, ModelRoleConfig, UnknownPromptError

ROLES = """\
roles:
  quant_agent:
    provider: p
    model: m
  style_critic:
    provider: p
    model: m
"""

PINS = """\
quant_agent:
  tutor: v2
style_critic:
  rubric: v1
"""

CODE = """\
from shared.src.shared.config import load_model_config

def run():
    cfg = load_model_config()
    model = cfg.litellm_model("quant_agent")
    version = cfg.prompt_version("quant_agent", "tutor")
    cfg.litellm_model("style_critic")
    cfg.prompt_version("style_critic", "rubric")
    return model, version
"""


def make_repo(
    tmp_path: Path,
    *,
    pins: str = PINS,
    roles: str = ROLES,
    code: str = CODE,
    prompts: tuple[str, ...] = (
        "quant_agent/tutor/v1.md",
        "quant_agent/tutor/v2.md",
        "style_critic/rubric/v1.md",
    ),
) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "prompt_pins.yaml").write_text(pins)
    (tmp_path / "config" / "model_config.test.yaml").write_text(roles)
    for name in prompts:
        path = tmp_path / "prompts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("---\nversion: 1\n---\n\nbody\n")
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "agent.py").write_text(textwrap.dedent(code))
    return tmp_path


def messages(root: Path) -> list[str]:
    return [str(p) for p in lint(root, reserved={}).problems]


def test_a_consistent_tree_has_no_problems(tmp_path):
    assert messages(make_repo(tmp_path)) == []


def test_a_prompt_call_for_an_unpinned_pair_names_the_file_and_line(tmp_path):
    code = CODE.replace('("style_critic", "rubric")', '("orchestrator", "completeness")')
    problems = messages(make_repo(tmp_path, code=code))
    assert problems == [
        "services/agent.py:8: prompt_version('orchestrator', 'completeness') names a pair that "
        "config/prompt_pins.yaml does not pin"
    ]


def test_a_mode_constant_and_a_module_constant_are_read_like_literals(tmp_path):
    code = """\
        from shared.src.shared.models import Mode
        AGENT = "quant_agent"

        def run(cfg):
            cfg.litellm_model("quant_agent")
            cfg.litellm_model("style_critic")
            cfg.prompt_version(AGENT, Mode.solve.value)
        """
    problems = messages(make_repo(tmp_path, code=code))
    assert len(problems) == 1
    assert "services/agent.py:7: prompt_version('quant_agent', 'solve')" in problems[0]


def test_a_call_with_variable_arguments_is_not_judged_but_is_reported(tmp_path):
    code = CODE + "\ndef other(cfg, a, m):\n    return cfg.prompt_version(a, m)\n"
    root = make_repo(tmp_path, code=code)
    report = lint(root, reserved={})
    assert report.problems == []
    assert report.dynamic_prompt_calls == [("services/agent.py", 12)]


def test_a_pin_without_its_prompt_file_names_the_pin_line(tmp_path):
    root = make_repo(tmp_path, pins=PINS.replace("tutor: v2", "tutor: v9"))
    assert messages(root) == [
        "config/prompt_pins.yaml:2: (quant_agent, tutor) is pinned to v9 but "
        "prompts/quant_agent/tutor/v9.md does not exist"
    ]


def test_a_prompt_directory_nothing_pins_is_reported(tmp_path):
    root = make_repo(tmp_path, prompts=(
        "quant_agent/tutor/v2.md", "style_critic/rubric/v1.md", "quant_agent/solve/v1.md"))
    problems = messages(root)
    assert len(problems) == 1
    assert problems[0].startswith("prompts/quant_agent/solve:1: prompt files exist for")


def test_a_prompt_file_newer_than_its_pin_is_reported_but_older_history_is_not(tmp_path):
    root = make_repo(tmp_path, prompts=(
        "quant_agent/tutor/v1.md", "quant_agent/tutor/v2.md", "quant_agent/tutor/v3.md",
        "style_critic/rubric/v1.md"))
    problems = messages(root)
    assert len(problems) == 1
    assert problems[0].startswith("prompts/quant_agent/tutor/v3.md:1: is newer than the pin")


def test_a_call_for_a_role_no_config_defines_is_reported(tmp_path):
    code = CODE.replace('litellm_model("style_critic")', 'litellm_model("style_critc")')
    problems = messages(make_repo(tmp_path, code=code))
    assert problems == [
        "services/agent.py:7: litellm_model('style_critc') names a role that "
        "config/model_config.test.yaml does not define",
    ]


def test_a_configured_role_no_code_names_is_reported_until_it_is_reserved(tmp_path):
    roles = ROLES + "  orphan:\n    provider: p\n    model: m\n"
    root = make_repo(tmp_path, roles=roles)
    assert len(messages(root)) == 1
    assert messages(root)[0].startswith("config/model_config.test.yaml:8: role 'orphan'")
    assert lint(root, reserved={"orphan": "not built yet"}).problems == []


def test_a_reserved_role_that_is_now_used_or_gone_must_be_removed(tmp_path):
    root = make_repo(tmp_path)
    problems = [str(p) for p in lint(root, reserved={"quant_agent": "x", "ghost": "y"}).problems]
    assert any("lists 'quant_agent', but code now names it" in p for p in problems)
    assert any("lists 'ghost', which no model config defines" in p for p in problems)


def test_model_configs_must_define_the_same_roles(tmp_path):
    root = make_repo(tmp_path)
    (root / "config" / "model_config.other.yaml").write_text(ROLES.split("  style_critic")[0])
    problems = messages(root)
    assert any("model_config.other.yaml:1: role 'style_critic' is defined in another" in p
               for p in problems)


def test_this_repository_has_no_reference_problems():
    report = lint()
    assert [str(p) for p in report.problems] == []


# --- the run time half --------------------------------------------------------------------


def test_prompt_version_raises_for_an_unpinned_pair_and_allows_an_explicit_default():
    cfg = ModelConfig(
        roles={"quant_agent": ModelRoleConfig(provider="p", model="m")},
        prompts={"quant_agent": {"tutor": "v3"}},
    )
    assert cfg.prompt_version("quant_agent", "tutor") == "v3"
    with pytest.raises(UnknownPromptError, match=r"\('quant_agent', 'solve'\)"):
        cfg.prompt_version("quant_agent", "solve")
    with pytest.raises(KeyError):  # an unknown agent is the same error, not an empty default
        cfg.prompt_version("nope", "tutor")
    assert cfg.prompt_version("quant_agent", "solve", default="v1") == "v1"


@pytest.mark.asyncio
async def test_the_style_critic_reports_its_own_prompt_version_and_sends_its_rubric_file():
    from shared.src.shared.config import load_model_config
    from shared.src.shared.critics.style_critic import run_style_critic

    cfg = load_model_config()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "PASS"
    meter = MagicMock()
    meter.complete = AsyncMock(return_value=response)

    await run_style_critic("An essay.", trace_id="t", meter=meter)

    sent = meter.complete.await_args.kwargs
    rubric = sent["messages"][0]["content"]
    assert rubric.startswith("You are a strict GRE essay structure reviewer")
    assert sent["metadata"]["prompt_version"] == cfg.prompt_version("style_critic", "rubric")


@pytest.mark.asyncio
async def test_the_completeness_check_reports_a_pinned_version_and_fills_in_the_section():
    from services.gateway.src.gateway.orchestrator import enrich as guard
    from shared.src.shared.config import load_model_config
    from shared.src.shared.models import Session

    cfg = load_model_config()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = '{"complete": true, "clarification": ""}'
    meter = MagicMock()
    meter.complete = AsyncMock(return_value=response)
    session = Session(candidate_id="c", test_type="gre", section="quant")

    with patch.object(guard, "load_model_config", return_value=cfg):
        await guard._call_enrichment_llm("What is 2+2?", session, "t", meter)

    sent = meter.complete.await_args.kwargs
    assert "GRE QUANT session" in sent["messages"][0]["content"]
    assert "{section}" not in sent["messages"][0]["content"]
    assert sent["metadata"]["prompt_version"] == cfg.prompt_version("orchestrator", "completeness")
