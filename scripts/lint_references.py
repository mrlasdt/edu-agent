"""
Reference lint: the names the code, the configs and the prompt files use for each other must agree.

The stack refers to things by string: `prompt_version("quant_agent", "tutor")` names a pin in
config/prompt_pins.yaml, which names prompts/quant_agent/tutor/v3.md;
`litellm_model("style_critic")` names a role in config/model_config*.yaml. A wrong name used to
resolve quietly (an unpinned prompt reported "v1" to Langfuse and the turn record), so a typo or
a half-finished rename looked like a working system. This script turns each of those into a
failing line, `path:line: message`, and exits 1 when there is any. It runs in CI (the lint job)
and as tests/test_lint_references.py.

Checks:
  1. Every `x.prompt_version(agent, mode)` call site whose arguments can be read statically (a
     string literal, `Mode.solve.value`, or a module-level string constant) names a pinned pair.
  2. Every pin has its prompt file: prompts/<agent>/<mode>/<version>.md.
  3. Every prompt directory <agent>/<mode> is pinned, and no file is newer than its pin (a newer
     file that nothing selects). Older files are history and stay.
  4. Every role named at a `litellm_model(...)` / `role(...)` call site exists in every model
     config, and the model configs define the same roles.
  5. Every configured role is used: its name appears as a string in the code, or it is listed in
     RESERVED_ROLES below with the reason.

Call sites whose arguments are variables (a loop over modes, a role passed through a wrapper) are
counted and printed but not judged here; `prompt_version` raises for an unknown pair at run time,
and checks 2 and 3 cover the data side of them.
"""
from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Code that is scanned for call sites. Tests are not: they patch and fake these names on purpose.
CODE_DIRS = ("services", "shared", "evals", "mcp_servers", "question_pipeline")

PINS_PATH = "config/prompt_pins.yaml"
MODEL_CONFIG_GLOB = "config/model_config*.yaml"
PROMPT_DIR = "prompts"

# Roles that are configured but not called by any code yet, and why. A role listed here that is
# used (or not configured) fails the lint, so this list cannot go stale.
RESERVED_ROLES: dict[str, str] = {
    "math_retry": "the math verifier's retry tier; the verifier service retries without a model",
    "aw_judge_pr": "the PR-tier AW judge; only the nightly judge (aw_judge_nightly) is wired",
    "style_extractor": "the background style-extraction job, not built yet",
}

_PROMPT_CALL = "prompt_version"
_ROLE_CALLS = frozenset({"litellm_model", "litellm_fallbacks", "role"})
_VERSION = re.compile(r"^v(\d+)$")


@dataclass(frozen=True)
class Problem:
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


@dataclass
class Report:
    problems: list[Problem]
    # (path, line) of `prompt_version` calls the lint could not read statically.
    dynamic_prompt_calls: list[tuple[str, int]]


# --- reading the code ---------------------------------------------------------------------------


def _python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for name in CODE_DIRS:
        base = root / name
        if base.is_dir():
            files.extend(p for p in sorted(base.rglob("*.py")) if ".venv" not in p.parts)
    return files


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return None


def _string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level `NAME = "text"` assignments."""
    found: dict[str, str] = {}
    for node in tree.body:
        value: ast.expr | None = None
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for target in targets:
                if isinstance(target, ast.Name):
                    found[target.id] = value.value
    return found


class _Resolver:
    """Reads an argument expression as a string when it is knowable without running the code."""

    def __init__(self, local: dict[str, str], shared: dict[str, str]) -> None:
        self._local = local
        self._shared = shared

    def __call__(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) else None
        if isinstance(node, ast.Name):
            return self._local.get(node.id, self._shared.get(node.id))
        # `llm.FEEDBACK_ROLE`: the constant is looked up by its attribute name.
        if isinstance(node, ast.Attribute) and node.attr != "value":
            return self._shared.get(node.attr)
        # `Mode.solve.value`: the member name is the mode string.
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "value"
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "Mode"
        ):
            return node.value.attr
        return None


def _unique_constants(per_file: dict[Path, dict[str, str]]) -> dict[str, str]:
    """Constants whose name means one value everywhere it is defined."""
    values: dict[str, set[str]] = {}
    for constants in per_file.values():
        for name, value in constants.items():
            values.setdefault(name, set()).add(value)
    return {name: next(iter(v)) for name, v in values.items() if len(v) == 1}


# --- reading the configs ------------------------------------------------------------------------


def _pins(mapping: Any) -> dict[tuple[str, str], str]:
    pins: dict[tuple[str, str], str] = {}
    for agent, modes in (mapping or {}).items():
        for mode, version in (modes or {}).items():
            pins[(str(agent), str(mode))] = str(version)
    return pins


def _pin_lines(path: Path, section: str | None) -> dict[tuple[str, str], int]:
    """Line of each (agent, mode) key in the YAML file, so a problem can name its line."""
    node = yaml.compose(path.read_text())
    if section is not None and isinstance(node, yaml.MappingNode):
        node = next((v for k, v in node.value if k.value == section), None)
    lines: dict[tuple[str, str], int] = {}
    if not isinstance(node, yaml.MappingNode):
        return lines
    for agent_key, agent_val in node.value:
        if not isinstance(agent_val, yaml.MappingNode):
            continue
        for mode_key, _ in agent_val.value:
            lines[(agent_key.value, mode_key.value)] = mode_key.start_mark.line + 1
    return lines


@dataclass
class _Config:
    path: Path
    rel: str
    roles: dict[str, int]  # role name -> line
    pins: dict[tuple[str, str], str]
    pin_lines: dict[tuple[str, str], int]
    pin_source: str  # file the pins came from


def _load_configs(root: Path) -> tuple[list[_Config], list[Problem]]:
    problems: list[Problem] = []
    pins_file = root / PINS_PATH
    shared_pins: dict[tuple[str, str], str] = {}
    shared_lines: dict[tuple[str, str], int] = {}
    if pins_file.is_file():
        shared_pins = _pins(yaml.safe_load(pins_file.read_text()))
        shared_lines = _pin_lines(pins_file, None)
    else:
        problems.append(Problem(PINS_PATH, 1, "prompt pins file is missing"))

    configs: list[_Config] = []
    for path in sorted(root.glob(MODEL_CONFIG_GLOB)):
        rel = path.relative_to(root).as_posix()
        raw = yaml.safe_load(path.read_text()) or {}
        node = yaml.compose(path.read_text())
        role_lines: dict[str, int] = {}
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                if key.value == "roles" and isinstance(value, yaml.MappingNode):
                    role_lines = {k.value: k.start_mark.line + 1 for k, _ in value.value}
        if "prompts" in raw:  # this config pins its own prompts
            pins, lines, source = _pins(raw["prompts"]), _pin_lines(path, "prompts"), rel
        else:
            pins, lines, source = shared_pins, shared_lines, PINS_PATH
        configs.append(_Config(path, rel, role_lines, pins, lines, source))
    return configs, problems


# --- the checks ---------------------------------------------------------------------------------


def _version_number(version: str) -> int | None:
    match = _VERSION.match(version)
    return int(match.group(1)) if match else None


def _check_pins_and_files(root: Path, configs: list[_Config]) -> list[Problem]:
    problems: list[Problem] = []
    prompts = root / PROMPT_DIR
    on_disk: dict[tuple[str, str], set[str]] = {}
    if prompts.is_dir():
        for file in sorted(prompts.glob("*/*/*.md")):
            agent, mode = file.parent.parent.name, file.parent.name
            on_disk.setdefault((agent, mode), set()).add(file.stem)

    seen_sources: set[str] = set()
    for config in configs:
        if config.pin_source in seen_sources:
            continue
        seen_sources.add(config.pin_source)
        for (agent, mode), version in sorted(config.pins.items()):
            line = config.pin_lines.get((agent, mode), 1)
            if version not in on_disk.get((agent, mode), set()):
                problems.append(Problem(
                    config.pin_source, line,
                    f"({agent}, {mode}) is pinned to {version} but "
                    f"{PROMPT_DIR}/{agent}/{mode}/{version}.md does not exist",
                ))
        for (agent, mode), versions in sorted(on_disk.items()):
            pinned = config.pins.get((agent, mode))
            directory = f"{PROMPT_DIR}/{agent}/{mode}"
            if pinned is None:
                problems.append(Problem(
                    directory, 1,
                    f"prompt files exist for ({agent}, {mode}) but {config.pin_source} does not "
                    "pin that pair, so nothing selects them",
                ))
                continue
            pinned_number = _version_number(pinned)
            for version in sorted(versions):
                number = _version_number(version)
                if pinned_number is not None and number is not None and number > pinned_number:
                    problems.append(Problem(
                        f"{directory}/{version}.md", 1,
                        f"is newer than the pin ({agent}, {mode}) -> {pinned} in "
                        f"{config.pin_source}, so nothing selects it; pin it or remove it",
                    ))
    return problems


def _check_call_sites(
    root: Path, configs: list[_Config], files: list[Path]
) -> tuple[list[Problem], list[tuple[str, int]], set[str]]:
    trees = {path: _parse(path) for path in files}
    constants = {path: _string_constants(tree) for path, tree in trees.items() if tree}
    shared = _unique_constants(constants)

    problems: list[Problem] = []
    dynamic: list[tuple[str, int]] = []
    strings_in_code: set[str] = set()
    config_roles = {c.rel: set(c.roles) for c in configs}
    pin_sets = list({c.pin_source: c for c in configs}.values())

    for path, tree in trees.items():
        if tree is None:
            continue
        rel = path.relative_to(root).as_posix()
        resolve = _Resolver(constants[path], shared)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                strings_in_code.add(node.value)
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            name = node.func.attr
            if name == _PROMPT_CALL and len(node.args) == 2:
                agent, mode = resolve(node.args[0]), resolve(node.args[1])
                if agent is None and mode is None:
                    dynamic.append((rel, node.lineno))
                    continue
                for config in pin_sets:
                    if agent is not None and mode is not None:
                        if (agent, mode) not in config.pins:
                            problems.append(Problem(
                                rel, node.lineno,
                                f"prompt_version({agent!r}, {mode!r}) names a pair that "
                                f"{config.pin_source} does not pin",
                            ))
                    elif agent is not None and not any(a == agent for a, _ in config.pins):
                        problems.append(Problem(
                            rel, node.lineno,
                            f"prompt_version({agent!r}, ...) names an agent that "
                            f"{config.pin_source} does not pin",
                        ))
                    elif mode is not None and not any(m == mode for _, m in config.pins):
                        problems.append(Problem(
                            rel, node.lineno,
                            f"prompt_version(..., {mode!r}) names a mode that "
                            f"{config.pin_source} does not pin for any agent",
                        ))
            elif name in _ROLE_CALLS and node.args:
                # `.role("x")` is also a common method name; only judge it on a model config.
                if name == "role" and not _looks_like_config(node.func.value):
                    continue
                role = resolve(node.args[0])
                if role is None:
                    continue
                for config_rel, roles in config_roles.items():
                    if role not in roles:
                        problems.append(Problem(
                            rel, node.lineno,
                            f"{name}({role!r}) names a role that {config_rel} does not define",
                        ))
    return problems, dynamic, strings_in_code


def _looks_like_config(node: ast.expr) -> bool:
    text = ast.unparse(node)
    return "config" in text or text in {"cfg", "self._config"}


def _check_roles(
    configs: list[_Config], strings_in_code: set[str], reserved: dict[str, str]
) -> list[Problem]:
    problems: list[Problem] = []
    if not configs:
        return problems
    all_roles = {r for c in configs for r in c.roles}
    for config in configs:
        for role in sorted(all_roles - set(config.roles)):
            problems.append(Problem(
                config.rel, 1,
                f"role {role!r} is defined in another model config but not in this one",
            ))
    for config in configs:
        for role, line in sorted(config.roles.items(), key=lambda kv: kv[1]):
            if role in strings_in_code or role in reserved:
                continue
            problems.append(Problem(
                config.rel, line,
                f"role {role!r} is configured but no code names it; use it, remove it, or list "
                "it in RESERVED_ROLES in scripts/lint_references.py with the reason",
            ))
    for role in sorted(reserved):
        if role not in all_roles:
            problems.append(Problem(
                "scripts/lint_references.py", 1,
                f"RESERVED_ROLES lists {role!r}, which no model config defines; remove the entry",
            ))
        elif role in strings_in_code:
            problems.append(Problem(
                "scripts/lint_references.py", 1,
                f"RESERVED_ROLES lists {role!r}, but code now names it; remove the entry",
            ))
    return problems


def lint(root: Path = ROOT, reserved: dict[str, str] | None = None) -> Report:
    """Run every check over the tree at `root`."""
    reserved = RESERVED_ROLES if reserved is None else reserved
    configs, problems = _load_configs(root)
    problems += _check_pins_and_files(root, configs)
    call_problems, dynamic, strings = _check_call_sites(root, configs, _python_files(root))
    problems += call_problems
    problems += _check_roles(configs, strings, reserved)
    unique = sorted(set(problems), key=lambda p: (p.path, p.line, p.message))
    return Report(unique, sorted(dynamic))


def main() -> int:
    report = lint()
    for problem in report.problems:
        print(problem)
    if report.dynamic_prompt_calls:
        sites = ", ".join(f"{path}:{line}" for path, line in report.dynamic_prompt_calls)
        print(f"note: {len(report.dynamic_prompt_calls)} prompt_version call(s) with variable "
              f"arguments were not judged statically: {sites}", file=sys.stderr)
    if report.problems:
        print(f"{len(report.problems)} reference problem(s)", file=sys.stderr)
        return 1
    print("references ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
