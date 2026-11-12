"""Import every module a service image ships, using only that image's dependency group.

Run inside a built image (the `compose-smoke` job does, for each service):
    docker compose ... run --rm --no-deps -T --entrypoint python <service> - < infra/import_smoke.py

Starting a service already imports its app, but not what is loaded lazily, inside a function
or behind a route. This imports every shipped module, then every third-party package any of
them imports anywhere (function bodies included), so a package the service uses but its group
in pyproject.toml forgot fails here, before merge, as a ModuleNotFoundError.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path.cwd()  # /app in the image: services/ and shared/ sit beside it
# The question-generation pipeline (question_pipeline/, outside services/) is imported only by
# evals/ and tests/, never by a service (its graph.py needs langgraph), so no service image
# carries it and no service group carries langgraph: finding langgraph installed here fails.
EVAL_ONLY_PACKAGES = ("langgraph",)
FIRST_PARTY = {"services", "shared"}


def shipped_modules() -> dict[str, Path]:
    modules = {}
    for path in sorted((ROOT / "services").rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        modules[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
    return modules


def imported_packages(path: Path) -> set[str]:
    """Top-level third-party packages a module imports, wherever the import statement sits."""
    tree = ast.parse(path.read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots - set(sys.stdlib_module_names) - FIRST_PARTY


def main() -> int:
    sys.path.insert(0, str(ROOT))
    failures = []
    modules = shipped_modules()
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:
            failures.append(name)
            print(f"FAIL {name}\n{traceback.format_exc()}", file=sys.stderr)
    for package in sorted(set().union(*map(imported_packages, modules.values()))):
        try:
            importlib.import_module(package)
        except Exception:
            failures.append(package)
            print(f"FAIL {package}\n{traceback.format_exc()}", file=sys.stderr)
    for package in EVAL_ONLY_PACKAGES:
        if importlib.util.find_spec(package) is not None:
            failures.append(package)
            print(f"FAIL {package} is installed but only evals/ uses it", file=sys.stderr)
    if failures:
        return 1
    print(f"ok: {len(modules)} modules import")
    return 0


if __name__ == "__main__":
    sys.exit(main())
