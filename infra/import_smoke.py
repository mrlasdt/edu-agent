"""Import every module a service image ships, using only that image's dependency group.

Run inside a built image (the `compose-smoke` job does, for each service):
    docker compose ... run --rm --no-deps -T --entrypoint python <service> - < infra/import_smoke.py

Starting a service already imports its app, but not the modules loaded lazily, inside a
function or behind a route. This imports all of them, so a package the service uses but its
group in pyproject.toml forgot fails here, before merge, as a ModuleNotFoundError.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path.cwd()  # /app in the image: services/ and shared/ sit beside it
# The question-generation pipeline is imported only by evals/ and tests/, never by a service
# (its graph.py needs langgraph, its tools.py sympy), so those modules are skipped, and no
# service group carries langgraph.
EVAL_ONLY_PREFIXES = ("services.agent.src.agent.pipeline",)
EVAL_ONLY_PACKAGES = ("langgraph",)


def shipped_modules() -> list[str]:
    names = []
    for path in sorted((ROOT / "services").rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return names


def main() -> int:
    sys.path.insert(0, str(ROOT))
    failures = []
    modules = [m for m in shipped_modules() if not m.startswith(EVAL_ONLY_PREFIXES)]
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:
            failures.append(name)
            print(f"FAIL {name}\n{traceback.format_exc()}", file=sys.stderr)
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
