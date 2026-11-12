"""
Services depend on `shared/`, never on each other, and no production code (services, shared,
mcp_servers) imports `evals/`: the gateway image carries no agent code, the agent image carries
no gateway code, and no image carries the eval harness. A cross-service import works on a
developer machine (everything is on the path) and then fails at start-up in the container, so
each case imports every module of one package in a fresh interpreter whose import system
refuses the forbidden package, and fails if the import was even attempted (a swallowed
ImportError still counts).
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_SCRIPT = textwrap.dedent(
    """
    import importlib
    import sys
    from importlib.abc import MetaPathFinder
    from pathlib import Path

    root, package_dir, blocked = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    sys.path.insert(0, str(root))
    attempts = []


    class Refuse(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == blocked or fullname.startswith(blocked + "."):
                attempts.append(fullname)
                raise ImportError(f"{fullname} is not available in this image")
            return None


    sys.meta_path.insert(0, Refuse())
    for py in sorted(package_dir.rglob("*.py")):
        relative = py.relative_to(root).with_suffix("")
        parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        importlib.import_module(".".join(parts))
    if attempts:
        sys.exit("imported a forbidden package: " + ", ".join(sorted(set(attempts))))
    """
)


def _import_every_module(package_dir: Path, blocked: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-c", _SCRIPT, str(ROOT), str(package_dir), blocked],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )


_SERVICES = ("agent", "corpus", "gateway", "ingestion", "math_verifier")
# Production code that must import cleanly with the eval harness absent.
_PRODUCTION = (*(f"services/{name}" for name in _SERVICES), "shared", "mcp_servers")

_CASES = [
    # No service imports another service. Each is run against every other one.
    *(
        pytest.param(
            ROOT / "services" / name, f"services.{other}", id=f"{name}-without-{other}"
        )
        for name in _SERVICES
        for other in _SERVICES
        if other != name
    ),
    pytest.param(ROOT / "shared", "services", id="shared-without-any-service"),
    pytest.param(ROOT / "mcp_servers", "services", id="mcp_servers-without-any-service"),
    # Production never imports the eval harness (it is not in any image).
    *(pytest.param(ROOT / path, "evals", id=f"{path.replace('/', '.')}-without-evals")
      for path in _PRODUCTION),
]


@pytest.mark.parametrize(("package_dir", "blocked"), _CASES)
def test_package_imports_without_the_forbidden_package(package_dir, blocked):
    result = _import_every_module(package_dir, blocked)
    assert result.returncode == 0, result.stderr[-2000:]
