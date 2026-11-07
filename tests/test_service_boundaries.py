"""
Services depend on `shared/`, never on each other: the gateway image carries no agent code, and
the agent image carries no gateway code. A cross-service import works on a developer machine
(everything is on the path) and then fails at start-up in the container, so catch it here.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SERVICES = ("agent", "gateway", "corpus", "ingestion", "math_verifier")


def _imported_modules(path: Path) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def _offending_imports(package_dir: Path, forbidden: str) -> list[str]:
    prefix = f"services.{forbidden}"
    return [
        f"{py.relative_to(ROOT)}: {module}"
        for py in sorted(package_dir.rglob("*.py"))
        for module in _imported_modules(py)
        if module == prefix or module.startswith(prefix + ".")
    ]


def test_gateway_imports_nothing_from_the_agent_service():
    assert _offending_imports(ROOT / "services" / "gateway", "agent") == []


def test_the_gateway_image_does_not_copy_the_agent_service():
    dockerfile = (ROOT / "services" / "gateway" / "Dockerfile").read_text()
    assert "services/agent" not in dockerfile


@pytest.mark.parametrize("service", [s for s in SERVICES if (ROOT / "services" / s).is_dir()])
def test_shared_imports_no_service(service):
    assert _offending_imports(ROOT / "shared", service) == []


def test_agent_imports_nothing_from_the_gateway_service():
    assert _offending_imports(ROOT / "services" / "agent", "gateway") == []
