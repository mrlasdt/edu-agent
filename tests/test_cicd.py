"""
Tests for CI/CD infrastructure (issue 16).

Behaviors under test:
  1. GitHub Actions workflows exist and are valid YAML
  2. ci.yml runs the offline eval on every change (no path gate)
  3. No Kubernetes manifests or cluster deploy workflow (Docker Compose is the one deploy path)
  4. Each FastAPI service exposes GET /healthz → 200
  5. One parametrized Dockerfile, one dependency group per service, a pinned uv
"""

import os
import re
import shlex
import subprocess
import tomllib
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent


# ── 1. GitHub Actions workflows ───────────────────────────────────────────────


def test_ci_workflow_exists():
    assert (ROOT / ".github/workflows/ci.yml").exists()


def test_ci_workflow_is_valid_yaml():
    content = (ROOT / ".github/workflows/ci.yml").read_text()
    data = yaml.safe_load(content)
    assert isinstance(data, dict)
    assert "jobs" in data


def _commands(job: dict) -> list[list[str]]:
    """Every shell command a job's `run` steps execute, as shlex-split argv lists."""
    commands = []
    for step in job.get("steps", []):
        script = str(step.get("run", "")).replace("\\\n", " ")
        for line in script.splitlines():
            if line.strip():
                commands.append(shlex.split(line, comments=True))
    return commands


# ── 2. The offline eval always runs in CI ─────────────────────────────────────


def test_ci_offline_eval_always_runs():
    """
    The offline eval takes about 10 s, so it is not path-gated: a change to shared/, the math
    verifier, pyproject.toml or uv.lock must run it too.
    """
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    assert "changes" not in jobs
    eval_job = jobs["eval-slice"]
    assert "if" not in eval_job
    assert "needs" not in eval_job
    assert any(
        argv[:5] == ["uv", "run", "python", "-m", "evals.agent_eval"] and "offline" in argv
        for argv in _commands(eval_job)
    )


def test_ci_workflow_audits_dependencies_and_scans_for_secrets():
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]

    def runs(job_id: str, prefix: list[str]) -> bool:
        return any(argv[: len(prefix)] == prefix for argv in _commands(jobs[job_id]))

    pip_audit = [
        argv
        for argv in _commands(jobs["audit-python"])
        if argv[:2] == ["uvx", "--from"] and argv[3:4] == ["pip-audit"]
    ]
    assert len(pip_audit) == 1
    assert pip_audit[0][2].startswith("pip-audit==")
    assert runs("audit-web", ["npm", "audit", "--audit-level=high"])
    assert jobs["audit-web"]["defaults"]["run"]["working-directory"] == "web"
    assert runs("secret-scan", ["./gitleaks", "detect"])


# ── 3. No cluster deploy ─────────────────────────────────────────────────────


def test_docker_compose_is_the_only_deploy_path():
    """
    The k8s manifests and the deploy workflow were removed unapplied because they disagreed with
    each other (image names, ports, missing services). Do not bring them back untested.
    """
    assert (ROOT / "infra/docker-compose.yml").is_file()
    assert not (ROOT / "infra/k8s").exists()
    assert not (ROOT / ".github/workflows/deploy.yml").exists()


# ── 4. Health check endpoints ─────────────────────────────────────────────────


def test_gateway_healthz():
    from services.gateway.src.gateway.main import app

    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_agent_healthz():
    from services.agent.src.agent.main import app

    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200


def test_corpus_healthz():
    from services.corpus.src.corpus.main import app

    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200


def test_ingestion_healthz():
    from services.ingestion.src.ingestion.main import app

    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200


# ── 5. One Dockerfile, one dependency group per service ──────────────────────

SERVICES = ["gateway", "agent", "corpus", "ingestion", "math_verifier"]


def _compose_service_args() -> dict[str, str]:
    """Compose service name -> the SERVICE build arg of every service built from the Dockerfile."""
    compose = yaml.safe_load((ROOT / "infra/docker-compose.yml").read_text())
    return {
        name: service["build"]["args"]["SERVICE"]
        for name, service in compose["services"].items()
        if "build" in service and service["build"]["dockerfile"] == "infra/Dockerfile"
    }


def test_one_dockerfile_builds_every_python_service():
    assert (ROOT / "infra/Dockerfile").is_file()
    assert not list((ROOT / "services").glob("*/Dockerfile")), "per-service Dockerfiles are gone"
    assert sorted(set(_compose_service_args().values())) == sorted(SERVICES)


def _resolve_group(groups: dict, name: str) -> set[str]:
    """Package names a dependency group installs, following every `include-group`."""
    packages = set()
    for entry in groups[name]:
        if isinstance(entry, dict):
            packages |= _resolve_group(groups, entry["include-group"])
        else:
            packages.add(re.match(r"[A-Za-z0-9_.-]+", entry)[0].lower())
    return packages


APP_PACKAGES = {"fastapi", "uvicorn", "httpx", "pydantic", "pydantic-settings", "pyyaml"}
EXPECTED_GROUP_PACKAGES = {
    "gateway": APP_PACKAGES | {"litellm", "langfuse"},
    "agent": APP_PACKAGES | {"litellm", "langfuse"},
    "corpus": APP_PACKAGES | {"qdrant-client"},
    "ingestion": APP_PACKAGES | {"qdrant-client", "pypdf", "python-docx", "python-multipart"},
    "math-verifier": {"mcp", "starlette", "sympy"},
}
EVAL_ONLY_PACKAGES = {"langgraph", "python-dotenv", "pytest", "numpy"}


def test_every_service_group_resolves_to_exactly_the_packages_it_imports():
    """
    `dependencies` stays empty so an image never installs what its service does not import,
    and the eval-only packages sit in `dev` only.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    groups = pyproject["dependency-groups"]
    assert pyproject["project"]["dependencies"] == []
    assert pyproject["tool"]["uv"]["default-groups"] == ["dev"]
    for service in SERVICES:
        group = service.replace("_", "-")
        assert _resolve_group(groups, group) == EXPECTED_GROUP_PACKAGES[group], group
    dev = _resolve_group(groups, "dev")
    assert EVAL_ONLY_PACKAGES <= dev
    for group, packages in EXPECTED_GROUP_PACKAGES.items():
        assert packages <= dev, group


def _fake_repo(root: Path) -> Path:
    """A minimal repo: every service, shared/ with several modules, config/ and prompts/."""
    repo = root / "repo"
    (repo / "services").mkdir(parents=True)
    (repo / "services/__init__.py").write_text("")
    for service in SERVICES:
        package = repo / "services" / service / "src" / service
        (package / "__pycache__").mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "__pycache__/x.pyc").write_bytes(b"")
    shared = repo / "shared/src/shared"
    shared.mkdir(parents=True)
    for module in SHARED_MODULES:
        (shared / f"{module}.py").write_text("")
    for top in ("config", "prompts"):
        (repo / top).mkdir()
        (repo / top / "x.yaml").write_text("")
    return repo


SHARED_MODULES = ("__init__", "math_tool_specs", "config", "models")


def _assemble(service: str, repo: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sh", str(ROOT / "infra/assemble-service.sh"), service, str(repo), str(out)],
        capture_output=True,
        text=True,
    )


def _tree(base: Path) -> set[str]:
    return {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()}


def test_assemble_writes_an_executable_serve_and_healthcheck_for_every_service(tmp_path):
    repo = _fake_repo(tmp_path)
    for service in SERVICES:
        out = tmp_path / f"out-{service}"
        result = _assemble(service, repo, out)
        assert result.returncode == 0, (service, result.stderr)
        for command in ("serve", "healthcheck"):
            assert os.access(out / "bin" / command, os.X_OK), (service, command)
        assert (out / "app/services" / service / "src" / service / "__init__.py").is_file()
        assert (out / "app/services/__init__.py").is_file()
        assert not [p for p in _tree(out / "app") if "__pycache__" in p], service


def test_assemble_ships_only_what_each_service_imports(tmp_path):
    repo = _fake_repo(tmp_path)
    shipped = {}
    for service in SERVICES:
        out = tmp_path / f"out-{service}"
        assert _assemble(service, repo, out).returncode == 0
        shipped[service] = _tree(out / "app")
    verifier_shared = {p for p in shipped["math_verifier"] if p.startswith("shared/")}
    assert verifier_shared == {
        "shared/src/shared/__init__.py",
        "shared/src/shared/math_tool_specs.py",
    }
    assert not any(p.startswith(("config/", "prompts/")) for p in shipped["math_verifier"])
    assert not any(p.startswith("services/agent/") for p in shipped["corpus"])
    assert {f"shared/src/shared/{m}.py" for m in SHARED_MODULES} <= shipped["agent"]
    assert {"config/x.yaml", "prompts/x.yaml"} <= shipped["agent"]
    # The gateway no longer imports agent code (ielts and critics live in shared/).
    assert not any(p.startswith("services/agent/") for p in shipped["gateway"])
    assert "services/gateway/src/gateway/__init__.py" not in shipped["agent"]


def test_assemble_rejects_an_unknown_service(tmp_path):
    result = _assemble("nope", _fake_repo(tmp_path), tmp_path / "out")
    assert result.returncode != 0
    assert not (tmp_path / "out/bin/serve").exists()


def _tokens(line: str) -> list[str]:
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def test_ci_runs_the_import_smoke_in_every_image_the_compose_file_builds():
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    commands = [
        _tokens(line)
        for step in ci["jobs"]["compose-smoke"]["steps"]
        for line in str(step.get("run", "")).splitlines()
        if line.strip()
    ]
    (loop,) = [argv for argv in commands if argv[:3] == ["for", "service", "in"]]
    built = _compose_service_args()
    looped = loop[3 : loop.index(";")]
    assert set(looped) <= set(built)
    assert sorted(built[name] for name in looped) == sorted(SERVICES)
    (smoke,) = [argv for argv in commands if "--entrypoint" in argv]
    assert smoke[0] == "$COMPOSE"
    assert smoke[1:3] == ["run", "--rm"]
    assert smoke[-5:] == ["python", "$service", "-", "<", "infra/import_smoke.py"]


def _steps(workflow: dict) -> list[tuple[str, dict]]:
    return [
        (job_id, step) for job_id, job in workflow["jobs"].items() for step in job.get("steps", [])
    ]


def test_live_eval_workflow_is_manual_only_and_fails_when_the_cli_fails():
    import yaml

    live = yaml.safe_load((ROOT / ".github/workflows/eval-live.yml").read_text())
    triggers = live.get("on", live.get(True))  # YAML reads a bare `on` key as True
    assert list(triggers) == ["workflow_dispatch"]
    (run_step,) = [step for _, step in _steps(live) if "--mode live" in step.get("run", "")]
    # `shell: bash` is what turns on pipefail; without it `... | tee` reports tee's exit code
    # and a regression (exit 1) or aborted run (exit 2) would show as a green job.
    assert run_step.get("shell") == "bash" or "pipefail" in run_step["run"]


def test_ci_runs_only_the_offline_quant_suite():
    import yaml

    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    offline_runs = [step.get("run", "") for step in ci["jobs"]["quant-suite-offline"]["steps"]]
    assert any("--mode offline --suite quant" in run for run in offline_runs)
    for _, step in _steps(ci):
        assert "--mode live" not in step.get("run", "")
        referenced = repr(step.get("env", {})) + repr(step.get("with", {})) + step.get("run", "")
        assert "OPENAI_API_KEY" not in referenced
        assert "OPENROUTER_API_KEY" not in referenced


def test_compose_app_services_run_the_dockerfile_cmd_and_reload_through_uvicorn_env():
    """
    The four FastAPI services take their `uvicorn <app> --host --port` from the Dockerfile CMD;
    compose adds only auto-reload, as the UVICORN_* env uvicorn reads for its own options.
    """
    compose = yaml.safe_load((ROOT / "infra/docker-compose.yml").read_text())
    ports = {"gateway": 8000, "corpus": 8001, "agent": 8002, "ingestion": 8003}
    for svc, port in ports.items():
        service = compose["services"][svc]
        assert "command" not in service, svc
        assert service["environment"]["UVICORN_RELOAD"] == "true", svc
        assert service["environment"]["UVICORN_RELOAD_DIRS"] == "/app/services:/app/shared", svc
        assert f"{port}:{port}" in service["ports"], svc
