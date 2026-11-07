"""
Tests for CI/CD infrastructure (issue 16).

Behaviors under test:
  1. GitHub Actions workflows exist and are valid YAML
  2. ci.yml runs the offline eval on every change (no path gate)
  3. No Kubernetes manifests or cluster deploy workflow (Docker Compose is the one deploy path)
  4. Each FastAPI service exposes GET /healthz → 200
  5. Dockerfiles exist for all services
"""

import shlex
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


# ── 5. Dockerfiles ────────────────────────────────────────────────────────────


def test_dockerfiles_exist_for_all_services():
    services = ["gateway", "agent", "corpus", "ingestion", "math_verifier"]
    for svc in services:
        df = ROOT / "services" / svc / "Dockerfile"
        assert df.exists(), f"Missing Dockerfile for {svc}"


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
