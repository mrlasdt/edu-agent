"""
Tests for CI/CD infrastructure (issue 16).

Behaviors under test:
  1. GitHub Actions workflows exist and are valid YAML
  2. ci.yml has path-gated eval step
  3. k8s manifests exist and are valid YAML
  4. Each FastAPI service exposes GET /healthz → 200
  5. Dockerfiles exist for all services
"""

import yaml
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent


# ── 1. GitHub Actions workflows ───────────────────────────────────────────────

def test_ci_workflow_exists():
    assert (ROOT / ".github/workflows/ci.yml").exists()


def test_deploy_workflow_exists():
    assert (ROOT / ".github/workflows/deploy.yml").exists()


def test_ci_workflow_is_valid_yaml():
    content = (ROOT / ".github/workflows/ci.yml").read_text()
    data = yaml.safe_load(content)
    assert isinstance(data, dict)
    assert "jobs" in data


def test_deploy_workflow_is_valid_yaml():
    content = (ROOT / ".github/workflows/deploy.yml").read_text()
    data = yaml.safe_load(content)
    assert isinstance(data, dict)


# ── 2. Path-gated eval in CI ──────────────────────────────────────────────────

def test_ci_workflow_has_path_filter_for_eval():
    content = (ROOT / ".github/workflows/ci.yml").read_text()
    # Should reference path filters for eval-triggering paths
    assert "prompts" in content or "paths" in content


# ── 3. k8s manifests ─────────────────────────────────────────────────────────

def test_k8s_directory_exists():
    assert (ROOT / "infra/k8s").is_dir()


def test_k8s_manifests_are_valid_yaml():
    k8s_dir = ROOT / "infra/k8s"
    manifests = list(k8s_dir.glob("*.yaml")) + list(k8s_dir.glob("*.yml"))
    assert len(manifests) > 0, "No k8s manifests found"
    for manifest in manifests:
        content = manifest.read_text()
        docs = list(yaml.safe_load_all(content))
        assert all(isinstance(doc, dict) for doc in docs if doc is not None), \
            f"Invalid YAML in {manifest.name}"


def test_k8s_has_gateway_manifest():
    k8s_dir = ROOT / "infra/k8s"
    names = {f.stem for f in k8s_dir.glob("*.yaml")}
    assert any("gateway" in n for n in names)


def test_k8s_has_agent_manifest():
    k8s_dir = ROOT / "infra/k8s"
    names = {f.stem for f in k8s_dir.glob("*.yaml")}
    assert any("agent" in n for n in names)


def test_k8s_manifests_use_secret_refs_not_literal_keys():
    k8s_dir = ROOT / "infra/k8s"
    for manifest in k8s_dir.glob("*.yaml"):
        content = manifest.read_text()
        # Should not contain literal API key values
        assert "sk-" not in content, f"{manifest.name} contains a literal API key"


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
