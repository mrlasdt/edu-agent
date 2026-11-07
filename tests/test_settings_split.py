"""
Settings are per service, and paths do not depend on the working directory.

(1) A service reads only its own settings: values that belong to another service (here, ones that
    do not even parse) must not stop it from starting.
(2) A default path resolves the same from any working directory, and a `.env` found in the current
    directory is not read.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Env values that only the gateway, the agent or the evals understand, set to garbage.
FOREIGN_GARBAGE = {
    "MODERATION_ON_ERROR": "sometimes",
    "IELTS_GRADER_SAMPLES": "many",
    "AI_SPEND_CAP_USD": "lots",
    "MATH_TOOLS_ENABLED": "perhaps",
    "TURN_CAPTURE_TEXT": "maybe",
}

SERVICES_WITHOUT_THOSE_SETTINGS = [
    "services.corpus.src.corpus.main",
    "services.ingestion.src.ingestion.main",
]


def _reset_shared_settings() -> None:
    """Forget every cached settings object in shared.config (the env just changed)."""
    from shared.src.shared import config

    for obj in vars(config).values():
        if callable(getattr(obj, "cache_clear", None)):
            obj.cache_clear()


def _fresh(module_name: str):
    module = importlib.import_module(module_name)
    module.get_settings.cache_clear()
    return module


@pytest.mark.parametrize("module_name", SERVICES_WITHOUT_THOSE_SETTINGS)
def test_service_starts_without_other_services_settings(module_name, monkeypatch):
    for key, value in FOREIGN_GARBAGE.items():
        monkeypatch.setenv(key, value)
    module = _fresh(module_name)
    try:
        module.get_settings()  # raises pydantic.ValidationError while every service validates all
    finally:
        module.get_settings.cache_clear()


def test_math_verifier_starts_without_any_other_settings(monkeypatch):
    settings = importlib.import_module("services.math_verifier.src.math_verifier.settings")
    server = importlib.import_module("services.math_verifier.src.math_verifier.server")
    for key, value in FOREIGN_GARBAGE.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("MATH_VERIFIER_PORT", "8123")
    settings.get_settings.cache_clear()
    try:
        assert importlib.reload(server).mcp.settings.port == 8123
    finally:  # leave the module as the rest of the suite expects it
        monkeypatch.undo()
        settings.get_settings.cache_clear()
        importlib.reload(server)


def test_math_verifier_imports_with_only_its_shipped_dependencies():
    """The image has no shared/config.py or pyyaml: block them and import the server."""
    script = (
        "import sys\n"
        "BLOCKED = ('shared.src.shared.config', 'yaml')\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name in BLOCKED:\n"
        "            raise ImportError(f'{name} is not in the math-verifier image')\n"
        "sys.meta_path.insert(0, Block())\n"
        "from services.math_verifier.src.math_verifier import server\n"
        "assert not any(m in sys.modules for m in BLOCKED)\n"
        "print(server.mcp.settings.host, server.mcp.settings.port)\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("MATH_VERIFIER_")}
    env.update(FOREIGN_GARBAGE)
    env["MATH_VERIFIER_PORT"] = "8123"
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["127.0.0.1", "8123"]


def test_load_model_config_works_from_another_working_directory(tmp_path, monkeypatch):
    from shared.src.shared.config import load_model_config

    monkeypatch.chdir(tmp_path)
    _reset_shared_settings()
    assert load_model_config().roles  # config/model_config.dev.yaml and prompt_pins.yaml


def test_agent_prompts_load_from_another_working_directory(tmp_path, monkeypatch):
    from services.agent.src.agent.agents import aw, quant
    from services.agent.src.agent.pipeline import llm as pipeline_llm
    from shared.src.shared.ielts import llm as ielts_llm
    from shared.src.shared.models import Mode

    monkeypatch.chdir(tmp_path)
    assert aw._load_prompt(Mode.tutor, "v1")
    assert quant._read_prompt("quant_agent", "tutor", "v1")
    assert ielts_llm.load_prompt("ielts_grader", "writing_t2", "v1")
    assert pipeline_llm.load_prompt("planner", "v1")


def test_ielts_calibration_is_read_from_another_working_directory(tmp_path, monkeypatch):
    from shared.src.shared.ielts import gate

    committed = gate.load_calibration(REPO_ROOT / "config" / "ielts_calibration.json")
    monkeypatch.chdir(tmp_path)
    assert gate.load_calibration() == committed


def test_default_ledger_and_logs_are_anchored_not_cwd_relative(tmp_path, monkeypatch):
    from shared.src.shared.budget import spend_ledger

    monkeypatch.delenv("AI_LEDGER_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    _reset_shared_settings()
    assert spend_ledger().path == REPO_ROOT / ".eval-ledger" / "ledger.jsonl"


def test_explicit_absolute_path_from_env_wins(tmp_path, monkeypatch):
    from shared.src.shared.budget import spend_ledger

    wanted = tmp_path / "elsewhere" / "ledger.jsonl"
    monkeypatch.setenv("AI_LEDGER_PATH", str(wanted))
    monkeypatch.chdir(tmp_path)
    _reset_shared_settings()
    assert spend_ledger().path == wanted


def test_dotenv_in_the_working_directory_is_not_read(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("INGESTION_ADMIN_KEY=from-the-cwd-dotenv\n")
    monkeypatch.delenv("INGESTION_ADMIN_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    module = _fresh("services.ingestion.src.ingestion.main")
    try:
        assert module.get_settings().ingestion_admin_key != "from-the-cwd-dotenv"
    finally:
        module.get_settings.cache_clear()
