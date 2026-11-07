"""
The model config files and how the services pick one.

`MODEL_CONFIG_PATH` is the one switch every service honours (the chat path, the critics, the
IELTS mock, the evals), so `config/model_config.free.yaml` can run the whole stack on a free
OpenRouter model with no OpenAI key.
"""
from unittest.mock import patch

import yaml

from shared.src.shared.config import Settings, load_model_config

FREE = "config/model_config.free.yaml"
DEV = "config/model_config.dev.yaml"
_SETTINGS = "shared.src.shared.config.get_settings"


def test_without_a_path_the_configured_one_is_used():
    with patch(_SETTINGS, return_value=Settings(model_config_path=FREE)):
        cfg = load_model_config()
    assert cfg.role("orchestrator").provider == "openrouter"
    assert cfg.litellm_model("quant_agent").startswith("openrouter/")


def test_default_path_is_the_dev_config():
    with patch(_SETTINGS, return_value=Settings(model_config_path=DEV)):
        cfg = load_model_config()
    assert cfg.role("orchestrator").provider == "openai"


def test_an_explicit_path_wins_over_the_setting():
    with patch(_SETTINGS, return_value=Settings(model_config_path=FREE)):
        cfg = load_model_config(DEV)
    assert cfg.role("orchestrator").provider == "openai"


IELTS_ROLES = {"ielts_grader", "ielts_feedback"}
IELTS_MODEL = "google/gemma-4-26b-a4b-it:free"
IELTS_BACKUP = "openrouter/google/gemma-4-31b-it:free"


def test_free_config_covers_every_dev_role_and_prompt_version_on_one_free_model():
    free = yaml.safe_load(open(FREE).read())
    dev = yaml.safe_load(open(DEV).read())
    assert free["roles"].keys() == dev["roles"].keys()
    assert free["prompts"] == dev["prompts"]
    models = {
        (r["provider"], r["model"]) for name, r in free["roles"].items() if name not in IELTS_ROLES
    }
    assert len(models) == 1
    provider, model = next(iter(models))
    assert provider == "openrouter"
    assert model.endswith(":free")


def test_free_config_grades_ielts_on_the_gated_gemma_with_the_other_gemma_as_backup():
    cfg = load_model_config(FREE)
    for role in IELTS_ROLES:
        assert cfg.litellm_model(role) == f"openrouter/{IELTS_MODEL}"
        assert cfg.litellm_fallbacks(role) == [IELTS_BACKUP]
    assert cfg.litellm_fallbacks("orchestrator") == []
    assert load_model_config(DEV).litellm_fallbacks("ielts_grader") == []


def test_pinning_one_model_for_an_eval_drops_the_role_fallbacks():
    from evals.agent_eval.runner import write_model_config

    pinned = load_model_config(write_model_config("openai/gpt-4o-mini"))
    for role in pinned.roles:
        assert pinned.litellm_model(role) == "openai/gpt-4o-mini"
        assert pinned.litellm_fallbacks(role) == []


def test_default_config_is_the_free_one(monkeypatch):
    from shared.src.shared.config import Settings

    monkeypatch.delenv("MODEL_CONFIG_PATH", raising=False)
    assert Settings(_env_file=None).model_config_path == FREE


def test_free_eval_chain_starts_with_the_product_default(monkeypatch):
    from evals.agent_eval import cli
    from shared.src.shared.config import get_settings

    monkeypatch.setenv("MODEL_CONFIG_PATH", FREE)
    get_settings.cache_clear()
    chain = cli.provider_chain("free")
    assert chain[0] == "openrouter/recursive/flint-v1.0:free"
    assert len(chain) == len(set(chain))
