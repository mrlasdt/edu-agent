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


def test_free_config_covers_every_dev_role_and_prompt_version_on_one_free_model():
    free = yaml.safe_load(open(FREE).read())
    dev = yaml.safe_load(open(DEV).read())
    assert free["roles"].keys() == dev["roles"].keys()
    assert free["prompts"] == dev["prompts"]
    models = {(r["provider"], r["model"]) for r in free["roles"].values()}
    assert len(models) == 1
    provider, model = next(iter(models))
    assert provider == "openrouter"
    assert model.endswith(":free")


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
