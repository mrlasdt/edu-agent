"""
The model config files and how the services pick one.

`MODEL_CONFIG_PATH` is the one switch every service honours (the chat path, the critics, the
IELTS mock, the evals), so `config/model_config.free.yaml` can run the whole stack on a free
OpenRouter model with no OpenAI key.
"""
from unittest.mock import patch

import pytest
import yaml

from shared.src.shared.config import LlmSettings, anchor, load_model_config

FREE = "config/model_config.free.yaml"
DEV = "config/model_config.dev.yaml"
_SETTINGS = "shared.src.shared.config.get_llm_settings"


def test_without_a_path_the_configured_one_is_used():
    with patch(_SETTINGS, return_value=LlmSettings(model_config_path=FREE)):
        cfg = load_model_config()
    assert cfg.role("orchestrator").provider == "openrouter"
    assert cfg.litellm_model("quant_agent").startswith("openrouter/")


def test_default_path_is_the_dev_config():
    with patch(_SETTINGS, return_value=LlmSettings(model_config_path=DEV)):
        cfg = load_model_config()
    assert cfg.role("orchestrator").provider == "openai"


def test_an_explicit_path_wins_over_the_setting():
    with patch(_SETTINGS, return_value=LlmSettings(model_config_path=FREE)):
        cfg = load_model_config(DEV)
    assert cfg.role("orchestrator").provider == "openai"


IELTS_ROLES = {"ielts_grader", "ielts_feedback"}
IELTS_MODEL = "gemini/gemma-4-26b-a4b-it"
TUTOR_BACKUP = "openrouter/google/gemma-4-31b-it:free"
IELTS_BACKUPS = [
    "openrouter/google/gemma-4-26b-a4b-it:free",
    "openrouter/google/gemma-4-31b-it:free",
]


# The role -> temperature table both configs carried line by line before the YAML anchors, and
# the (provider, model, fallbacks) each group of roles resolved to.
TEMPERATURES = {
    "orchestrator": 0,
    "quant_agent": 0.2,
    "aw_agent": 0.5,
    "math_retry": 0.2,
    "math_escalation": 1,
    "style_critic": 0,
    "aw_judge_nightly": 0,
    "aw_judge_pr": 0,
    "question_planner": 0,
    "question_generator": 0.4,
    "question_critic": 0,
    "ielts_grader": 0.3,
    "ielts_feedback": 0.3,
    "style_extractor": 0,
}
MODELS = {
    DEV: {"default": ("openai", "gpt-4o-mini", []), "ielts": ("openai", "gpt-4.1-mini", [])},
    FREE: {
        "default": ("openrouter", "recursive/flint-v1.0:free", []),
        "ielts": ("gemini", "gemma-4-26b-a4b-it", IELTS_BACKUPS),
    },
}
PROMPT_PINS = {
    "quant_agent": {"tutor": "v3", "solve": "v2", "formalize": "v1"},
    "quant_agent_math_tools": {"tutor": "v1", "solve": "v1"},
    "quant_agent_notes_tool": {"tutor": "v1"},
    "aw_agent": {"tutor": "v1", "solve": "v1"},
    "ielts_grader": {"writing_t2": "v1"},
    "ielts_feedback": {"writing_t2": "v2"},
    "orchestrator": {"completeness": "v1"},
    "style_critic": {"rubric": "v1"},
    "question_pipeline": {
        "planner": "v1",
        "numeric_expert": "v1",
        "choice_expert": "v1",
        "comparison_expert": "v1",
        "critic": "v1",
        "baseline": "v1",
    },
}


@pytest.mark.parametrize("path", [DEV, FREE])
def test_each_config_resolves_to_the_same_roles_models_and_temperatures_as_before(path):
    cfg = load_model_config(path)
    assert cfg.roles.keys() == TEMPERATURES.keys()
    for name, role in cfg.roles.items():
        group = "ielts" if name in IELTS_ROLES else "default"
        provider, model, fallbacks = MODELS[path][group]
        if path == FREE and name == "quant_agent":
            fallbacks = [TUTOR_BACKUP]
        assert (role.provider, role.model, role.fallbacks) == (provider, model, fallbacks), name
        assert role.temperature == TEMPERATURES[name], name


@pytest.mark.parametrize("path", [DEV, FREE])
def test_both_configs_get_the_one_shared_set_of_prompt_pins(path):
    assert yaml.safe_load(open(path).read()).get("prompts") is None
    assert load_model_config(path).prompts == PROMPT_PINS
    assert load_model_config(DEV).prompts == load_model_config(FREE).prompts


def test_a_config_with_its_own_prompts_section_keeps_it(tmp_path):
    path = tmp_path / "model_config.yaml"
    path.write_text(
        "roles:\n  orchestrator: {provider: openai, model: m}\n"
        "prompts:\n  quant_agent: {tutor: v9}\n"
    )
    cfg = load_model_config(str(path))
    assert cfg.prompts == {"quant_agent": {"tutor": "v9"}}


def test_free_config_runs_every_role_but_the_ielts_ones_on_one_free_model():
    cfg = load_model_config(FREE)
    models = {(r.provider, r.model) for name, r in cfg.roles.items() if name not in IELTS_ROLES}
    assert len(models) == 1
    provider, model = next(iter(models))
    assert provider == "openrouter"
    assert model.endswith(":free")


def test_free_config_gives_the_tutor_a_different_model_to_fall_back_on():
    cfg = load_model_config(FREE)
    assert cfg.litellm_fallbacks("quant_agent") == [TUTOR_BACKUP]
    assert cfg.litellm_model("quant_agent") != TUTOR_BACKUP
    # the Router keys a chain on the model: the roles that share the tutor's model share its backup
    from shared.src.shared.llm_router import fallback_chains

    assert fallback_chains(cfg)[cfg.litellm_model("quant_agent")] == [TUTOR_BACKUP]
    assert cfg.litellm_fallbacks("orchestrator") == []  # the role's own list stays empty
    assert load_model_config(DEV).litellm_fallbacks("quant_agent") == []  # dev has one model


def test_free_config_grades_ielts_on_the_gated_gemma_with_the_other_gemma_as_backup():
    cfg = load_model_config(FREE)
    for role in IELTS_ROLES:
        assert cfg.litellm_model(role) == IELTS_MODEL
        assert cfg.litellm_fallbacks(role) == IELTS_BACKUPS
    assert cfg.litellm_fallbacks("orchestrator") == []
    assert load_model_config(DEV).litellm_fallbacks("ielts_grader") == []


def test_pinning_one_model_for_an_eval_drops_the_role_fallbacks():
    from evals.agent_eval.runner import write_model_config

    pinned = load_model_config(write_model_config("openai/gpt-4o-mini"))
    for role in pinned.roles:
        assert pinned.litellm_model(role) == "openai/gpt-4o-mini"
        assert pinned.litellm_fallbacks(role) == []


def test_default_config_is_the_free_one(monkeypatch):
    monkeypatch.delenv("MODEL_CONFIG_PATH", raising=False)
    assert LlmSettings(_env_file=None).model_config_path == str(anchor(FREE))


def test_free_eval_chain_starts_with_the_product_default(monkeypatch):
    from evals.agent_eval import cli
    from shared.src.shared.config import clear_settings_caches

    monkeypatch.setenv("MODEL_CONFIG_PATH", FREE)
    clear_settings_caches()
    chain = cli.provider_chain("free")
    assert chain[0] == "openrouter/recursive/flint-v1.0:free"
    assert len(chain) == len(set(chain))


def test_pinning_a_model_works_from_any_directory(tmp_path, monkeypatch):
    from evals.agent_eval.runner import write_model_config

    monkeypatch.chdir(tmp_path)
    pinned = load_model_config(write_model_config("openai/gpt-4o-mini"))
    assert {pinned.litellm_model(role) for role in pinned.roles} == {"openai/gpt-4o-mini"}


def test_pinning_a_model_starts_from_the_configured_model_config(tmp_path, monkeypatch):
    from evals.agent_eval.runner import write_model_config
    from shared.src.shared.config import clear_settings_caches

    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump({"roles": {"only_role": {"provider": "p", "model": "m"}}}))
    monkeypatch.setenv("MODEL_CONFIG_PATH", str(custom))
    clear_settings_caches()
    pinned = load_model_config(write_model_config("openai/gpt-4o-mini"))
    assert list(pinned.roles) == ["only_role"]
