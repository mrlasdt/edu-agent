from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import AfterValidator, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# The directory holding config/, prompts/ and .env: the repo root here, /app in the images. Every
# default path resolves against it, never against the process's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def anchor(path: str | Path) -> Path:
    """`path` as an absolute path: a relative one is taken from PROJECT_ROOT, not from the cwd."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _anchored(value: str) -> str:
    # An empty value is meaningful ("off") for the log paths, so it stays empty.
    return str(anchor(value)) if value.strip() else value


# A path setting kept as a string: relative values (the defaults, or MODEL_CONFIG_PATH=config/x.yaml
# in `.env`) are taken from PROJECT_ROOT; an absolute value from the environment is used as is.
AnchoredPath = Annotated[str, AfterValidator(_anchored)]

# Request header carrying the ingestion admin key (IngestionSettings) to the /admin routes.
ADMIN_KEY_HEADER = "X-Admin-Key"

# Defaults that more than one module needs; the settings classes and every fallback read these.
DEFAULT_MODEL_CONFIG_PATH = "config/model_config.free.yaml"
# The active prompt version per (agent, mode), shared by every model config that has no
# `prompts:` section of its own.
PROMPT_PINS_PATH = "config/prompt_pins.yaml"
DEFAULT_TURN_LOG_PATH = ".turn-log/turns.jsonl"
DEFAULT_FEEDBACK_LOG_PATH = ".turn-log/feedback.jsonl"
DEFAULT_TURN_CAPTURE_PATH = ".turn-log/capture.jsonl"
DEFAULT_AI_LEDGER_PATH = ".eval-ledger/ledger.jsonl"


class ModelRoleConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0.0
    fallbacks: list[str] = []


class ModelConfig(BaseModel):
    roles: dict[str, ModelRoleConfig]
    # Active prompt version per agent → mode (e.g. {"quant_agent": {"tutor": "v1"}}).
    # Absent entries default to "v1", so a config without this section stays valid.
    # load_model_config fills it from PROMPT_PINS_PATH when the config file has no section.
    prompts: dict[str, dict[str, str]] = {}

    def role(self, name: str) -> ModelRoleConfig:
        if name not in self.roles:
            raise KeyError(
                f"Model role '{name}' not found in model_config. Known roles: {list(self.roles)}"
            )
        return self.roles[name]

    def litellm_model(self, role: str) -> str:
        """Return the model string LiteLLM expects, e.g. 'openai/gpt-4o'."""
        cfg = self.role(role)
        return f"{cfg.provider}/{cfg.model}"

    def litellm_fallbacks(self, role: str) -> list[str]:
        """LiteLLM model strings tried in order after the role's own model fails."""
        return list(self.role(role).fallbacks)

    def prompt_version(self, agent: str, mode: str) -> str:
        """Active prompt version for (agent, mode). Defaults to 'v1' when unset."""
        return self.prompts.get(agent, {}).get(mode, "v1")


def load_model_config(path: str | None = None) -> ModelConfig:
    """
    The model config at `path`; without one, the file named by MODEL_CONFIG_PATH (default
    config/model_config.free.yaml, every role on a free OpenRouter model), so every service
    honours the same switch (config/model_config.dev.yaml is the paid OpenAI config).
    A config with no `prompts:` section gets the shared pins in config/prompt_pins.yaml.
    """
    return _load_model_config(str(anchor(path or get_llm_settings().model_config_path)))


@lru_cache
def _load_model_config(path: str) -> ModelConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    if "prompts" not in raw:
        raw["prompts"] = yaml.safe_load(anchor(PROMPT_PINS_PATH).read_text())
    return ModelConfig(**raw)


class ServiceSettings(BaseSettings):
    """
    What every settings class has in common: where `.env` is and how it is read.

    Each service has its own subclass (GatewaySettings, CorpusSettings, ...) that declares only
    the variables that service reads, so it validates nothing else and a bad value meant for
    another service cannot stop it. Variable names are the field names upper-cased and are the
    same as before the split, so `.env` files and docker compose keep working unchanged.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )


# Field groups used by more than one service, or by `shared/` modules that run inside several.
# A service class inherits the groups it uses; a shared module reads the one group it needs
# through the `get_*_settings` accessor below.


class LlmSettings(ServiceSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    # Google AI Studio key: the IELTS mock's default models run on Google directly (provider
    # `gemini`). Empty = those calls fail fast and the role's OpenRouter fallbacks grade instead.
    gemini_api_key: str = ""
    model_config_path: AnchoredPath = DEFAULT_MODEL_CONFIG_PATH


class TurnLogSettings(ServiceSettings):
    # Per-turn measurement of the live chat path (shared.turn_meter): one JSON line per service
    # per turn, read by `python -m evals.turn_report`. Recorded, never capped. Empty = off.
    turn_log_path: AnchoredPath = DEFAULT_TURN_LOG_PATH


class FeedbackSettings(TurnLogSettings):
    # Thumbs up/down from the web app (POST /feedback), one JSON line each, keyed by trace id.
    # No message text. Empty = /feedback is switched off.
    feedback_log_path: AnchoredPath = DEFAULT_FEEDBACK_LOG_PATH
    # Opt-in local capture of the learner's question and the reply, keyed by trace id, so a
    # thumbs-down can become a complete draft eval item (evals.feedback_to_item). Off by default:
    # the turn log never holds message text. The file is local and gitignored, never sent anywhere.
    turn_capture_text: bool = False
    turn_capture_path: AnchoredPath = DEFAULT_TURN_CAPTURE_PATH


class SpendSettings(ServiceSettings):
    # AI spend cap (USD) for the IELTS mock and the evals, and the ledger it is counted in.
    # Both share the ledger, so the cap is one number; the GRE tutor's /chat is not capped.
    ai_spend_cap_usd: float = 5.0
    ai_ledger_path: AnchoredPath = DEFAULT_AI_LEDGER_PATH


class LangfuseSettings(ServiceSettings):
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "http://localhost:3000"


@lru_cache
def get_llm_settings() -> LlmSettings:
    return LlmSettings()


@lru_cache
def get_turn_log_settings() -> TurnLogSettings:
    return TurnLogSettings()


@lru_cache
def get_feedback_settings() -> FeedbackSettings:
    return FeedbackSettings()


@lru_cache
def get_spend_settings() -> SpendSettings:
    return SpendSettings()


@lru_cache
def get_langfuse_settings() -> LangfuseSettings:
    return LangfuseSettings()


def clear_settings_caches() -> None:
    """Forget the cached group settings and model configs (tests do this after changing env)."""
    for accessor in (
        get_llm_settings, get_turn_log_settings, get_feedback_settings, get_spend_settings,
        get_langfuse_settings, _load_model_config,
    ):
        accessor.cache_clear()
