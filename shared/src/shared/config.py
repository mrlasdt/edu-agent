from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# Request header carrying Settings.ingestion_admin_key to the ingestion service's /admin routes.
ADMIN_KEY_HEADER = "X-Admin-Key"


class ModelRoleConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0.0
    fallbacks: list[str] = []


class ModelConfig(BaseModel):
    roles: dict[str, ModelRoleConfig]
    # Active prompt version per agent → mode (e.g. {"quant_agent": {"tutor": "v1"}}).
    # Absent entries default to "v1", so a config without this section stays valid.
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
    """
    return _load_model_config(path or get_settings().model_config_path)


@lru_cache
def _load_model_config(path: str) -> ModelConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return ModelConfig(**raw)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    database_url: str = "postgresql+asyncpg://edu:edu@localhost:5432/edu_agent"
    qdrant_url: str = "http://localhost:6333"
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    tei_embedder_url: str = "http://localhost:8080"
    # Dense + sparse fused by RRF (ADR-0004). false = dense-only queries. Not a
    # rollback to the old collection layout: both modes need the named
    # dense/sparse vectors, so a legacy collection must be recreated and re-ingested.
    hybrid_retrieval_enabled: bool = True
    # BGE-reranker-v2-m3 cross-encoder over the RRF top candidates (ADR-0004).
    # Empty URL = no rerank.
    tei_reranker_url: str = ""
    corpus_service_url: str = "http://localhost:8001"
    agent_service_url: str = "http://localhost:8002"
    ingestion_service_url: str = "http://localhost:8003"
    math_verifier_url: str = "http://localhost:8090"
    admin_username: str = "admin"
    admin_password: str = "changeme"
    # Shared secret for the ingestion service's /admin routes (sent in ADMIN_KEY_HEADER). Empty =
    # not configured: the routes then refuse every request (fail closed) instead of staying open.
    ingestion_admin_key: str = ""
    # IELTS mock: attempts database and other local-only files (gitignored), and the
    # number of grader samples whose per-criterion median is reported.
    ielts_data_dir: str = "ielts-local"
    ielts_grader_samples: int = 3
    # AI spend cap (USD) for the IELTS mock and the evals, and the ledger it is counted in.
    # Both share the ledger, so the cap is one number; the GRE tutor's /chat is not capped.
    ai_spend_cap_usd: float = 5.0
    ai_ledger_path: str = ".eval-ledger/ledger.json"
    # Per-turn measurement of the live chat path (shared.turn_meter): one JSON line per service
    # per turn, read by `python -m evals.turn_report`. Recorded, never capped. Empty = off.
    turn_log_path: str = ".turn-log/turns.jsonl"
    # Thumbs up/down from the web app (POST /feedback), one JSON line each, keyed by trace id.
    # No message text. Empty = /feedback is switched off.
    feedback_log_path: str = ".turn-log/feedback.jsonl"
    # Opt-in local capture of the learner's question and the reply, keyed by trace id, so a
    # thumbs-down can become a complete draft eval item (evals.feedback_to_item). Off by default:
    # the turn log never holds message text. The file is local and gitignored, never sent anywhere.
    turn_capture_text: bool = False
    turn_capture_path: str = ".turn-log/capture.jsonl"
    # Math tools for the Quant agent (math_evaluate, math_solve, math_compare): the model
    # computes with sympy instead of in its head. Off by default: a tool turn is not streamed and
    # costs more tokens (README, "Math tools"). true = the tool loop.
    math_tools_enabled: bool = False
    env: str = "dev"
    # What the gateway does when the content-safety (moderation) call itself fails: "closed"
    # answers "safety check unavailable, try again" and stops the turn, "open" lets the turn
    # go on and records `moderation_failed_open`. Empty = closed everywhere except ENV=dev.
    moderation_on_error: Literal["", "closed", "open"] = ""
    # Dev-only: a message containing this text is treated as a self-harm moderation hit, so the
    # supportive reply can be tried without a paid moderation key. Ignored unless ENV=dev.
    safety_test_trigger: str = ""
    model_config_path: str = "config/model_config.free.yaml"


@lru_cache
def get_settings() -> Settings:
    return Settings()
