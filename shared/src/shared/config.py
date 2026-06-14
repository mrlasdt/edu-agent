from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelRoleConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0.0


class ModelConfig(BaseModel):
    roles: dict[str, ModelRoleConfig]

    def role(self, name: str) -> ModelRoleConfig:
        if name not in self.roles:
            raise KeyError(f"Model role '{name}' not found in model_config. Known roles: {list(self.roles)}")
        return self.roles[name]

    def litellm_model(self, role: str) -> str:
        """Return the model string LiteLLM expects, e.g. 'openai/gpt-4o'."""
        cfg = self.role(role)
        return f"{cfg.provider}/{cfg.model}"


@lru_cache
def load_model_config(path: str | None = None) -> ModelConfig:
    resolved = Path(path) if path else Path("config/model_config.dev.yaml")
    raw: dict[str, Any] = yaml.safe_load(resolved.read_text())
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
    corpus_service_url: str = "http://localhost:8001"
    agent_service_url: str = "http://localhost:8002"
    ingestion_service_url: str = "http://localhost:8003"
    math_verifier_socket: str = "/tmp/math-verifier.sock"
    admin_username: str = "admin"
    admin_password: str = "changeme"
    env: str = "dev"
    model_config_path: str = "config/model_config.dev.yaml"


@lru_cache
def get_settings() -> Settings:
    return Settings()
