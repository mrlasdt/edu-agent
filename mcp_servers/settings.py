"""Settings for the admin and candidate MCP servers: the services they call."""
from __future__ import annotations

from functools import lru_cache

from shared.src.shared.config import ServiceSettings


class McpSettings(ServiceSettings):
    corpus_service_url: str = "http://localhost:8001"
    ingestion_service_url: str = "http://localhost:8003"
    ingestion_admin_key: str = ""


@lru_cache
def get_settings() -> McpSettings:
    return McpSettings()
