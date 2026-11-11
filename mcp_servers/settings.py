"""Settings for the admin and candidate MCP servers: the services they call, and who is asking."""
from __future__ import annotations

from functools import lru_cache

from shared.src.shared.config import ServiceSettings


class McpSettings(ServiceSettings):
    corpus_service_url: str = "http://localhost:8001"
    ingestion_service_url: str = "http://localhost:8003"
    ingestion_admin_key: str = ""
    # Who the candidate server acts for. The operator sets these in the server's environment (the
    # `env` block of the client's MCP config); the model never supplies them, so it cannot ask for
    # another candidate's notes. Unset means no identity: only the global tier is searched (the
    # corpus gives an empty id no tier at all).
    mcp_candidate_id: str = ""
    mcp_school_id: str = ""


@lru_cache
def get_settings() -> McpSettings:
    return McpSettings()
