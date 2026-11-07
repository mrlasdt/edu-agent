"""The ingestion service's settings (and the seed job's): only what indexing reads."""
from __future__ import annotations

from functools import lru_cache

from shared.src.shared.config import AnchoredPath, ServiceSettings


class IngestionSettings(ServiceSettings):
    qdrant_url: str = "http://localhost:6333"
    tei_embedder_url: str = "http://localhost:8080"
    ingestion_service_url: str = "http://localhost:8003"
    # Shared secret for the /admin routes (sent in ADMIN_KEY_HEADER). Empty = not configured:
    # the routes then refuse every request (fail closed) instead of staying open.
    ingestion_admin_key: str = ""
    # The notes the seed job ingests (docker compose sets SEED_DIR=/app/seed).
    seed_dir: AnchoredPath = "seed"


@lru_cache
def get_settings() -> IngestionSettings:
    return IngestionSettings()
