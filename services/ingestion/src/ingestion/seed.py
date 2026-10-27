"""
Seed the global corpus from the self-written notes in seed/<test_type>/*.txt.

Run once by the `seed` service in infra/docker-compose.yml, after the embedder and the
ingestion service are up: python -m services.ingestion.src.ingestion.seed
Idempotent: does nothing when the corpus collection already holds points, so a restart
never duplicates chunks. To re-seed, remove the Qdrant volume (`docker compose down -v`).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import httpx

from shared.src.shared.config import get_settings
from shared.src.shared.vectorstore import COLLECTION, get_qdrant_client

logger = logging.getLogger("seed")

SEED_DIR = Path(os.environ.get("SEED_DIR", "seed"))
TEST_TYPE = "gre"
# The first embedding call can wait on a cold model download (BGE-M3 is ~2GB).
_READY_TIMEOUT_S = 900.0
_RETRY_DELAY_S = 5.0


async def corpus_is_seeded() -> bool:
    client = get_qdrant_client()
    try:
        if not await client.collection_exists(COLLECTION):
            return False
        return (await client.count(collection_name=COLLECTION)).count > 0
    finally:
        await client.close()


async def upload(client: httpx.AsyncClient, base_url: str, path: Path) -> None:
    """Upload one note, retrying until the embedder behind ingestion is ready."""
    deadline = asyncio.get_running_loop().time() + _READY_TIMEOUT_S
    while True:
        try:
            resp = await client.post(
                f"{base_url}/admin/documents",
                files={"file": (path.name, path.read_bytes(), "text/plain")},
                data={"test_type": TEST_TYPE, "tier": "global"},
            )
            resp.raise_for_status()
            return
        except httpx.HTTPError:
            if asyncio.get_running_loop().time() > deadline:
                raise
            logger.info("ingestion or embedder not ready for %s; retrying", path.name)
            await asyncio.sleep(_RETRY_DELAY_S)


async def main() -> int:
    notes = sorted((SEED_DIR / "gre-quant").glob("*.txt"))
    if not notes:
        logger.error("no seed notes found under %s", SEED_DIR)
        return 1
    if await corpus_is_seeded():
        logger.info("corpus already seeded; nothing to do")
        return 0
    base_url = get_settings().ingestion_service_url.rstrip("/")
    async with httpx.AsyncClient(timeout=120.0) as client:
        for path in notes:
            await upload(client, base_url, path)
            logger.info("ingested %s", path.name)
    logger.info("seeded %d notes", len(notes))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    sys.exit(asyncio.run(main()))
