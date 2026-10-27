"""
Seed the global corpus from the self-written notes in seed/<test_type>/*.txt.

Run once by the `seed` service in infra/docker-compose.yml, after the embedder and the
ingestion service are up: python -m services.ingestion.src.ingestion.seed
Idempotent per note: a note already indexed as a global-tier source is skipped, so a
restart never duplicates chunks and a re-run finishes an interrupted seed. Ingestion
upserts all chunks of a note in one call, so a note is either fully indexed or absent.
To re-seed from scratch, remove the Qdrant volume (`docker compose down -v`).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import httpx
from qdrant_client import models

from shared.src.shared.config import get_settings
from shared.src.shared.vectorstore import COLLECTION, get_qdrant_client

logger = logging.getLogger("seed")

SEED_DIR = Path(os.environ.get("SEED_DIR", "seed"))
TEST_TYPE = "gre"
# The first embedding call can wait on a cold model download (BGE-M3 is ~2GB).
_READY_TIMEOUT_S = 900.0
_RETRY_DELAY_S = 5.0


class SeedError(RuntimeError):
    """A note could not be ingested."""


async def missing_notes(notes: list[Path]) -> list[Path]:
    """The notes that have no global-tier chunks in the corpus yet."""
    client = get_qdrant_client()
    try:
        if not await client.collection_exists(COLLECTION):
            return notes
        missing = []
        for path in notes:
            indexed = await client.count(
                collection_name=COLLECTION,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_uri", match=models.MatchValue(value=path.name)
                        ),
                        models.FieldCondition(key="tier", match=models.MatchValue(value="global")),
                        models.FieldCondition(
                            key="test_type", match=models.MatchValue(value=TEST_TYPE)
                        ),
                    ]
                ),
                exact=True,
            )
            if indexed.count == 0:
                missing.append(path)
        return missing
    finally:
        await client.close()


async def upload(client: httpx.AsyncClient, base_url: str, path: Path) -> None:
    """
    Upload one note. Transport errors and 5xx (the embedder behind ingestion is not ready
    yet) are retried until the deadline; a 4xx is a bad request and fails at once.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _READY_TIMEOUT_S
    while True:
        try:
            resp = await client.post(
                f"{base_url}/admin/documents",
                files={"file": (path.name, path.read_bytes(), "text/plain")},
                data={"test_type": TEST_TYPE, "tier": "global"},
            )
        except httpx.TransportError as exc:
            reason = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code < 500:
                if resp.is_error:
                    raise SeedError(
                        f"ingestion rejected {path.name}: HTTP {resp.status_code} {resp.text}"
                    )
                return
            reason = f"HTTP {resp.status_code}"
        if loop.time() > deadline:
            raise SeedError(f"gave up on {path.name} after {_READY_TIMEOUT_S:.0f}s: {reason}")
        logger.info("ingestion or embedder not ready for %s (%s); retrying", path.name, reason)
        await asyncio.sleep(_RETRY_DELAY_S)


async def main() -> int:
    notes = sorted((SEED_DIR / "gre-quant").glob("*.txt"))
    if not notes:
        logger.error("no seed notes found under %s", SEED_DIR)
        return 1
    todo = await missing_notes(notes)
    if not todo:
        logger.info("corpus already seeded; nothing to do")
        return 0
    base_url = get_settings().ingestion_service_url.rstrip("/")
    async with httpx.AsyncClient(timeout=120.0) as client:
        for path in todo:
            try:
                await upload(client, base_url, path)
            except SeedError as exc:
                logger.error("%s", exc)
                return 1
            logger.info("ingested %s", path.name)
    logger.info("seeded %d of %d notes", len(todo), len(notes))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    sys.exit(asyncio.run(main()))
