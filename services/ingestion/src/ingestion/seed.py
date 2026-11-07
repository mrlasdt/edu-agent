"""
Seed the global corpus from the self-written notes in seed/<test_type>/*.txt.

Run once by the `seed` service in infra/docker-compose.yml, after the embedder and the
ingestion service are up: python -m services.ingestion.src.ingestion.seed
Idempotent: every run uploads every note, and ingestion gives each chunk a stable point id
(see `chunk_point_id`), so a note that is already indexed is replaced in place, never
duplicated, and a re-run finishes an interrupted seed. A restart therefore re-embeds the notes
once; that is the price of not asking Qdrant first.
To re-seed from scratch, remove the Qdrant volume (`docker compose down -v`).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import httpx

from shared.src.shared.config import ADMIN_KEY_HEADER, get_settings

logger = logging.getLogger("seed")

SEED_DIR = Path(os.environ.get("SEED_DIR", "seed"))
TEST_TYPE = "gre"
# The first embedding call can wait on a cold model download (BGE-M3 is ~2GB).
_READY_TIMEOUT_S = 900.0
_RETRY_DELAY_S = 5.0


class SeedError(RuntimeError):
    """A note could not be ingested."""


async def upload(client: httpx.AsyncClient, base_url: str, path: Path) -> None:
    """
    Upload one note. The ingestion service files it under the global tier; the client's
    default headers carry the admin key. Transport errors and 5xx (the embedder behind
    ingestion is not ready yet) are retried until the deadline; a 4xx is a bad request and
    fails at once.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _READY_TIMEOUT_S
    while True:
        try:
            resp = await client.post(
                f"{base_url}/admin/documents",
                files={"file": (path.name, path.read_bytes(), "text/plain")},
                data={"test_type": TEST_TYPE},
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
    settings = get_settings()
    if not settings.ingestion_admin_key:
        logger.error("INGESTION_ADMIN_KEY is not set; add it to .env (see .env.example)")
        return 1
    base_url = settings.ingestion_service_url.rstrip("/")
    headers = {ADMIN_KEY_HEADER: settings.ingestion_admin_key}
    async with httpx.AsyncClient(timeout=120.0, headers=headers) as client:
        for path in notes:
            try:
                await upload(client, base_url, path)
            except SeedError as exc:
                logger.error("%s", exc)
                return 1
            logger.info("ingested %s", path.name)
    logger.info("seeded %d notes", len(notes))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    sys.exit(asyncio.run(main()))
