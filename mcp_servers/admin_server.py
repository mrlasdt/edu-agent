"""
Admin MCP server.

Allows an Admin to trigger corpus ingestion and check job status from a
Claude Desktop session (or any MCP-capable client), without needing the
Admin UI open.

Tools:
  - ingest_document(file_path, test_type, tier) → job_id
  - list_ingestion_jobs(limit) → list of jobs
  - get_job_status(job_id) → job state

Run with:
    python -m mcp_servers.admin_server
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

from shared.src.shared.config import get_settings

mcp = FastMCP(
    "gre-tutor-admin",
    instructions="GRE corpus ingestion management",
    host="0.0.0.0",
    port=8092,
)


@mcp.tool()
async def ingest_document_tool(
    file_path: str,
    test_type: str,
    tier: str = "global",
) -> dict:
    """
    Ingest a document into the corpus.

    Args:
        file_path: Absolute path to the file on the server (PDF, DOCX, or TXT)
        test_type: Test type for this document (e.g. "gre")
        tier: Corpus tier — "global" (default) or "school"

    Returns:
        {"job_id": str, "status": "queued"}
    """
    job_id = await _ingest_document(
        file_path=file_path, test_type=test_type, tier=tier
    )
    return {"job_id": job_id, "status": "queued"}


@mcp.tool()
async def list_ingestion_jobs_tool(limit: int = 20) -> list[dict]:
    """
    List recent ingestion jobs.

    Args:
        limit: Maximum number of jobs to return (default 20)

    Returns:
        List of job dicts with job_id, filename, test_type, status, error_stage
    """
    return await _list_jobs(limit=limit)


@mcp.tool()
async def get_job_status_tool(job_id: str) -> dict:
    """
    Get the current status of an ingestion job.

    Args:
        job_id: The job identifier returned by ingest_document

    Returns:
        Job dict with status and error_stage, or {"error": "not found"} if unknown
    """
    job = await _get_job(job_id=job_id)
    if job is None:
        return {"error": f"job '{job_id}' not found"}
    return job


# ── service seams (patched in tests) ─────────────────────────────────────────


async def _ingest_document(
    file_path: str, test_type: str, tier: str
) -> str:
    settings = get_settings()
    content = Path(file_path).read_bytes()
    filename = Path(file_path).name
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{settings.ingestion_service_url}/admin/documents",
            files={"file": (filename, content, "application/octet-stream")},
            data={"test_type": test_type, "tier": tier},
        )
        r.raise_for_status()
        return r.json()["job_id"]


async def _list_jobs(limit: int) -> list[dict[str, Any]]:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(
            f"{settings.ingestion_service_url}/admin/documents",
            params={"limit": limit},
        )
        r.raise_for_status()
        return r.json()


async def _get_job(job_id: str) -> dict[str, Any] | None:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(
            f"{settings.ingestion_service_url}/admin/documents/{job_id}"
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    mcp.run(transport="sse")
