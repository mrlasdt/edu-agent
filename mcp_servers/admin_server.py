"""
Admin MCP server.

Allows an Admin to trigger corpus ingestion and check job status from a
Claude Desktop session (or any MCP-capable client), without needing the
Admin UI open.

Tools (named <area>_<verb>):
  - ingestion_submit_document(file_path, test_type) -> job_id
  - ingestion_list_jobs(limit) -> list of jobs
  - ingestion_get_job(job_id) -> job state

A call that cannot be answered raises ToolError, which the client sees as an error result whose
message says what was wrong and what to do next.

Run with (stdio by default; MCP_TRANSPORT=sse serves SSE on 127.0.0.1:8092):
    python -m mcp_servers.admin_server
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from mcp_servers.launch import bind_host, run
from mcp_servers.settings import get_settings
from shared.src.shared.config import ADMIN_KEY_HEADER

mcp = FastMCP(
    "gre-tutor-admin",
    instructions=(
        "GRE corpus ingestion: submit a document (ingestion_submit_document), then follow it "
        "with ingestion_get_job or find it in ingestion_list_jobs."
    ),
    host=bind_host(),
    port=8092,
)


@mcp.tool()
async def ingestion_submit_document(
    file_path: str,
    test_type: str = "gre",
) -> dict[str, Any]:
    """
    Submit a document to be indexed into the global corpus. Small files are indexed before this
    returns (status "completed"); larger ones return "queued" and are indexed in the background,
    so follow those with ingestion_get_job. Needs INGESTION_ADMIN_KEY in this server's environment.

    Args:
        file_path: Absolute path of a PDF, DOCX or TXT file on the machine running this server.
        test_type: The exam the document belongs to; "gre" by default.

    Returns:
        {"job_id": str, "status": "completed" | "queued"}
    """
    path = Path(file_path)
    if not path.is_absolute():
        raise ToolError(
            f"file_path {file_path!r} is not an absolute path. Send the full path, "
            f"e.g. /Users/me/notes/{path.name}."
        )
    if not path.is_file():
        raise ToolError(
            f"no file at {file_path!r}. Check the path; it must be a file, not a folder."
        )
    return await _ingest_document(file_path=file_path, test_type=test_type)


@mcp.tool()
async def ingestion_list_jobs(limit: int = 20) -> dict[str, Any]:
    """
    List the most recent ingestion jobs, newest first. Use it to find a job id or to see what is
    still running or failed; for one job's details use ingestion_get_job.

    Args:
        limit: How many jobs to return, 1 to 100 (default 20).

    Returns:
        {"count": n, "jobs": [{"job_id", "filename", "test_type", "status", "error_stage"}]}
    """
    if not 1 <= limit <= 100:
        raise ToolError(f"limit {limit} is out of range. Use a whole number from 1 to 100.")
    jobs = await _list_jobs(limit=limit)
    return {"count": len(jobs), "jobs": jobs}


@mcp.tool()
async def ingestion_get_job(job_id: str) -> dict[str, Any]:
    """
    Get the current state of one ingestion job: queued, processing, completed or failed, and at
    which stage it failed. Get the job_id from ingestion_submit_document or ingestion_list_jobs.

    Args:
        job_id: The job's identifier.

    Returns:
        The job: job_id, filename, test_type, status, error_stage.
    """
    job = await _get_job(job_id=job_id)
    if job is None:
        raise ToolError(
            f"no ingestion job has id {job_id!r}. Use ingestion_list_jobs to see the ids."
        )
    return job


# ── service seams (patched in tests) ─────────────────────────────────────────


@contextmanager
def _ingestion_service() -> Iterator[None]:
    """Turn a failed call to the ingestion service into an error the model can act on."""
    try:
        yield
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code == 503:
            raise ToolError(
                "admin access is not configured: the ingestion service answers 503 while its "
                "INGESTION_ADMIN_KEY is unset. Set the same key there and in this server's "
                "environment."
            ) from exc
        if code in (401, 403):
            raise ToolError(
                "the ingestion service rejected the admin key. Check that INGESTION_ADMIN_KEY in "
                "this server's environment matches the service's."
            ) from exc
        raise ToolError(f"the ingestion service answered HTTP {code}.") from exc
    except httpx.HTTPError as exc:
        raise ToolError(
            f"the ingestion service is not reachable ({type(exc).__name__}). "
            "Check it is running, then try again."
        ) from exc


def _admin_headers() -> dict[str, str]:
    return {ADMIN_KEY_HEADER: get_settings().ingestion_admin_key}


async def _ingest_document(file_path: str, test_type: str) -> dict[str, Any]:
    settings = get_settings()
    content = Path(file_path).read_bytes()
    filename = Path(file_path).name
    with _ingestion_service():
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{settings.ingestion_service_url}/admin/documents",
                files={"file": (filename, content, "application/octet-stream")},
                data={"test_type": test_type},
                headers=_admin_headers(),
            )
            r.raise_for_status()
            body = r.json()
            return {"job_id": str(body["job_id"]), "status": str(body["status"])}


async def _list_jobs(limit: int) -> list[dict[str, Any]]:
    settings = get_settings()
    with _ingestion_service():
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{settings.ingestion_service_url}/admin/documents",
                params={"limit": limit},
                headers=_admin_headers(),
            )
            r.raise_for_status()
            return cast(list[dict[str, Any]], r.json())


async def _get_job(job_id: str) -> dict[str, Any] | None:
    settings = get_settings()
    with _ingestion_service():
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{settings.ingestion_service_url}/admin/documents/{job_id}",
                headers=_admin_headers(),
            )
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return cast(dict[str, Any], r.json())


if __name__ == "__main__":
    run(mcp)
