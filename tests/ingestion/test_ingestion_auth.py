"""
Tests for the ingestion service's admin-key gate (slice E1, plan chapter 18).

Behaviors under test:
  1. Without the X-Admin-Key header, or with a wrong one, /admin routes answer 401 and write
     nothing; /healthz stays open
  2. With no INGESTION_ADMIN_KEY configured, every /admin request is refused (503): the gate
     fails closed
  3. With the right key an upload works, and the server picks the tier: a `tier` form field
     (school, candidate, anything) is ignored and the chunks land in the global tier with no
     owner ids
  4. The job-status route is gated too
"""

import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from services.ingestion.src.ingestion.main import app
from services.ingestion.src.ingestion.settings import get_settings
from shared.src.shared.config import ADMIN_KEY_HEADER

KEY = "correct-horse-battery-staple"
PIPELINE = "services.ingestion.src.ingestion.pipeline.run_ingestion_pipeline"


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("INGESTION_ADMIN_KEY", KEY)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _upload(client, headers=None, **form):
    return client.post(
        "/admin/documents",
        files={"file": ("note.txt", io.BytesIO(b"A short note about ratios."), "text/plain")},
        data={"test_type": "gre", **form},
        headers=headers or {},
    )


@pytest.mark.parametrize(
    "headers",
    [{}, {ADMIN_KEY_HEADER: "wrong"}, {ADMIN_KEY_HEADER: ""}, {ADMIN_KEY_HEADER: KEY + "x"}],
    ids=["no-header", "wrong-key", "empty-key", "near-miss"],
)
def test_upload_without_the_right_key_is_401_and_writes_nothing(configured, headers):
    with patch(PIPELINE, new_callable=AsyncMock) as pipeline:
        resp = _upload(TestClient(app), headers, tier="global")
    assert resp.status_code == 401
    pipeline.assert_not_called()


def test_job_status_route_needs_the_key_too(configured):
    client = TestClient(app)
    assert client.get("/admin/documents/job-1").status_code == 401
    with patch.dict(
        "services.ingestion.src.ingestion.main._jobs",
        {"job-1": {"job_id": "job-1", "status": "completed"}},
    ):
        resp = client.get("/admin/documents/job-1", headers={ADMIN_KEY_HEADER: KEY})
    assert resp.status_code == 200


def test_healthz_stays_open(configured):
    assert TestClient(app).get("/healthz").status_code == 200


@pytest.mark.parametrize("supplied", [None, "", "anything"])
def test_unset_key_refuses_everything(monkeypatch, supplied):
    monkeypatch.setenv("INGESTION_ADMIN_KEY", "")
    get_settings.cache_clear()
    headers = {} if supplied is None else {ADMIN_KEY_HEADER: supplied}
    try:
        with patch(PIPELINE, new_callable=AsyncMock) as pipeline:
            resp = _upload(TestClient(app), headers)
        assert resp.status_code == 503
        assert "INGESTION_ADMIN_KEY" in resp.json()["detail"]
        pipeline.assert_not_called()
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "form", [{}, {"tier": "global"}, {"tier": "school"}, {"tier": "candidate"}]
)
def test_upload_with_the_key_is_filed_under_the_global_tier_whatever_the_form_says(
    configured, form
):
    captured: list[dict] = []

    async def capture(chunks, metadata):
        captured.extend(chunks)
        return {"chunks_indexed": len(chunks)}

    with patch(PIPELINE, side_effect=capture):
        resp = _upload(TestClient(app), {ADMIN_KEY_HEADER: KEY}, **form)

    assert resp.status_code == 200
    assert captured
    for chunk in captured:
        assert chunk["tier"] == "global"
        assert chunk["school_id"] is None
        assert chunk["candidate_id"] is None
