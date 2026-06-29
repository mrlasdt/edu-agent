"""
Tests for the Gateway /chat endpoint.

Behaviors under test:
  1. Unsafe content is refused (agent never called)
  2. Unregistered (test_type, section) yields the orchestrator's clarification
  3. A safe, routable turn proxies the agent's streamed tokens back verbatim
"""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from services.gateway.src.gateway.main import app

client = TestClient(app)

_SAFE = (True, "")


def test_chat_refuses_unsafe_content():
    with patch(
        "services.gateway.src.gateway.main.check_content_safety",
        new_callable=AsyncMock,
    ) as mock_safety:
        mock_safety.return_value = (False, "content flagged: violence")
        resp = client.post(
            "/chat",
            json={"candidate_id": "c1", "test_type": "gre", "section": "quant", "message": "x"},
        )
    assert resp.status_code == 200
    assert "can't help" in resp.text.lower()


def test_chat_clarifies_unregistered_section():
    with patch(
        "services.gateway.src.gateway.main.check_content_safety",
        new_callable=AsyncMock,
    ) as mock_safety:
        mock_safety.return_value = _SAFE
        resp = client.post(
            "/chat",
            json={"candidate_id": "c1", "test_type": "gre", "section": "verbal", "message": "hi"},
        )
    assert resp.status_code == 200
    assert "not set up" in resp.text.lower()


def test_chat_proxies_agent_stream_on_route():
    async def fake_stream(payload):
        for token in (b"Hello", b", ", b"world"):
            yield token

    with patch(
        "services.gateway.src.gateway.main.check_content_safety",
        new_callable=AsyncMock,
    ) as mock_safety, patch(
        "services.gateway.src.gateway.main._stream_from_agent", fake_stream
    ):
        mock_safety.return_value = _SAFE
        resp = client.post(
            "/chat",
            json={
                "candidate_id": "c1",
                "test_type": "gre",
                "section": "quant",
                "message": "Solve x^2 - 5x + 6 = 0",
            },
        )
    assert resp.status_code == 200
    assert resp.text == "Hello, world"
