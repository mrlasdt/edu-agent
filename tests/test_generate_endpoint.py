"""
Tests for the Agent Service /generate endpoint.

Behaviors under test:
  1. A registered (gre, quant) tutor turn streams the agent's tokens
  2. An unregistered (test_type, section) returns 404
"""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from services.agent.src.agent.main import app

client = TestClient(app)


def _mock_stream(tokens: list[str]):
    async def _stream():
        for token in tokens:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = token
            yield chunk

    return _stream()


def test_generate_streams_quant_tutor_tokens():
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _mock_stream(["Let's ", "think ", "step by step."])
        resp = client.post(
            "/generate",
            json={
                "candidate_id": "c1",
                "test_type": "gre",
                "section": "quant",
                "message": "How do I start x^2 - 5x + 6 = 0?",
                "mode": "tutor",
            },
        )
    assert resp.status_code == 200
    assert "step by step." in resp.text


def test_generate_unregistered_section_returns_404():
    resp = client.post(
        "/generate",
        json={
            "candidate_id": "c1",
            "test_type": "gre",
            "section": "verbal",
            "message": "anything",
        },
    )
    assert resp.status_code == 404
