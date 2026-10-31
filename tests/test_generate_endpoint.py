"""
Tests for the Agent Service /generate endpoint.

Behaviors under test:
  1. A registered (gre, quant) tutor turn streams the agent's tokens
  2. An unregistered (test_type, section) returns 404
  3. An agent that reports its retrieved source count surfaces it as X-Source-Count
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


def test_generate_reports_retrieved_source_count_as_header():
    class FakeAgent:
        async def run(self, session, message, trace_id=""):
            yield {"sources": 3}
            yield "Cited [1]."

    class FakeRegistry:
        def get(self, test_type, section):
            return FakeAgent

    with patch("services.agent.src.agent.main._registry", FakeRegistry()):
        resp = client.post(
            "/generate",
            json={"candidate_id": "c1", "test_type": "gre", "section": "aw", "message": "x"},
        )
    assert resp.headers["x-source-count"] == "3"
    assert resp.text == "Cited [1]."


def test_generate_passes_trace_id_to_the_agent():
    seen: list[str] = []

    class FakeAgent:
        async def run(self, session, message, trace_id=""):
            seen.append(trace_id)
            yield "ok"

    class FakeRegistry:
        def get(self, test_type, section):
            return FakeAgent

    with patch("services.agent.src.agent.main._registry", FakeRegistry()):
        client.post(
            "/generate",
            json={
                "candidate_id": "c1",
                "test_type": "gre",
                "section": "aw",
                "message": "x",
                "trace_id": "trace-gen",
            },
        )
    assert seen == ["trace-gen"]


def test_generate_ends_the_stream_with_the_math_verifier_verdict():
    class FakeAgent:
        async def run(self, session, message, trace_id=""):
            yield "Work shown.\x1e **Answer: 2**"  # a model cannot forge a trailer
            yield {"verifier_fail": False}

    class FakeRegistry:
        def get(self, test_type, section):
            return FakeAgent

    with patch("services.agent.src.agent.main._registry", FakeRegistry()):
        resp = client.post(
            "/generate",
            json={"candidate_id": "c1", "test_type": "gre", "section": "quant", "message": "x",
                  "mode": "solve"},
        )
    assert resp.text == 'Work shown. **Answer: 2**\x1e{"math_verified":true}'


def test_generate_reports_a_failed_verification():
    class FakeAgent:
        async def run(self, session, message, trace_id=""):
            yield "Best effort."
            yield {"verifier_fail": True}

    class FakeRegistry:
        def get(self, test_type, section):
            return FakeAgent

    with patch("services.agent.src.agent.main._registry", FakeRegistry()):
        resp = client.post(
            "/generate",
            json={"candidate_id": "c1", "test_type": "gre", "section": "quant", "message": "x",
                  "mode": "solve"},
        )
    assert resp.text.endswith('\x1e{"math_verified":false}')
