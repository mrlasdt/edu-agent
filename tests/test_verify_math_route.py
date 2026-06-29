"""
Tests for the math-verifier's plain-HTTP /tools/verify_math route.

The verify-math Skill calls this REST facade (the MCP SSE transport exposes no
tool route). These guard the wire contract the Skill depends on: a JSON body of
{expression, expected} in, {verified, computed, error} out.
"""

from starlette.testclient import TestClient

from services.math_verifier.src.math_verifier.server import mcp

client = TestClient(mcp.sse_app())


def test_route_reports_verified_true():
    resp = client.post("/tools/verify_math", json={"expression": "2 + 3 * 4", "expected": "14"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["verified"] is True
    assert body["error"] is None


def test_route_reports_verified_false_on_mismatch():
    resp = client.post("/tools/verify_math", json={"expression": "2 + 2", "expected": "5"})
    assert resp.status_code == 200
    assert resp.json()["verified"] is False
