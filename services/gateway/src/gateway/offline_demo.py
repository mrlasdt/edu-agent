"""
Offline gateway for trying the guardrails from the web chat page without API keys.

Moderation and the agent are replaced by canned replies, so /chat exercises the real
guardrail path (rate limit, PII, answer leak, citations) at no cost. Run with:

    uv run python -m services.gateway.src.gateway.offline_demo

then `cd web && npm run dev` and open http://localhost:5173. Try a message containing
"leak", "email", "cite", or anything else for a plain hint.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch

import uvicorn

from services.gateway.src.gateway import main


async def _no_moderation(text: str, trace_id: str) -> tuple[bool, str]:
    return True, ""


async def _canned_agent(
    payload: dict[str, Any], meta: dict[str, Any] | None = None
) -> AsyncGenerator[bytes, None]:
    message = payload["message"].lower()
    if "cite" in message and meta is not None:
        meta["num_sources"] = 2
    replies = {
        "leak": ["Isolate x: subtract 3 from both sides.\n", "So the answer is x = 4.\n"],
        "email": ["Write to tutor@example.com or call 555-123-4567 for help.\n"],
        "cite": ["Planning matters [1]. Fame alone persuades [9]. Plan well.\n"],
    }
    chunks = next((v for k, v in replies.items() if k in message), ["Try isolating x first.\n"])
    for chunk in chunks:
        yield chunk.encode()


patch.object(main, "check_content_safety", _no_moderation).start()
patch.object(main, "_stream_from_agent", _canned_agent).start()

if __name__ == "__main__":
    uvicorn.run(main.app, host="127.0.0.1", port=8000)
