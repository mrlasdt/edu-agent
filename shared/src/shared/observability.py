"""
Langfuse LLM observability wiring.

setup_observability() — call once at service startup.
  When LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are present in the environment,
  LiteLLM's langfuse success/failure callbacks are enabled. All subsequent
  litellm.acompletion() calls are automatically traced.
  No-ops silently when keys are absent (dev / test with no Langfuse instance).

build_llm_metadata(trace_id, prompt_version, ...) — call per LLM invocation.
  Returns a metadata dict to pass as `metadata=` to litellm.acompletion().
  LiteLLM forwards this to the Langfuse callback, which attaches the fields
  to the trace. This is how trace IDs, prompt versions, and experiment names
  flow from the gateway session into the Langfuse UI.
"""
from __future__ import annotations

import os
from typing import Any


def setup_observability() -> None:
    """
    Configure LiteLLM → Langfuse callback if keys are present.
    Safe to call multiple times; idempotent.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")

    if not (public_key and secret_key):
        return  # no-op in dev/test without Langfuse

    import litellm

    if "langfuse" not in litellm.success_callback:
        litellm.success_callback.append("langfuse")
    if "langfuse" not in litellm.failure_callback:
        litellm.failure_callback.append("langfuse")


def build_llm_metadata(
    trace_id: str,
    prompt_version: str,
    session_id: str | None = None,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """
    Build the metadata dict for a litellm.acompletion() call.

    LiteLLM forwards this to the Langfuse callback, which attaches it to the trace:
      - trace_id: links this LLM call to the enclosing gateway session / turn
      - prompt_version: which prompt file version was used (e.g. "v1", "v3")
      - session_id: Langfuse session grouping (same as trace_id when one turn = one session)
      - experiment_name: when an A/B variant is active

    Langfuse cache hit/miss is reported automatically by the LiteLLM callback
    when LiteLLM detects a cache hit on the request.
    """
    tags: list[str] = [f"prompt:{prompt_version}"]
    if experiment_name:
        tags.append(f"experiment:{experiment_name}")

    meta: dict[str, Any] = {
        "trace_id": trace_id,
        "prompt_version": prompt_version,
        "tags": tags,
    }
    if session_id:
        meta["session_id"] = session_id
    if experiment_name:
        meta["experiment_name"] = experiment_name

    return {"metadata": meta}
