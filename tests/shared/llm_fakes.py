"""
Offline stand-ins for model providers, built on litellm's own `mock_response`.

`mock_router` returns a real LiteLLM Router (`shared.llm_router.build_router`) whose deployments
each answer, or fail, as scripted. Nothing patches `litellm.acompletion`: the Router's retries,
fallbacks and cooldowns run for real, only the provider call is faked.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import litellm
from litellm.router import Router

from shared.src.shared.llm_router import RouterPolicy, build_router

RATE_LIMITED = "litellm.RateLimitError"  # litellm raises a 429 for this mock_response
SERVER_ERROR = "litellm.InternalServerError"  # ... and a 500 for this one

# Retries are instant in tests; everything else is the product policy.
FAST = RouterPolicy(retry_after=0)


class Unauthorized(litellm.AuthenticationError):
    """A 401. Router settings are deep-copied, so this exception copies itself as itself."""

    def __init__(self) -> None:
        super().__init__("bad key", "openai", "mock")

    def __deepcopy__(self, memo: dict[int, Any]) -> Unauthorized:
        return self


class BadRequest(litellm.BadRequestError):
    """A 400."""

    def __init__(self) -> None:
        super().__init__("bad request", "mock", "openai")

    def __deepcopy__(self, memo: dict[int, Any]) -> BadRequest:
        return self


def mock_router(
    outcomes: Mapping[str, Any],
    chains: Mapping[str, Sequence[str]] | None = None,
    policy: RouterPolicy = FAST,
) -> Router:
    """
    A Router with one deployment per model in `outcomes`: a string is the model's reply, one of
    `RATE_LIMITED` / `SERVER_ERROR` or an exception instance (`Unauthorized()`) is its failure.
    `chains` maps a model to the fallbacks the Router tries after it, in order.
    """
    return build_router(
        None,
        policy=policy,
        models=list(outcomes),
        fallbacks=chains,
        deployment_params={m: {"mock_response": o, "api_key": "k"} for m, o in outcomes.items()},
    )


class ScriptedRouter:
    """
    A stand-in for the Router for tests that need a different reply on every call (a sequence,
    a counter, a captured request). `behave(**kwargs)` answers each `acompletion`. The Router's
    own retry, fallback and cooldown behaviour is tested against `mock_router`, not through this.
    """

    def __init__(self, behave: Any) -> None:
        self.behave = behave
        self.fallbacks: list[dict[str, list[str]]] = []

    async def acompletion(self, **kwargs: Any) -> Any:
        return await self.behave(**kwargs)


def calls(router: Router) -> dict[str, int]:
    """How many provider calls each model got, retries included (litellm's own counter)."""
    return dict(router.total_calls)
