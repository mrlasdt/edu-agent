"""
Model access for the IELTS mock, behind one seam.

`Completer` is what the grader and the feedback generator call. The production completer calls
the role's model from the model config through the LiteLLM Router (`shared.llm_router`), which
owns the retries, the fallback order and the cooldowns; the eval harness passes its own that calls
the same `get_router()` so the metered transport (budget ledger, cache) sits underneath unchanged.

The production completer meters itself in the same budget ledger: a call that could take total
spend over the cap is refused before it is made (`BudgetExceededError`), and the cost of every
paid call is recorded after it.
"""
from __future__ import annotations

import re
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, Protocol

from shared.src.shared.budget import Ledger, estimate_cost, estimate_tokens, spend_ledger
from shared.src.shared.config import get_settings, load_model_config
from shared.src.shared.llm_router import (
    Completions,
    answered_model,
    configured_chain,
    get_router,
)
from shared.src.shared.observability import build_llm_metadata

PROMPT_ROOT = Path("prompts")
GRADER_ROLE = "ielts_grader"
FEEDBACK_ROLE = "ielts_feedback"
_SECRET = re.compile(r"(sk-[\w-]{8,}|Bearer\s+\S+)", re.I)


class ModelUnavailableError(RuntimeError):
    """The model failed or returned nothing for a call."""


class Completer(Protocol):
    def __call__(
        self, role: str, messages: list[dict[str, str]], *, temperature: float, max_tokens: int,
        trace_id: str = "", prompt_version: str = "",
    ) -> Awaitable[str]: ...


def load_prompt(agent: str, mode: str, version: str) -> str:
    raw = (PROMPT_ROOT / agent / mode / f"{version}.md").read_text()
    return re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL).strip()


def prompt_version(role: str) -> str:
    return load_model_config(get_settings().model_config_path).prompt_version(role, "writing_t2")


def role_temperature(role: str) -> float:
    return load_model_config(get_settings().model_config_path).role(role).temperature


def role_model(role: str) -> str:
    return load_model_config(get_settings().model_config_path).litellm_model(role)


class LiteLLMCompleter:
    """
    Call the role's model through the Router: a provider failure moves on to the role's
    `fallbacks` in order (after the Router's own retry of a transient error such as Google's
    HTTP 500/503), and only when every one has failed is it `ModelUnavailableError`. An empty
    reply is not an error to the Router, so it is handled here: the call continues on the models
    after the one that answered.

    Spend is checked against the cap before every call to the Router, at the dearest model of the
    chain still ahead (`configured_chain`, the one the Router will actually run; worst case: the
    whole prompt plus `max_tokens` of output, plus whatever concurrent calls have reserved), so no
    fallback can take the call past the cap; the reported usage is recorded after each call.
    `BudgetExceededError` is not wrapped: it is not a model outage.
    """

    def __init__(self, ledger: Ledger | None = None, router: Completions | None = None) -> None:
        self._ledger = ledger
        self._router = router
        self._reserved = 0.0  # worst-case cost of calls in flight, so concurrent samples add up

    @property
    def ledger(self) -> Ledger:
        return self._ledger or spend_ledger()

    @property
    def router(self) -> Completions:
        return self._router or get_router()

    async def __call__(
        self, role: str, messages: list[dict[str, str]], *, temperature: float, max_tokens: int,
        trace_id: str = "", prompt_version: str = "",
    ) -> str:
        chain = configured_chain(self.router, role_model(role))
        prompt_tokens = estimate_tokens("\n".join(m["content"] for m in messages))
        return await self._complete(
            chain, messages, temperature=temperature, max_tokens=max_tokens,
            trace_id=trace_id, prompt_version=prompt_version, prompt_tokens=prompt_tokens,
        )

    async def _complete(
        self, chain: list[str], messages: list[dict[str, str]], *, temperature: float,
        max_tokens: int, trace_id: str, prompt_version: str, prompt_tokens: int,
    ) -> str:
        extra: dict[str, Any] = (
            build_llm_metadata(trace_id, prompt_version or "v1") if trace_id else {}
        )
        errors: list[str] = []
        start = 0
        while start < len(chain):
            model = chain[start]
            projected = max(estimate_cost(m, prompt_tokens, max_tokens) for m in chain[start:])
            self.ledger.ensure_capacity(self._reserved + projected)  # raises BudgetExceededError
            # The Router already holds the configured fallbacks for the first model; after an
            # empty reply the rest of the chain is passed for the model we resume from.
            resume = {"fallbacks": [{model: chain[start + 1:]}]} if start else {}
            self._reserved += projected
            try:
                response = await self.router.acompletion(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, stream=False, **resume, **extra,
                )
                text = str(response.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001 - the Router has already tried the whole chain
                reason = _SECRET.sub("[redacted]", " ".join(str(exc).split()))[:120]
                errors.append(f"{' > '.join(chain[start:])}: {type(exc).__name__}: {reason}")
                break
            finally:
                self._reserved -= projected
            answered = answered_model(response, model)
            usage = getattr(response, "usage", None)
            self.ledger.record(
                f"ielts-mock-{trace_id or 'untraced'}", answered,
                estimate_cost(
                    answered,
                    int(getattr(usage, "prompt_tokens", 0) or prompt_tokens),
                    int(getattr(usage, "completion_tokens", 0) or estimate_tokens(text)),
                ),
            )
            if text.strip():
                return text
            errors.append(f"{answered}: empty reply")
            start = chain.index(answered) + 1 if answered in chain else start + 1
        raise ModelUnavailableError("; ".join(errors))
