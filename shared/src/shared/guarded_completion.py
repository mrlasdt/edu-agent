"""
The one way a metered model call is made: check the budget, call the Router, record the spend.

Two call sites meter spend against the shared ledger (`shared.budget`): the IELTS mock's
`LiteLLMCompleter` and the eval harness's `MeteredTransport`. Both go through
`guarded_acompletion`, so the rule is written once:

  1. before the first attempt, project the worst case over the whole fallback chain (the whole
     prompt plus `max_tokens` of output at the dearest model the Router could fall back to) and
     refuse the call with `BudgetExceededError` if that could take spend over the cap;
  2. call the Router, which owns the retries, the fallback order and the cooldowns;
  3. record the cost under the model that actually answered, from the provider's token report
     (or an estimate when the provider sent none).

Cache, offline replay, per-call tags and the rest of what the eval transport adds stay with the
transport; this function only does the part both call sites share.
"""
from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from shared.src.shared.budget import Ledger, estimate_cost, estimate_tokens
from shared.src.shared.llm_router import Completions, answered_model, configured_chain

_SECRET = re.compile(r"(sk-[\w-]{8,}|Bearer\s+\S+)", re.I)


def failure_reason(exc: BaseException, limit: int = 200) -> str:
    """One-line, redacted, bounded description of a provider failure."""
    text = _SECRET.sub("[redacted]", " ".join(str(exc).split()))
    return f"{type(exc).__name__}: {text[:limit]}"


@dataclass
class InFlight:
    """
    Worst-case cost of the calls a caller has started and not finished. A caller that runs
    several calls at once shares one of these between them, so the second call's budget check
    counts what the first has reserved instead of seeing only the ledger.
    """

    usd: float = 0.0


@dataclass(frozen=True)
class GuardedCall:
    """A completed call: the Router's response and what it cost."""

    response: Any
    answered: str  # the model that produced the reply; after a fallback, not the one asked for
    chain: list[str]  # the models the Router was allowed to try, in order
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float

    @property
    def message(self) -> Any:
        return self.response.choices[0].message

    @property
    def text(self) -> str:
        return str(self.message.content or "")


def prompt_text(messages: Sequence[dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


async def guarded_acompletion(
    router: Completions,
    model: str,
    messages: list[dict[str, Any]],
    ledger: Ledger,
    run_id: str,
    *,
    max_tokens: int,
    fallbacks: Sequence[str] | None = None,
    inflight: InFlight | None = None,
    **kwargs: Any,
) -> GuardedCall:
    """
    Budget-check, call `router.acompletion` (never streamed) and record the spend under `run_id`.

    `fallbacks` replaces the Router's configured fallbacks for this call (a caller resuming a
    chain after an empty reply passes the models still ahead); the budget check follows it.
    `inflight` is the caller's shared reservation (see `InFlight`).

    `BudgetExceededError` is raised before the Router is called, so no model is tried and nothing
    is recorded. A provider failure is raised as it came, after the Router has tried the whole
    chain; nothing is recorded for it.
    """
    kwargs.pop("stream", None)
    chain = [model, *fallbacks] if fallbacks is not None else configured_chain(router, model)
    prompt_tokens = estimate_tokens(prompt_text(messages))
    projected = max(estimate_cost(m, prompt_tokens, max_tokens) for m in chain)
    ledger.ensure_capacity((inflight.usd if inflight else 0.0) + projected)
    route = {"fallbacks": [{model: list(fallbacks)}]} if fallbacks is not None else {}
    if inflight:
        inflight.usd += projected
    start = time.perf_counter()
    try:
        response = await router.acompletion(
            model=model, messages=messages, max_tokens=max_tokens, stream=False,
            **route, **kwargs,
        )
    finally:
        if inflight:
            inflight.usd -= projected
    latency = time.perf_counter() - start
    answered = answered_model(response, model)
    text = str(response.choices[0].message.content or "")
    usage = getattr(response, "usage", None)
    p_tok = int(getattr(usage, "prompt_tokens", 0) or prompt_tokens)
    c_tok = int(getattr(usage, "completion_tokens", 0) or estimate_tokens(text))
    cost = estimate_cost(answered, p_tok, c_tok)
    ledger.record(run_id, answered, cost)
    return GuardedCall(response, answered, chain, p_tok, c_tok, cost, latency)
