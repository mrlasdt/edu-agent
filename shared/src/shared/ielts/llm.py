"""
Model access for the IELTS mock, behind one seam.

`Completer` is what the grader and the feedback generator call. The production completer calls
the role's model from the model config through the LiteLLM Router (`shared.llm_router`), which
owns the retries, the fallback order and the cooldowns; the eval harness passes its own that calls
the same `get_router()` so the metered transport (budget ledger, cache) sits underneath unchanged.

The production completer meters itself in the same budget ledger through `guarded_acompletion`
(`shared.guarded_completion`, the call the eval transport makes too): a call that could take
total spend over the cap is refused before it is made (`BudgetExceededError`), and the cost of
every paid call is recorded after it.
"""
from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol

from shared.src.shared.budget import BudgetExceededError, Ledger, spend_ledger
from shared.src.shared.config import get_llm_settings, load_model_config
from shared.src.shared.guarded_completion import InFlight, failure_reason, guarded_acompletion
from shared.src.shared.llm_router import Completions, configured_chain, get_router
from shared.src.shared.observability import build_llm_metadata
from shared.src.shared.prompts import read_prompt

GRADER_ROLE = "ielts_grader"
FEEDBACK_ROLE = "ielts_feedback"


class ModelUnavailableError(RuntimeError):
    """The model failed or returned nothing for a call."""


class Completer(Protocol):
    def __call__(
        self, role: str, messages: list[dict[str, str]], *, temperature: float, max_tokens: int,
        trace_id: str = "", prompt_version: str = "",
    ) -> Awaitable[str]: ...


def load_prompt(agent: str, mode: str, version: str) -> str:
    return read_prompt(agent, mode, version)


def prompt_version(role: str) -> str:
    config = load_model_config(get_llm_settings().model_config_path)
    return config.prompt_version(role, "writing_t2")


def role_temperature(role: str) -> float:
    return load_model_config(get_llm_settings().model_config_path).role(role).temperature


def role_model(role: str) -> str:
    return load_model_config(get_llm_settings().model_config_path).litellm_model(role)


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
        # worst-case cost of calls in flight, so concurrent samples add up
        self._inflight = InFlight()

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
        extra: dict[str, Any] = (
            build_llm_metadata(trace_id, prompt_version or "v1") if trace_id else {}
        )
        run_id = f"ielts-mock-{trace_id or 'untraced'}"
        chain = configured_chain(self.router, role_model(role))
        errors: list[str] = []
        start = 0
        while start < len(chain):
            model = chain[start]
            # The Router already holds the configured fallbacks for the first model; after an
            # empty reply the rest of the chain is passed for the model we resume from.
            resume = {"fallbacks": chain[start + 1:]} if start else {}
            try:
                call = await guarded_acompletion(
                    self.router, model, messages, self.ledger, run_id, max_tokens=max_tokens,
                    temperature=temperature, inflight=self._inflight, **resume, **extra,
                )
            except BudgetExceededError:
                raise
            except Exception as exc:  # noqa: BLE001 - the Router has already tried the whole chain
                errors.append(f"{' > '.join(chain[start:])}: {failure_reason(exc, 120)}")
                break
            if call.text.strip():
                return call.text
            errors.append(f"{call.answered}: empty reply")
            start = chain.index(call.answered) + 1 if call.answered in chain else start + 1
        raise ModelUnavailableError("; ".join(errors))
