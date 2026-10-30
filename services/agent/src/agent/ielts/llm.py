"""
Model access for the IELTS mock, behind one seam.

`Completer` is what the grader and the feedback generator call. The production completer calls
the role's model from the model config through litellm; the eval harness passes its own that
calls `litellm.acompletion` directly so the metered transport (budget ledger, fallbacks, cache)
sits underneath unchanged.

The production completer meters itself in the same budget ledger: a call that could take total
spend over the cap is refused before it is made (`BudgetExceededError`), and the cost of every
paid call is recorded after it.
"""
from __future__ import annotations

import re
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, Protocol

import litellm

from shared.src.shared.budget import Ledger, estimate_cost, estimate_tokens, spend_ledger
from shared.src.shared.config import get_settings, load_model_config
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
    Call the role's model; a provider failure or an empty reply is `ModelUnavailableError`.

    Spend is checked against the cap before each call (worst case: the whole prompt plus
    `max_tokens` of output, plus whatever concurrent calls have reserved) and the reported
    usage is recorded after it. `BudgetExceededError` is not wrapped: it is not a model outage.
    """

    def __init__(self, ledger: Ledger | None = None) -> None:
        self._ledger = ledger
        self._reserved = 0.0  # worst-case cost of calls in flight, so concurrent samples add up

    @property
    def ledger(self) -> Ledger:
        return self._ledger or spend_ledger()

    async def __call__(
        self, role: str, messages: list[dict[str, str]], *, temperature: float, max_tokens: int,
        trace_id: str = "", prompt_version: str = "",
    ) -> str:
        model = role_model(role)
        extra: dict[str, Any] = (
            build_llm_metadata(trace_id, prompt_version or "v1") if trace_id else {}
        )
        ledger = self.ledger
        prompt_tokens = estimate_tokens("\n".join(m["content"] for m in messages))
        projected = estimate_cost(model, prompt_tokens, max_tokens)
        ledger.ensure_capacity(self._reserved + projected)  # raises BudgetExceededError
        self._reserved += projected
        try:
            response = await litellm.acompletion(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, stream=False, **extra,
            )
            text = str(response.choices[0].message.content or "")
        except Exception as exc:  # noqa: BLE001 - any provider failure is an unavailable model
            reason = _SECRET.sub("[redacted]", " ".join(str(exc).split()))[:120]
            raise ModelUnavailableError(f"{model}: {type(exc).__name__}: {reason}") from exc
        finally:
            self._reserved -= projected
        usage = getattr(response, "usage", None)
        ledger.record(
            f"ielts-mock-{trace_id or 'untraced'}", model,
            estimate_cost(
                model,
                int(getattr(usage, "prompt_tokens", 0) or prompt_tokens),
                int(getattr(usage, "completion_tokens", 0) or estimate_tokens(text)),
            ),
        )
        if not text.strip():
            raise ModelUnavailableError(f"{model}: empty reply")
        return text
