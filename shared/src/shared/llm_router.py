"""
One resilience policy for every model call: the LiteLLM Router (ADR-0007, "Amendment: resilience
is Router-native").

The Router owns what used to be hand-written loops in three places: the order of fallbacks, the
retry of transient provider errors, the timeout, and the cooldown of a model that keeps failing.
This module only turns the model config into Router settings:

  - every model string in `config/model_config*.yaml` (a role's own model and its `fallbacks`)
    becomes one deployment, named by the same `provider/model` string callers already use;
  - a role's `fallbacks:` list becomes that model's Router fallback chain, in order;
  - a catch-all `*` deployment passes any other model string straight through (an eval pinned to
    one model, a test), with the Router's retry and timeout but no fallback or cooldown.

Cooldown only exists where there is somewhere else to go: a model with a fallback chain is cooled
down, a model without one (a lone /chat model, the `*` pass-through) is always tried, so a burst
of transient errors can never turn into 30 seconds of "no deployments available".

Callers ask `get_router()` for the Router and call `acompletion` on it exactly as they called
`litellm.acompletion`. The eval harness swaps in its metered transport with `use_router`, which
wraps a Router of its own (`evals.agent_eval.transport`).

Policy, in plain words (`RouterPolicy`): a transient error (timeout, 429, 500, 503) is retried on
the same model a couple of times, then the call moves to the next fallback; any other error
(a 400, a 401 from a missing key) is not retried but still moves to the next fallback, so a
missing Google key hands the call to OpenRouter at once. A model that fails more than
`allowed_fails` times inside `cooldown_time` seconds is skipped until the cooldown ends, if it has
fallbacks to skip to.
Streaming falls back too, as long as the failure comes before the first token (litellm also
resumes a stream that breaks midway, on the fallback).

The spend cap is not the Router's job: callers check it before the call, against the dearest model
in the chain, so a fallback can never take a call past the cap (`shared.guarded_completion`, used
by `LiteLLMCompleter` and `evals.agent_eval.transport.MeteredTransport`).
"""
from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import litellm
from litellm.router import Router
from litellm.types.router import RetryPolicy

from shared.src.shared.config import ModelConfig, get_settings, load_model_config

# litellm prints multi-line "Give Feedback / Provider List" banners to stdout on errors, which
# corrupts the JSON summary the eval CLI prints.
litellm.suppress_debug_info = True

_ID_PREFIX = "edu:"
_WILDCARD = "*"


@dataclass(frozen=True)
class RouterPolicy:
    """The Router's reliability settings, one place for every call site."""

    num_retries: int = 2  # retries of a transient error on the same model, before the next one
    retry_after: int = 0  # least seconds to wait before a retry (litellm backs off from 0.5s)
    timeout: float = 120.0  # seconds for one attempt; free reasoning models are slow
    # A model failing more than this many attempts inside `cooldown_time` seconds is skipped.
    # litellm ignores a Router-level value equal to its own default (3), hence 5.
    allowed_fails: int = 5
    cooldown_time: float = 30.0

    def cooldown_policy(self) -> dict[str, int]:
        """
        Per-deployment cooldown for a model that has fallbacks: the same threshold for every
        provider error that counts. A 400 is left out, as litellm itself never cools down on one.
        """
        kinds = ("Authentication", "Timeout", "RateLimit", "InternalServer", "ServiceUnavailable",
                 "BadGateway", "NotFound")
        return {f"{kind}ErrorAllowedFails": self.allowed_fails for kind in kinds}

    def retry_policy(self) -> RetryPolicy:
        """Retry only transient errors: timeout, 429, 500 and 503. A 400 or 401 never repeats."""
        return RetryPolicy(
            TimeoutErrorRetries=self.num_retries,
            RateLimitErrorRetries=self.num_retries,
            InternalServerErrorRetries=self.num_retries,
            ServiceUnavailableErrorRetries=self.num_retries,
            BadRequestErrorRetries=0,
            AuthenticationErrorRetries=0,
            ContentPolicyViolationErrorRetries=0,
            DefaultRetries=0,
        )


# The product: the live /chat path and the IELTS mock.
LIVE_POLICY = RouterPolicy()
# The evals run against free-tier models whose per-minute quotas clear in about a minute, so they
# wait longer between retries (the hand-rolled loop slept 15s, 30s, 45s).
EVAL_POLICY = RouterPolicy(num_retries=3, retry_after=15)


class Completions(Protocol):
    """What a call site needs: `await router.acompletion(model=..., messages=..., ...)`."""

    def acompletion(self, *args: Any, **kwargs: Any) -> Awaitable[Any]: ...


def fallback_chains(config: ModelConfig | None) -> dict[str, list[str]]:
    """
    Each role's own model mapped to its `fallbacks`, in order. Two roles on one model share a
    chain: their lists are merged, first appearance first, never repeating a model or itself.
    """
    chains: dict[str, list[str]] = {}
    if config is None:
        return chains
    for role in config.roles:
        primary = config.litellm_model(role)
        merged = chains.setdefault(primary, [])
        for backup in config.litellm_fallbacks(role):
            if backup != primary and backup not in merged:
                merged.append(backup)
    return {primary: backups for primary, backups in chains.items() if backups}


def _cooldown_info(has_fallbacks: bool, policy: RouterPolicy) -> dict[str, Any]:
    """
    litellm exempts a deployment with its own `allowed_fails` and no policy from cooldown while it
    is the only one for its model name, and a deployment with an `allowed_fails_policy` is cooled
    by it. Router-level `allowed_fails` alone would cool every model, lone ones included.
    """
    if has_fallbacks:
        return {"allowed_fails_policy": policy.cooldown_policy()}
    return {"allowed_fails": policy.allowed_fails}


def build_router(
    config: ModelConfig | None = None,
    *,
    policy: RouterPolicy = LIVE_POLICY,
    models: Iterable[str] = (),
    fallbacks: Mapping[str, Sequence[str]] | None = None,
    deployment_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> Router:
    """
    A Router for the model config: one deployment per model string it names, plus `models`, and
    the fallback chains its roles declare, plus `fallbacks` (model -> backups, in order).
    `deployment_params` adds litellm parameters to one model's deployment; tests use it to give
    each model a `mock_response`, which makes litellm answer (or fail) offline.
    """
    chains = fallback_chains(config)
    for primary, backups in (fallbacks or {}).items():
        merged = chains.setdefault(primary, [])
        merged.extend(b for b in backups if b != primary and b not in merged)
    role_models = [config.litellm_model(role) for role in config.roles] if config else []
    ordered = list(dict.fromkeys([
        *role_models, *models, *chains, *(b for backups in chains.values() for b in backups)
    ]))
    gemini_key = get_settings().gemini_api_key
    model_list: list[dict[str, Any]] = []
    for name in ordered:
        params: dict[str, Any] = {"model": name}
        if name.startswith("gemini/") and gemini_key:
            params["api_key"] = gemini_key  # .env is read by Settings, not exported to litellm
        params.update((deployment_params or {}).get(name, {}))
        model_list.append({
            "model_name": name, "litellm_params": params,
            "model_info": {"id": _ID_PREFIX + name, **_cooldown_info(name in chains, policy)},
        })
    model_list.append({
        "model_name": _WILDCARD, "litellm_params": {"model": _WILDCARD},
        "model_info": _cooldown_info(False, policy),
    })
    return Router(
        model_list=model_list,
        fallbacks=[{primary: backups} for primary, backups in chains.items()],
        num_retries=policy.num_retries,
        retry_policy=policy.retry_policy(),
        retry_after=policy.retry_after,
        timeout=policy.timeout,
        stream_timeout=policy.timeout,
        allowed_fails=policy.allowed_fails,
        cooldown_time=policy.cooldown_time,
    )


def configured_chain(router: Completions, model: str) -> list[str]:
    """
    `model` followed by the fallbacks `router` is configured to try after it, in order.

    A router that wraps others (the eval transport) answers for itself through `chain_for(model)`;
    a Router lists them in `fallbacks` as `{model: [backups]}` entries.
    """
    chain_for = getattr(router, "chain_for", None)
    if callable(chain_for):
        return list(chain_for(model))
    for entry in getattr(router, "fallbacks", None) or []:
        if isinstance(entry, dict) and model in entry:
            return [model, *entry[model]]
    return [model]


def answered_model(response: Any, requested: str) -> str:
    """
    The model string that actually produced `response`: after a fallback it is not the one
    asked for. Falls back to `requested` for a model that has no deployment of its own.
    """
    hidden = getattr(response, "_hidden_params", None)
    model_id = hidden.get("model_id") if isinstance(hidden, dict) else None
    if isinstance(model_id, str) and model_id.startswith(_ID_PREFIX):
        return model_id[len(_ID_PREFIX):]
    return requested


# ── the process Router ───────────────────────────────────────────────────────

_default: tuple[tuple[str, str], Router] | None = None
_installed: Completions | None = None


def get_router() -> Completions:
    """
    The Router every model call goes through: the installed one (see `use_router`), else the
    process default, built from the model config on first use and rebuilt if the config path or
    the Google key changes.
    """
    global _default
    if _installed is not None:
        return _installed
    settings = get_settings()
    key = (settings.model_config_path, settings.gemini_api_key)
    if _default is None or _default[0] != key:
        _default = (key, build_router(load_model_config(settings.model_config_path)))
    return _default[1]


def reset_router() -> None:
    """Forget the process default (tests do this between cases, each on its own event loop)."""
    global _default
    _default = None


@contextlib.contextmanager
def use_router(router: Completions) -> Iterator[None]:
    """Route every `get_router()` caller through `router` while the block runs."""
    global _installed
    previous, _installed = _installed, router
    try:
        yield
    finally:
        _installed = previous

