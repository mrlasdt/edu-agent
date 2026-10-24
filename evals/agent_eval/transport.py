"""
Metered LLM transport.

The product agents call `litellm.acompletion` directly, so the harness runs the
unmodified agent code and swaps in this transport underneath it. It adds, in order:
  1. a budget check before each call (worst-case projection),
  2. a fallback chain across models (free OpenRouter first, OpenAI last),
  3. per-call cost / token / latency records,
  4. an offline mode that serves a fake model and never touches the network.

Real calls are made non-streaming and replayed to the agent as a one-chunk stream,
which keeps retries on a failed model clean. Time-to-first-token is therefore not
measured; per-call latency is full generation time.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import litellm

from evals.agent_eval.budget import Ledger, estimate_cost, estimate_tokens

FakeModel = Callable[[str, list[dict[str, Any]]], str]

current_tag: contextvars.ContextVar[str] = contextvars.ContextVar("eval_tag", default="agent")

DEFAULT_MAX_TOKENS = 800
_RATE_LIMIT_RETRIES = 3
_DROPPED_KWARGS = ("stream_options", "max_tokens")
_RATE_LIMIT_BACKOFF_S = 15.0


@dataclass
class CallRecord:
    tag: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float


class AllModelsFailedError(RuntimeError):
    pass


def _chunk(content: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


class _ReplayStream:
    def __init__(self, text: str) -> None:
        self._text = text

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[Any]:
        yield _chunk(self._text)


def _response(text: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _text_of(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


class MeteredTransport:
    def __init__(
        self,
        ledger: Ledger,
        run_id: str,
        fallbacks: list[str] | None = None,
        fake_model: FakeModel | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.ledger = ledger
        self.run_id = run_id
        self.fallbacks = fallbacks or []
        self.fake_model = fake_model
        self.max_tokens = max_tokens
        self.records: list[CallRecord] = []
        self.spent_this_run = 0.0
        self._real = litellm.acompletion
        self.sleep: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep

    async def acompletion(self, *, model: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
        stream = bool(kw.pop("stream", False))
        if self.fake_model is not None:
            return self._wrap(self._fake(model, messages), stream)
        errors: list[str] = []
        chain = [model, *[m for m in self.fallbacks if m != model]]
        for candidate in chain:
            prompt_tokens = estimate_tokens(_text_of(messages))
            self.ledger.ensure_capacity(
                estimate_cost(candidate, prompt_tokens, self.max_tokens)
            )  # BudgetExceededError propagates: never fall through to a pricier model
            response = None
            for attempt in range(_RATE_LIMIT_RETRIES + 1):
                start = time.perf_counter()
                try:
                    response = await self._real(
                        model=candidate,
                        messages=messages,
                        stream=False,
                        max_tokens=kw.get("max_tokens", self.max_tokens),
                        num_retries=0,
                        timeout=120,
                        **{k: v for k, v in kw.items() if k not in _DROPPED_KWARGS},
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - any provider failure triggers fallback
                    if "RateLimit" in type(exc).__name__ and attempt < _RATE_LIMIT_RETRIES:
                        await self.sleep(_RATE_LIMIT_BACKOFF_S * (attempt + 1))
                        continue
                    errors.append(f"{candidate}: {type(exc).__name__}")
                    break
            if response is None:
                continue
            latency = time.perf_counter() - start
            text = str(response.choices[0].message.content or "")
            usage = getattr(response, "usage", None)
            p_tok = int(getattr(usage, "prompt_tokens", 0) or prompt_tokens)
            c_tok = int(getattr(usage, "completion_tokens", 0) or estimate_tokens(text))
            self._record(candidate, p_tok, c_tok, latency)
            return self._wrap(text, stream)
        raise AllModelsFailedError("; ".join(errors))

    def _fake(self, model: str, messages: list[dict[str, Any]]) -> str:
        assert self.fake_model is not None
        start = time.perf_counter()
        text = self.fake_model(model, messages)
        p_tok, c_tok = estimate_tokens(_text_of(messages)), estimate_tokens(text)
        self.records.append(
            CallRecord(current_tag.get(), model, p_tok, c_tok, 0.0, time.perf_counter() - start)
        )
        return text

    def _record(self, model: str, p_tok: int, c_tok: int, latency: float) -> None:
        cost = estimate_cost(model, p_tok, c_tok)
        self.spent_this_run += cost
        self.ledger.record(self.run_id, model, cost)
        self.records.append(CallRecord(current_tag.get(), model, p_tok, c_tok, cost, latency))

    def record_external(
        self, tag: str, model: str, p_tok: int, c_tok: int, cost: float, latency: float
    ) -> None:
        """Record a call made outside litellm (the jev decisions endpoint)."""
        self.spent_this_run += cost
        self.ledger.record(self.run_id, model, cost)
        self.records.append(CallRecord(tag, model, p_tok, c_tok, cost, latency))

    @staticmethod
    def _wrap(text: str, stream: bool) -> Any:
        return _ReplayStream(text) if stream else _response(text)

    @contextlib.contextmanager
    def installed(self) -> Iterator[None]:
        original = litellm.acompletion
        litellm.acompletion = self.acompletion
        try:
            yield
        finally:
            litellm.acompletion = original


@contextlib.contextmanager
def tagged(tag: str) -> Iterator[None]:
    token = current_tag.set(tag)
    try:
        yield
    finally:
        current_tag.reset(token)


async def gather_limited[T](coros: list[Coroutine[Any, Any, T]], limit: int = 2) -> list[T]:
    sem = asyncio.Semaphore(limit)

    async def run(c: Coroutine[Any, Any, T]) -> T:
        async with sem:
            return await c

    return list(await asyncio.gather(*[run(c) for c in coros]))
