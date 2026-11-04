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
import re
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import litellm

from evals.agent_eval.budget import Ledger, estimate_cost, estimate_tokens
from evals.agent_eval.cache import ResponseCache, cache_key

# litellm prints multi-line "Give Feedback / Provider List" banners to stdout on errors,
# which corrupts the JSON summary the CLI prints.
litellm.suppress_debug_info = True

FakeModel = Callable[[str, list[dict[str, Any]]], str]

current_tag: contextvars.ContextVar[str] = contextvars.ContextVar("eval_tag", default="agent")

# The response cache is keyed by the request, so k samples of one prompt would all be the same
# cached reply. A sample number set here (see `cache_slot`) is part of the key, which makes
# each of the k samples its own cache entry while a rerun of the same plan still re-pays nothing.
current_slot: contextvars.ContextVar[int] = contextvars.ContextVar("eval_cache_slot", default=0)

DEFAULT_MAX_TOKENS = 800
_RATE_LIMIT_RETRIES = 3
_DROPPED_KWARGS = ("stream_options", "max_tokens", "cache_slot")
_RATE_LIMIT_BACKOFF_S = 15.0
_REASON_MAX_CHARS = 200
# A provider quota that resets on a calendar boundary (e.g. OpenRouter's
# "free-models-per-day") will not clear within a retry window.
_DAILY_QUOTA = re.compile(r"free-models-per-day|per[- ]day|daily (?:rate )?(?:limit|quota)", re.I)
_SECRET = re.compile(r"(sk-[\w-]{8,}|Bearer\s+\S+)", re.I)


def is_daily_quota(exc: BaseException) -> bool:
    return _DAILY_QUOTA.search(str(exc)) is not None


def failure_reason(exc: BaseException) -> str:
    """One-line, redacted, bounded description of a provider failure."""
    text = _SECRET.sub("[redacted]", " ".join(str(exc).split()))
    return f"{type(exc).__name__}: {text[:_REASON_MAX_CHARS]}"


@dataclass
class CallRecord:
    tag: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float


@dataclass
class FailureRecord:
    model: str
    reason: str
    kind: str  # "daily_quota" | "rate_limit" | "error"


class AllModelsFailedError(RuntimeError):
    pass


def _chunk(content: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


def _usage_chunk(prompt_tokens: int, completion_tokens: int) -> Any:
    """The closing usage-only chunk a provider sends when `include_usage` is requested."""
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class _ReplayStream:
    def __init__(self, text: str, usage: tuple[int, int] | None = None) -> None:
        self._text = text
        self._usage = usage

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[Any]:
        yield _chunk(self._text)
        if self._usage is not None:
            yield _usage_chunk(*self._usage)


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
        cache: ResponseCache | None = None,
    ) -> None:
        self.ledger = ledger
        self.run_id = run_id
        self.fallbacks = fallbacks or []
        self.fake_model = fake_model
        self.max_tokens = max_tokens
        self.cache = cache
        self.records: list[CallRecord] = []
        self.failures: list[FailureRecord] = []
        self.exhausted: set[str] = set()  # models whose daily quota tripped this run
        self.spent_this_run = 0.0
        self._real = litellm.acompletion
        self.sleep: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep

    async def acompletion(self, *, model: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
        stream = bool(kw.pop("stream", False))
        # Replay the provider's token report only to a caller that asked for it (the product's
        # TurnMeter does); other callers iterate chunks and expect every chunk to have choices.
        wants_usage = bool((kw.get("stream_options") or {}).get("include_usage"))
        if self.fake_model is not None:
            return self._wrap(self._fake(model, messages), stream)
        errors: list[str] = []
        chain = [model, *[m for m in self.fallbacks if m != model]]
        for candidate in chain:
            if candidate in self.exhausted:
                errors.append(f"{candidate}: skipped (daily quota exhausted earlier in this run)")
                continue
            max_tokens = kw.get("max_tokens", self.max_tokens)
            slot = kw.get("cache_slot", current_slot.get())
            key = cache_key(
                model=candidate,
                messages=messages,
                temperature=kw.get("temperature"),
                max_tokens=max_tokens,
                **({"slot": slot} if slot else {}),
            )
            hit = self.cache.get(key) if self.cache else None
            if hit is not None:
                cached = hit.get("usage")
                usage = (int(cached[0]), int(cached[1])) if cached and wants_usage else None
                return self._wrap(str(hit["text"]), stream, usage)
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
                        max_tokens=max_tokens,
                        num_retries=0,
                        timeout=120,
                        **{k: v for k, v in kw.items() if k not in _DROPPED_KWARGS},
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - any provider failure triggers fallback
                    rate_limited = "RateLimit" in type(exc).__name__
                    if rate_limited and is_daily_quota(exc):
                        self.exhausted.add(candidate)
                        self._fail(errors, candidate, exc, "daily_quota")
                        break
                    if rate_limited and attempt < _RATE_LIMIT_RETRIES:
                        await self.sleep(_RATE_LIMIT_BACKOFF_S * (attempt + 1))
                        continue
                    self._fail(errors, candidate, exc, "rate_limit" if rate_limited else "error")
                    break
            if response is None:
                continue
            latency = time.perf_counter() - start
            text = str(response.choices[0].message.content or "")
            usage = getattr(response, "usage", None)
            p_tok = int(getattr(usage, "prompt_tokens", 0) or prompt_tokens)
            c_tok = int(getattr(usage, "completion_tokens", 0) or estimate_tokens(text))
            self._record(candidate, p_tok, c_tok, latency)
            if self.cache and text:  # an empty reply (a reasoning model out of tokens) is no answer
                self.cache.put(key, {"text": text, "usage": [p_tok, c_tok]})
            return self._wrap(text, stream, (p_tok, c_tok) if wants_usage else None)
        raise AllModelsFailedError("; ".join(errors))

    def _fail(self, errors: list[str], model: str, exc: BaseException, kind: str) -> None:
        reason = failure_reason(exc)
        errors.append(f"{model}: {reason}")
        self.failures.append(FailureRecord(model, reason, kind))

    def failure_summary(self) -> dict[str, Any]:
        """Aggregate failed model calls by model and kind, with one example reason each."""
        out: dict[str, Any] = {}
        for f in self.failures:
            entry = out.setdefault(f"{f.model} [{f.kind}]", {"count": 0, "example": f.reason})
            entry["count"] += 1
        return out

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
        if cost <= 0:
            cost = estimate_cost(model, p_tok, c_tok)
        self.spent_this_run += cost
        self.ledger.record(self.run_id, model, cost)
        self.records.append(CallRecord(tag, model, p_tok, c_tok, cost, latency))

    @staticmethod
    def _wrap(text: str, stream: bool, usage: tuple[int, int] | None = None) -> Any:
        return _ReplayStream(text, usage) if stream else _response(text)

    @contextlib.contextmanager
    def installed(self) -> Iterator[None]:
        original = litellm.acompletion
        litellm.acompletion = self.acompletion
        try:
            yield
        finally:
            litellm.acompletion = original


@contextlib.contextmanager
def cache_slot(slot: int) -> Iterator[None]:
    """Calls made inside use cache slot `slot` (the sample number) in their cache key."""
    token = current_slot.set(slot)
    try:
        yield
    finally:
        current_slot.reset(token)


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
