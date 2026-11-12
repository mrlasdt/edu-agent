"""
Metered LLM transport.

The product agents get their Router from `shared.llm_router.get_router()`, so the harness runs the
unmodified agent code and swaps this transport in underneath it (`installed`). It wraps a LiteLLM
Router, which owns the retries, the fallback order (free OpenRouter first, OpenAI last), the
timeout and the cooldown of a model that keeps failing, and adds:
  1. a budget check before each call (worst case over the whole fallback chain),
  2. a response cache,
  3. per-call cost / token / latency records, under the model that actually answered,
  4. an offline mode: a fake model's reply goes through the Router as litellm's `mock_response`,
     so the offline run exercises the same code path and never touches the network.

Real calls are made non-streaming and replayed to the agent as a one-chunk stream.
Time-to-first-token is therefore not measured; per-call latency is full generation time.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from dataclasses import dataclass
from typing import Any

from litellm import ModelResponse, ModelResponseStream
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Delta,
    Function,
    Message,
    StreamingChoices,
    Usage,
)

from evals.agent_eval.cache import ResponseCache, cache_key
from shared.src.shared.budget import BudgetExceededError, Ledger, estimate_tokens
from shared.src.shared.guarded_completion import (
    failure_reason,
    guarded_acompletion,
    prompt_text,
)
from shared.src.shared.llm_router import (
    EVAL_POLICY,
    Completions,
    build_router,
    configured_chain,
    use_router,
)


@dataclass
class FakeToolCalls:
    """What an offline fake model returns instead of text when it "decides" to call tools."""

    calls: list[tuple[str, dict[str, Any]]]


FakeModel = Callable[[str, list[dict[str, Any]]], "str | FakeToolCalls"]

current_tag: contextvars.ContextVar[str] = contextvars.ContextVar("eval_tag", default="agent")

# The response cache is keyed by the request, so k samples of one prompt would all be the same
# cached reply. A sample number set here (see `cache_slot`) is part of the key, which makes
# each of the k samples its own cache entry while a rerun of the same plan still re-pays nothing.
current_slot: contextvars.ContextVar[int] = contextvars.ContextVar("eval_cache_slot", default=0)

DEFAULT_MAX_TOKENS = 800
_MOCK_PROVIDER_MODEL = "openai/offline-fake"  # any provider litellm knows; mock_response answers
_DROPPED_KWARGS = ("stream_options", "max_tokens", "cache_slot")


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
    kind: str  # "rate_limit" | "error" | "fallback" (the call was answered by a later model)


class AllModelsFailedError(RuntimeError):
    pass


def _usage(prompt_tokens: int, completion_tokens: int) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _chunk(content: str) -> ModelResponseStream:
    return ModelResponseStream(choices=[StreamingChoices(delta=Delta(content=content))])


def _usage_chunk(prompt_tokens: int, completion_tokens: int) -> ModelResponseStream:
    """The closing usage-only chunk a provider sends when `include_usage` is requested."""
    return ModelResponseStream(choices=[], usage=_usage(prompt_tokens, completion_tokens))


class _ReplayStream:
    def __init__(self, text: str, usage: tuple[int, int] | None = None) -> None:
        self._text = text
        self._usage = usage

    def __aiter__(self) -> AsyncIterator[ModelResponseStream]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[ModelResponseStream]:
        yield _chunk(self._text)
        if self._usage is not None:
            yield _usage_chunk(*self._usage)


def _response(
    text: str,
    tool_calls: list[dict[str, str]] | None = None,
    usage: tuple[int, int] | None = None,
) -> ModelResponse:
    """A non-streamed reply: text and/or the tool calls a model asked for, as litellm builds it."""
    message = Message(
        content=text or None,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id=c["id"], type="function",
                function=Function(name=c["name"], arguments=c["arguments"]),
            )
            for c in tool_calls
        ]
        if tool_calls
        else None,
    )
    reply = ModelResponse(choices=[Choices(message=message)])
    if usage is not None:
        reply.usage = _usage(*usage)
    return reply


def _tool_calls_of(message: Any) -> list[dict[str, str]]:
    """The tool calls on a provider's message, as plain dicts (cacheable, replayable)."""
    return [
        {
            "id": str(c.id),
            "name": str(c.function.name),
            "arguments": str(c.function.arguments or ""),
        }
        for c in (getattr(message, "tool_calls", None) or [])
    ]


class MeteredTransport:
    """
    `router` is the LiteLLM Router the calls go through. Without one, each model asked for gets a
    Router of its own (`EVAL_POLICY`) whose fallback chain is `[model, *fallbacks]`, built on
    first use. The offline tests pass a Router whose deployments carry a `mock_response`.
    """

    def __init__(
        self,
        ledger: Ledger,
        run_id: str,
        fallbacks: list[str] | None = None,
        fake_model: FakeModel | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        cache: ResponseCache | None = None,
        router: Completions | None = None,
    ) -> None:
        self.ledger = ledger
        self.run_id = run_id
        self.fallback_models = fallbacks or []
        self.fake_model = fake_model
        self.max_tokens = max_tokens
        self.cache = cache
        self.records: list[CallRecord] = []
        self.failures: list[FailureRecord] = []
        self.spent_this_run = 0.0
        self._router = router
        self._routers: dict[str, Completions] = {}

    def chain_for(self, model: str) -> list[str]:
        """`model` and the fallbacks behind it: what `configured_chain` reads from a router."""
        return configured_chain(self.router_for(model), model)

    def router_for(self, model: str) -> Completions:
        """The Router that serves `model`, with its fallbacks."""
        if self._router is not None:
            return self._router
        if model not in self._routers:
            tail = [m for m in self.fallback_models if m != model]
            # Offline, whatever the fake model is called, the deployment behind it is litellm's
            # mock provider: no key, no network.
            offline = {m: {"model": _MOCK_PROVIDER_MODEL} for m in [model, *tail]}
            self._routers[model] = build_router(
                None, policy=EVAL_POLICY, models=[model, *tail], fallbacks={model: tail},
                deployment_params=offline if self.fake_model is not None else None,
            )
        return self._routers[model]

    async def acompletion(self, *, model: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
        stream = bool(kw.pop("stream", False))
        # Replay the provider's token report only to a caller that asked for it (the product's
        # TurnMeter does); other callers iterate chunks and expect every chunk to have choices.
        wants_usage = bool((kw.get("stream_options") or {}).get("include_usage"))
        router = self.router_for(model)
        if self.fake_model is not None:
            return await self._offline(router, model, messages, stream)
        max_tokens = kw.get("max_tokens", self.max_tokens)
        slot = kw.get("cache_slot", current_slot.get())
        key = cache_key(
            model=model,
            messages=messages,
            temperature=kw.get("temperature"),
            max_tokens=max_tokens,
            **({"slot": slot} if slot else {}),
            # a tool-offering request is a different request: its reply can be a tool call
            **({"tools": kw["tools"], "tool_choice": kw.get("tool_choice")}
               if kw.get("tools") else {}),
        )
        hit = self.cache.get(key) if self.cache else None
        if hit is not None:
            cached = hit.get("usage")
            usage = (int(cached[0]), int(cached[1])) if cached and wants_usage else None
            if hit.get("tool_calls") or kw.get("tools"):
                counts = (int(cached[0]), int(cached[1])) if cached else None
                return _response(str(hit["text"]), hit.get("tool_calls"), counts)
            return self._wrap(str(hit["text"]), stream, usage)
        # The budget check over the whole fallback chain, the Router call and the ledger entry
        # are `guarded_acompletion`, the same call the shipped IELTS completer makes.
        try:
            call = await guarded_acompletion(
                router, model, messages, self.ledger, self.run_id, max_tokens=max_tokens,
                **{k: v for k, v in kw.items() if k not in _DROPPED_KWARGS},
            )
        except BudgetExceededError:
            raise
        except Exception as exc:  # noqa: BLE001 - the Router has already tried the whole chain
            kind = "rate_limit" if "RateLimit" in type(exc).__name__ else "error"
            self._fail(model, exc, kind)
            chain = configured_chain(router, model)
            raise AllModelsFailedError(f"{' > '.join(chain)}: {failure_reason(exc)}") from exc
        if call.answered != model:
            self.failures.append(FailureRecord(model, f"answered by {call.answered}", "fallback"))
        text = call.text
        tool_calls = _tool_calls_of(call.message)
        p_tok, c_tok = call.prompt_tokens, call.completion_tokens
        self.spent_this_run += call.cost_usd
        self.records.append(CallRecord(
            current_tag.get(), call.answered, p_tok, c_tok, call.cost_usd, call.latency_s
        ))
        if self.cache and (text or tool_calls):  # an empty reply is no answer
            self.cache.put(key, {"text": text, "usage": [p_tok, c_tok], "tool_calls": tool_calls})
        if tool_calls or kw.get("tools"):
            # the agent's tool loop reads the message and the provider's token report
            return _response(text, tool_calls, (p_tok, c_tok))
        return self._wrap(text, stream, (p_tok, c_tok) if wants_usage else None)

    async def _offline(
        self, router: Completions, model: str, messages: list[dict[str, Any]], stream: bool
    ) -> Any:
        """The fake model's reply, served by the Router as a litellm `mock_response`."""
        assert self.fake_model is not None
        start = time.perf_counter()
        reply = self.fake_model(model, messages)
        mock: dict[str, Any] = (
            {"mock_response": reply}
            if isinstance(reply, str)
            else {
                "mock_response": "",
                "mock_tool_calls": [
                    {"id": f"call_{n}", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args)}}
                    for n, (name, args) in enumerate(reply.calls)
                ],
            }
        )
        response = await router.acompletion(model=model, messages=messages, stream=False, **mock)
        message = response.choices[0].message
        text = str(message.content or "")
        calls = _tool_calls_of(message)
        p_tok = estimate_tokens(prompt_text(messages))
        c_tok = estimate_tokens(text if not calls else json.dumps([c["name"] for c in calls]))
        self.records.append(
            CallRecord(current_tag.get(), model, p_tok, c_tok, 0.0, time.perf_counter() - start)
        )
        if calls:
            return _response("", calls)
        return self._wrap(text, stream)

    def _fail(self, model: str, exc: BaseException, kind: str) -> None:
        self.failures.append(FailureRecord(model, failure_reason(exc), kind))

    def failure_summary(self) -> dict[str, Any]:
        """Aggregate failed model calls by model and kind, with one example reason each."""
        out: dict[str, Any] = {}
        for f in self.failures:
            entry = out.setdefault(f"{f.model} [{f.kind}]", {"count": 0, "example": f.reason})
            entry["count"] += 1
        return out

    @staticmethod
    def _wrap(text: str, stream: bool, usage: tuple[int, int] | None = None) -> Any:
        return _ReplayStream(text, usage) if stream else _response(text)

    @contextlib.contextmanager
    def installed(self) -> Iterator[None]:
        """Route every `get_router()` caller (the product agents, the judges) through this."""
        with use_router(self):
            yield


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
