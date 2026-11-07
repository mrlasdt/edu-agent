"""
Per-turn measurement for the live chat path.

One learner message is one *turn*. A turn can make several model calls (a retry, an escalation,
a guardrail check), spread over two services: the gateway runs the guardrails, the agent runs
the tutor. Each service meters its own calls with a `TurnMeter` and appends ONE JSON line to a
shared log when its part of the turn ends; both lines carry the gateway's `trace_id`, and
`python -m evals.turn_report` joins them back into one record per turn.

What a `TurnMeter` records (see `TurnMeter.snapshot`):
  - model calls, input and output tokens, and the USD estimate (`shared.budget.estimate_cost`)
  - time to first token and total time, plus named phases such as retrieval
  - free-form fields the service adds with `annotate` (notes retrieved, guardrail outcome, ...)

Tokens come from the provider's own usage report whenever it sends one. Streaming asks for it
explicitly (`stream_options={"include_usage": True}`); when a provider does not send it, the
meter falls back to `estimate_tokens` and the record says so (`usage` is "estimate" or "mixed")
instead of passing a guess off as a measurement.

Cost is recorded here, never capped: the spend cap (`shared.budget.Ledger`) still covers only
the IELTS mock and the evals. Measuring must never break a turn, so every write failure is
logged and swallowed.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from shared.src.shared.budget import estimate_cost, estimate_tokens
from shared.src.shared.config import get_settings
from shared.src.shared.llm_router import answered_model, get_router

_log = logging.getLogger(__name__)


class TurnLog:
    """An append-only JSON-lines file; the one place the meters write and the report reads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, separators=(",", ":")) + "\n"
        # One write call per line: appends of a short line do not interleave across processes.
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def read(self) -> list[dict[str, Any]]:
        """Every well-formed record; a torn or foreign line is skipped, not fatal."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records

    def last_with_trace(self, trace_id: str) -> dict[str, Any] | None:
        """The newest well-formed record carrying `trace_id`, or None. Scans line by line."""
        if not trace_id or not self.path.exists():
            return None
        needle = f'"trace_id":{json.dumps(trace_id)}'
        found: dict[str, Any] | None = None
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if needle not in line:
                    continue
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict) and value.get("trace_id") == trace_id:
                    found = value
        return found


def default_turn_log() -> TurnLog | None:
    """The log at `TURN_LOG_PATH`; an empty path switches measurement off."""
    path = get_settings().turn_log_path.strip()
    return TurnLog(path) if path else None


@dataclass
class CallRecord:
    """One model call."""

    model: str
    input_tokens: int
    output_tokens: int
    estimated: bool  # True when the tokens are our estimate, not the provider's report
    cost_usd: float
    ms: float  # request sent to response complete
    ttft_ms: float | None  # request sent to first streamed token; None for a non-streamed call
    error: str | None = None


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _read_usage(obj: Any) -> tuple[int, int] | None:
    """(input, output) tokens from a response or stream chunk, or None when it carries none."""
    usage = getattr(obj, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    else:
        prompt, completion = getattr(usage, "prompt_tokens", None), getattr(
            usage, "completion_tokens", None
        )
    prompt_tokens, completion_tokens = _count(prompt), _count(completion)
    if prompt_tokens is None or completion_tokens is None:
        return None
    return prompt_tokens, completion_tokens


def _estimate(text: str) -> int:
    return estimate_tokens(text) if text else 0


def _prompt_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(m["content"] for m in messages if isinstance(m.get("content"), str))


class TurnMeter:
    """
    Meters one service's share of one turn. Create it when the turn starts, route every model
    call through `stream` or `complete`, and call `finish` when the turn ends (in a `finally`).
    """

    def __init__(
        self,
        trace_id: str,
        source: str,
        agent: str,
        mode: str,
        prompt_version: str = "",
        *,
        log: TurnLog | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.trace_id = trace_id
        self.source = source  # "gateway" or "agent": which service wrote the record
        self.agent = agent
        self.mode = mode
        self.prompt_version = prompt_version
        self._log = log
        self._clock = clock
        self._started = clock()
        self._calls: list[CallRecord] = []
        self._phases: dict[str, float] = {}
        self._extra: dict[str, Any] = {}
        self._record: dict[str, Any] | None = None

    @classmethod
    def detached(cls) -> TurnMeter:
        """A meter that belongs to no turn: it measures but is never written (no trace id)."""
        return cls("", "detached", "", "")

    # ── instrumentation ──────────────────────────────────────────────────────

    def annotate(self, **fields: Any) -> None:
        """Attach turn facts to the record (notes retrieved, guardrail outcome, ...)."""
        self._extra.update(fields)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time a non-model step; repeated phases of the same name add up."""
        start = self._clock()
        try:
            yield
        finally:
            self._phases[name] = self._phases.get(name, 0.0) + (self._clock() - start) * 1000

    async def stream(
        self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[Any]:
        """
        A metered streaming completion through the Router (`shared.llm_router`): retries,
        fallbacks and cooldowns apply, and the record names the model that answered. Yields the
        same content chunks the bare `litellm.acompletion` would, so callers stay unchanged; the
        closing usage-only chunk (no choices) is consumed here. Extra keyword arguments
        (temperature, metadata, ...) pass through.
        """
        start = self._clock()
        first_token: float | None = None
        out: list[str] = []
        usage: tuple[int, int] | None = None
        error: str | None = None
        response: Any = None
        try:
            response = await get_router().acompletion(
                model=model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )
            async for chunk in response:
                usage = _read_usage(chunk) or usage
                if not getattr(chunk, "choices", None):
                    continue
                content = chunk.choices[0].delta.content
                if isinstance(content, str) and content:
                    if first_token is None:
                        first_token = self._clock()
                    out.append(content)
                yield chunk
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            ttft = None if first_token is None else (first_token - start) * 1000
            self._add_call(
                answered_model(response, model), messages, "".join(out), usage, start, ttft, error
            )

    async def complete(
        self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        """A metered non-streaming completion through the Router; returns the response unchanged."""
        start = self._clock()
        response: Any = None
        error: str | None = None
        try:
            response = await get_router().acompletion(
                model=model, messages=messages, stream=False, **kwargs
            )
            return response
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            text = ""
            if response is not None:
                try:
                    content = response.choices[0].message.content
                    text = content if isinstance(content, str) else ""
                except (AttributeError, IndexError, TypeError):
                    text = ""
            self._add_call(
                answered_model(response, model), messages, text, _read_usage(response),
                start, None, error,
            )

    def _add_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        output_text: str,
        usage: tuple[int, int] | None,
        start: float,
        ttft_ms: float | None,
        error: str | None,
    ) -> None:
        no_response = usage is None and error is not None and not output_text
        estimated = usage is None and not no_response
        if no_response:
            input_tokens, output_tokens = 0, 0
        else:
            input_tokens, output_tokens = usage or (
                _estimate(_prompt_text(messages)),
                _estimate(output_text),
            )
        self._calls.append(
            CallRecord(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated=estimated,
                cost_usd=estimate_cost(model, input_tokens, output_tokens),
                ms=(self._clock() - start) * 1000,
                ttft_ms=ttft_ms,
                error=error,
            )
        )

    # ── the record ───────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """The record as of now."""
        calls = self._calls
        estimated = [c.estimated for c in calls]
        usage = "none" if not calls else "estimate" if all(estimated) else (
            "provider" if not any(estimated) else "mixed"
        )
        first_ttft = next((c.ttft_ms for c in calls if c.ttft_ms is not None), None)
        return {
            "ts": time.time(),
            "trace_id": self.trace_id,
            "source": self.source,
            "agent": self.agent,
            "mode": self.mode,
            "prompt_version": self.prompt_version,
            "llm_calls": len(calls),
            "input_tokens": sum(c.input_tokens for c in calls),
            "output_tokens": sum(c.output_tokens for c in calls),
            "usage": usage,
            "cost_usd": sum(c.cost_usd for c in calls),
            "ttft_ms": first_ttft,
            "llm_ms": sum(c.ms for c in calls),
            "total_ms": (self._clock() - self._started) * 1000,
            "phases_ms": {k: round(v, 3) for k, v in self._phases.items()},
            "calls": [asdict(c) for c in calls],
            **self._extra,
        }

    def finish(self, error: str | None = None) -> dict[str, Any]:
        """
        Close the turn and append its record to the log. Safe to call twice (the second call
        returns the first record). A turn with no trace id is not part of the live path
        (a direct call from a script or an eval) and is measured but not written.
        """
        if self._record is not None:
            return self._record
        if error is not None:
            self._extra["error"] = error
        self._record = self.snapshot()
        log = self._log if self._log is not None else default_turn_log()
        if log is not None and self.trace_id:
            try:
                log.append(self._record)
            except Exception:
                _log.warning("turn record not written", exc_info=True)
        return self._record
