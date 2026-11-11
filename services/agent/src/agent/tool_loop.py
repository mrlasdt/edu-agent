"""
A bounded tool loop: let the model call tools, feed it the results, stop.

The shape every tool-using agent shares, written small enough to read in one sitting:

    ask the model (with the tool list)
      -> it answers in text            -> done
      -> it asks for tool calls        -> run them, append the results, ask again

Four exits keep it from running away, and the loop says which one it took (`stop_reason`).
  answered        the model answered in text (the normal exit)
  max_rounds      `MAX_TOOL_ROUNDS` rounds of tool calls were used
  deadline        the wall-clock budget ran out, checked before each new round
  repeated_call   the model asked for a call it already made this turn with the same arguments:
                  the answer cannot change, so running it again only spends a round
When the loop stops for any reason but `answered`, one last call with `tool_choice="none"` and a
note saying so makes the model answer with what it already knows, and say what it could not
check, instead of the turn failing.

What the loop guarantees about the *tools*, whatever the model sends:
  - every tool call gets a result message, or the provider rejects the next request;
  - a call with unreadable arguments, an unknown name, or a tool that raises becomes an
    `{"error": ...}` result the model can read and retry, never an exception that kills the turn;
  - the calls of one round run concurrently, at most `MAX_CALLS_PER_ROUND` of them.

Tool results live only in the message list this function builds. The caller gets back the final
text and a record of the calls; nothing here writes a result into the reply, which is how the
tutor keeps tool results private to the model.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from shared.src.shared.turn_meter import TurnMeter

MAX_TOOL_ROUNDS = 4
MAX_CALLS_PER_ROUND = 4
DEADLINE_S = 60.0
_TRACE_ARGS_CHARS = 200
_TRACE_RESULT_CHARS = 300

ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

OUT_OF_BUDGET_NOTE = (
    "Tool budget used up. Answer now with what you have worked out. Mark any value you "
    "could not check with a tool as unchecked."
)
REPEATED_CALL_NOTE = (
    "You already ran this exact tool call and have its result. Do not repeat it. Answer now "
    "with what you have worked out. Mark any value you could not check with a tool as unchecked."
)

# Why the loop stopped; the values are what the turn record holds.
ANSWERED = "answered"
MAX_ROUNDS = "max_rounds"
DEADLINE = "deadline"
REPEATED_CALL = "repeated_call"


@dataclass
class ToolCallRecord:
    """One tool call, as the turn log and the eval transcript see it."""

    name: str
    arguments: str  # the model's JSON, shortened
    ok: bool
    ms: float
    result: str  # the tool's JSON result, shortened
    round: int

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolLoopResult:
    text: str
    calls: list[ToolCallRecord] = field(default_factory=list)
    rounds: int = 0  # rounds in which the model asked for tools
    stop_reason: str = ANSWERED

    @property
    def out_of_budget(self) -> bool:
        """The loop ended by forcing an answer, not by the model's choice."""
        return self.stop_reason != ANSWERED


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _read_arguments(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """The model's arguments as a dict, or a message telling it how to write them."""
    if isinstance(raw, dict):
        return raw, None
    try:
        parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else None
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed, None
    return None, (
        'the arguments were not a JSON object; send one like {"expression": "2 + 3"} '
        "with the argument names from the tool description."
    )


def _signature(call: Any) -> tuple[str, str]:
    """A call's identity for repeat detection: its name and its arguments, canonicalised."""
    raw = call.function.arguments
    arguments, _ = _read_arguments(raw)
    canonical = json.dumps(arguments, sort_keys=True) if arguments is not None else str(raw)
    return str(call.function.name), canonical


def _assistant_message(message: Any, calls: list[Any]) -> dict[str, Any]:
    """The model's own tool request, replayed to it ahead of the results."""
    return {
        "role": "assistant",
        "content": message.content if isinstance(message.content, str) else "",
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in calls
        ],
    }


async def _run_one(
    call: Any, execute: ToolExecutor, round_no: int, skipped: bool
) -> tuple[dict[str, Any], ToolCallRecord]:
    name = str(call.function.name)
    raw = call.function.arguments
    start = time.perf_counter()
    if skipped:
        result: dict[str, Any] = {
            "error": f"too many tool calls at once (limit {MAX_CALLS_PER_ROUND}); "
            "call this one again in your next step."
        }
    else:
        arguments, problem = _read_arguments(raw)
        if arguments is None:
            result = {"error": problem}
        else:
            try:
                result = await execute(name, arguments)
            except Exception as exc:
                result = {"error": f"`{name}` failed ({type(exc).__name__}); try again or go on."}
    ok = "error" not in result
    message = {
        "role": "tool",
        "tool_call_id": call.id,
        "name": name,
        "content": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
    }
    arguments_text = raw if isinstance(raw, str) else json.dumps(raw)
    record = ToolCallRecord(
        name=name,
        arguments=_shorten(arguments_text, _TRACE_ARGS_CHARS),
        ok=ok,
        ms=round((time.perf_counter() - start) * 1000, 1),
        result=_shorten(message["content"], _TRACE_RESULT_CHARS),
        round=round_no,
    )
    return message, record


def _text_of(message: Any) -> str:
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


async def run_tool_loop(
    meter: TurnMeter,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    execute: ToolExecutor,
    max_rounds: int = MAX_TOOL_ROUNDS,
    deadline_s: float = DEADLINE_S,
    **model_kwargs: Any,
) -> ToolLoopResult:
    """Run the loop on a copy of `messages`; every model call goes through `meter`."""
    convo = list(messages)
    result = ToolLoopResult(text="")
    started = time.monotonic()
    seen: set[tuple[str, str]] = set()
    note = OUT_OF_BUDGET_NOTE
    while True:
        if result.rounds >= max_rounds:
            result.stop_reason = MAX_ROUNDS
            break
        if time.monotonic() - started >= deadline_s:
            result.stop_reason = DEADLINE
            break
        response = await meter.complete(
            model=model, messages=list(convo), tools=tools, **model_kwargs
        )
        message = response.choices[0].message
        requested = list(getattr(message, "tool_calls", None) or [])
        if not requested:
            result.text = _text_of(message)
            return result
        signatures = [_signature(c) for c in requested]
        if all(sig in seen for sig in signatures):
            # Nothing new was asked: do not run it, and do not leave the request unanswered.
            result.stop_reason, note = REPEATED_CALL, REPEATED_CALL_NOTE
            break
        result.rounds += 1
        convo.append(_assistant_message(message, requested))
        with meter.phase("tools"):
            outcomes = await asyncio.gather(
                *(
                    _run_one(call, execute, result.rounds, skipped=i >= MAX_CALLS_PER_ROUND)
                    for i, call in enumerate(requested)
                )
            )
        for sig, (tool_message, record) in zip(signatures, outcomes):
            convo.append(tool_message)
            result.calls.append(record)
            if record.ok:
                seen.add(sig)
    convo.append({"role": "user", "content": note})
    response = await meter.complete(
        model=model, messages=list(convo), tools=tools, tool_choice="none", **model_kwargs
    )
    result.text = _text_of(response.choices[0].message)
    return result
