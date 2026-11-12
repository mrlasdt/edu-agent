"""
Tool-use eval for the candidate MCP server: given a plain request, does a model pick the right
tool, with the right arguments, and do its calls succeed?

Ten requests (evals/golden/mcp_tool_use.jsonl), each with the tool it should call first and the
argument values that call must carry. The model is shown the server's real tool list (the names,
descriptions and argument schemas a client such as Claude Desktop is shown) and its calls run
through the real server, with only the corpus HTTP seam answered by a canned note. Calls go
through the product's own tool loop (`agent.tool_loop.run_tool_loop`), so rounds and deadlines
are the product's; a tool error is handed back to the model as an `{"error": ...}` result, which
is what an MCP client does with an `isError` result.

Three rates, each with its n and 95% interval (the same `rate` the quant suite reports):
  right_tool       the first call named the expected tool
  args_match       the first call also carried the expected argument values
  tool_error_rate  tool errors / tool calls, the definition `quant_suite.summarise` uses

Modes:
  offline  a deterministic keyword router stands in for the model. It proves the wiring (tool
           list, execution, scoring) and nothing about any model's quality.
  live     a real model chain through the Router, metered against the spend ledger. Use free
           OpenRouter models: `--mode live` takes the same free chain the quant eval does.

    uv run python -m evals tool-use --mode offline
    uv run python -m evals tool-use --mode live --write-baseline evals/baselines/tool_use.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import TextContent

from evals.agent_eval.backend import Backend
from evals.agent_eval.cache import DEFAULT_CACHE_DIR
from evals.agent_eval.results import Rate
from evals.agent_eval.stats import rate
from evals.agent_eval.transport import FakeToolCalls, MeteredTransport, tagged
from shared.src.shared.budget import spend_ledger
from shared.src.shared.config import anchor
from shared.src.shared.jsonl import read_jsonl
from shared.src.shared.turn_meter import TurnMeter

GOLDEN = Path(__file__).parent / "golden" / "mcp_tool_use.jsonl"
DEFAULT_BASELINE = anchor("evals/baselines/tool_use.json")

SYSTEM_PROMPT = (
    "You help a GRE student study. You have tools for the study notes, practice questions and "
    "model essays. Call the tool that fits the request, then answer from its result."
)

CANNED_NOTE = {
    "text": "The discriminant b^2 - 4ac of a quadratic ax^2 + bx + c tells you its real roots: "
    "positive means two, zero means one repeated root, negative means none.",
    "tier": "global",
    "source_uri": "seed/gre-quant/algebra.txt",
    "page_or_section": "algebra.txt#Discriminant",
    "score": 0.9,
}


@dataclass(frozen=True)
class Task:
    id: str
    ask: str
    tool: str
    args: dict[str, Any]
    query_has: tuple[str, ...] = ()

    def accepts(self, name: str, arguments: dict[str, Any]) -> bool:
        """The call is to the expected tool with the expected values (a query by its keywords)."""
        if name != self.tool or any(arguments.get(k) != v for k, v in self.args.items()):
            return False
        query = str(arguments.get("query", "")).lower()
        return all(word in query for word in self.query_has)


def load_tasks(path: Path = GOLDEN) -> list[Task]:
    rows = read_jsonl(path)
    return [
        Task(r["id"], r["ask"], r["tool"], r["args"], tuple(r.get("query_has", ()))) for r in rows
    ]


# -- the server as a model sees it -------------------------------------------------------------


async def tool_specs(server: FastMCP) -> list[dict[str, Any]]:
    """The server's tool list in the function-calling format a model API takes."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in await server.list_tools()
    ]


def executor(server: FastMCP) -> Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Run a call through the server; an error result becomes {"error": message} for the model."""

    async def execute(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await server.call_tool(name, arguments)
        except ToolError as exc:
            return {"error": str(exc)}
        blocks = result[0] if isinstance(result, tuple) else result
        first = list(blocks)[0]
        assert isinstance(first, TextContent)
        parsed = json.loads(first.text)
        return parsed if isinstance(parsed, dict) else {"result": parsed}

    return execute


def canned_corpus() -> AbstractContextManager[Any]:
    """Answer the server's one outbound call (the corpus search) with a fixed note."""
    from unittest.mock import AsyncMock, patch

    return patch(
        "mcp_servers.candidate_server._search_corpus",
        new=AsyncMock(return_value=[CANNED_NOTE]),
    )


# -- the offline stand-in for a model ----------------------------------------------------------


def keyword_router(model: str, messages: list[dict[str, Any]]) -> str | FakeToolCalls:
    """
    A deliberately naive router: it reads keywords in the request and fills the arguments it can
    see. It never asks for the "detailed" format and does not know the question types, which is
    why offline scores are below 100% and why they say nothing about a real model.
    """
    if any(m.get("role") == "tool" for m in messages):
        return "Here is what I found."
    ask = str(next(m["content"] for m in messages if m["role"] == "user")).lower()
    prompt_id = re.search(r"\baw\d+\b", ask)
    if prompt_id:
        tier = re.search(r"(?:score|scoring|tier)\D{0,3}(\d)|\((\d)\)", ask)
        value = next((int(g) for g in tier.groups() if g), 6) if tier else 6
        call = {"prompt_id": prompt_id.group(), "score_tier": value}
        return FakeToolCalls([("essay_get_model", call)])
    if "practise" in ask or "practice" in ask or "question" in ask or "task" in ask:
        section = "aw" if "essay" in ask or "argument" in ask else "quant"
        kinds = {
            "quantitative comparison": "quantitative_comparison",
            "numeric entry": "numeric_entry",
            "issue": "issue",
            "argument": "argument",
        }
        found = next((v for k, v in kinds.items() if k in ask), "")
        call = {"section": section, "question_type": found}
        return FakeToolCalls([("practice_get_question", call)])
    return FakeToolCalls([("corpus_search", {"query": ask})])


# -- running the tasks -------------------------------------------------------------------------


@dataclass
class TaskResult:
    task_id: str
    first_tool: str | None
    right_tool: bool
    args_match: bool
    calls: int
    errors: int
    error_texts: list[str] = field(default_factory=list)


def _parse_arguments(raw: str) -> dict[str, Any]:
    """The tool loop shortens long argument strings, so a truncated one counts as no match."""
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def run_task(
    task: Task, *, model: str, specs: list[dict[str, Any]], server: FastMCP
) -> TaskResult:
    from services.agent.src.agent.tool_loop import run_tool_loop

    run = await run_tool_loop(
        TurnMeter.detached(),
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.ask},
        ],
        tools=specs,
        execute=executor(server),
        temperature=0,
    )
    first = run.calls[0] if run.calls else None
    arguments = _parse_arguments(first.arguments) if first else {}
    return TaskResult(
        task_id=task.id,
        first_tool=first.name if first else None,
        right_tool=bool(first and first.name == task.tool),
        args_match=bool(first and task.accepts(first.name, arguments)),
        calls=len(run.calls),
        errors=sum(1 for c in run.calls if not c.ok),
        error_texts=[c.result for c in run.calls if not c.ok],
    )


@dataclass
class Summary:
    right_tool: Rate
    args_match: Rate
    tool_error_rate: Rate
    results: list[TaskResult]


def summarise(results: list[TaskResult]) -> Summary:
    calls = sum(r.calls for r in results)
    return Summary(
        right_tool=rate(sum(r.right_tool for r in results), len(results)),
        args_match=rate(sum(r.args_match for r in results), len(results)),
        tool_error_rate=rate(sum(r.errors for r in results), calls),
        results=results,
    )


async def evaluate(
    transport: MeteredTransport, model: str, tasks: list[Task] | None = None
) -> Summary:
    """Run every task against the candidate server through `transport`'s router."""
    from mcp_servers.candidate_server import mcp

    specs = await tool_specs(mcp)
    results: list[TaskResult] = []
    with transport.installed(), canned_corpus(), tagged("agent"):
        for task in tasks or load_tasks():
            results.append(await run_task(task, model=model, specs=specs, server=mcp))
    return summarise(results)


def baseline_json(summary: Summary, model: str, mode: str) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    metrics = {
        name: {"value": None if r.value is None else round(r.value, 4), "n": r.n, "ci95": r.ci95}
        for name, r in (
            ("tool_use.right_tool_rate", summary.right_tool),
            ("tool_use.args_match_rate", summary.args_match),
            ("tool_use.tool_error_rate", summary.tool_error_rate),
        )
    }
    return {
        "schema": "eval-baseline/1",
        "suite": "tool_use",
        "meta": {
            "model": model,
            "mode": mode,
            "tasks": len(summary.results),
            "git_commit": commit or None,
            "date_utc": time.strftime("%Y-%m-%d", time.gmtime()),
        },
        "note": (
            "n is tasks for right_tool and args_match, and tool calls for tool_error_rate. With "
            "ten tasks a difference inside an interval is noise. Offline numbers come from a "
            "keyword router and only prove the wiring."
        ),
        "metrics": metrics,
        "tasks": [asdict(r) for r in summary.results],
    }


def render(summary: Summary, model: str) -> str:
    def line(label: str, r: Rate) -> str:
        ci = "" if r.ci95 is None else f"  95% interval {r.ci95[0]:.2f}-{r.ci95[1]:.2f}"
        value = "n/a" if r.value is None else f"{r.value:.0%}"
        return f"{label:<16}{value:>5}  ({r.successes}/{r.n}){ci}"

    rows = [f"tool-use eval, model {model}", ""]
    for r in summary.results:
        mark = "ok  " if r.args_match else ("tool" if r.right_tool else "MISS")
        rows.append(f"  {r.task_id}  {mark}  first call: {r.first_tool}  errors: {r.errors}")
    rows += [
        "",
        line("right tool", summary.right_tool),
        line("args match", summary.args_match),
        line("tool error rate", summary.tool_error_rate),
    ]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--mode", choices=["offline", "live"], default="offline")
    p.add_argument("--model", default=None, help="pin one litellm model string (live)")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--write-baseline", default=None, help=f"e.g. {DEFAULT_BASELINE}")
    args = p.parse_args(argv)

    from evals.agent_eval.cli import provider_chain

    run_id = f"tool-use-{time.strftime('%Y%m%d-%H%M%S')}"
    ledger = spend_ledger(None, None)
    if args.mode == "live":
        backend = Backend.for_mode(
            "live", chain=provider_chain("free", args.model), judge_model=None,
            cache_dir=None if args.no_cache else DEFAULT_CACHE_DIR,
        )
        backend.preflight()
        transport = backend.transport(ledger, run_id, args.max_tokens)
        model = backend.model
    else:
        model = "offline-keyword-router"
        transport = MeteredTransport(ledger, run_id, fake_model=keyword_router,
                                     max_tokens=args.max_tokens)
    summary = asyncio.run(evaluate(transport, model))
    print(render(summary, model))
    if args.write_baseline:
        out = Path(args.write_baseline)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(baseline_json(summary, model, args.mode), indent=2) + "\n")
        print(f"\nbaseline written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
