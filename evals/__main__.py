"""
One front door for the eval tools: `python -m evals <command> [args]`.

    python -m evals                      # list the commands
    python -m evals agent --mode offline # a command takes its own flags; add --help to see them

Each command is the `main(argv)` of one module, imported only when it is chosen, so listing the
commands needs no model client, no database driver and no network.
"""
from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from typing import NamedTuple


class Command(NamedTuple):
    module: str
    summary: str


COMMANDS: dict[str, Command] = {
    "agent": Command("evals.agent_eval.cli", "agent-in-the-loop eval (a dry run unless --mode)"),
    "exam-set": Command("evals.exam_set.__main__", "build, import and summarise the exam sets"),
    "retrieval": Command("evals.retrieval_eval", "retrieval recall, MRR and the relevance floor"),
    "tool-use": Command("evals.tool_use_eval", "can a model use the MCP server's tools"),
    "qdrant-filter": Command("evals.qdrant_filter_check", "filtered-search recall on Qdrant"),
    "turn-report": Command("evals.reports.turn_report", "report on the live chat's turn log"),
    "quant-report": Command("evals.reports.quant_report", "render a Quant suite baseline"),
    "transcripts": Command("evals.reports.read_transcripts", "print the worst transcripts"),
    "feedback-to-item": Command(
        "evals.reports.feedback_to_item", "turn a thumbs-down into a draft eval item"
    ),
}


def usage() -> str:
    width = max(len(name) for name in COMMANDS)
    rows = "\n".join(f"  {name:<{width}}  {c.summary}" for name, c in COMMANDS.items())
    return f"usage: python -m evals <command> [args]\n\ncommands:\n{rows}\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(usage(), end="")
        return 0
    name, rest = args[0], args[1:]
    command = COMMANDS.get(name)
    if command is None:
        print(f"python -m evals: unknown command {name!r}\n\n{usage()}", file=sys.stderr, end="")
        return 2
    # argparse names its program after argv[0]: make `--help` read as the command was typed.
    sys.argv[0] = f"python -m evals {name}"
    return int(importlib.import_module(command.module).main(rest) or 0)


def legacy(name: str) -> int:
    """The old `python -m evals.<module>` entry points call this: same command, with a note."""
    print(
        f"note: this entry point is kept for one more release; use `python -m evals {name}`.",
        file=sys.stderr,
    )
    return main([name, *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
