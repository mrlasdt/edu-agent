"""
Turn one rated reply into a DRAFT eval item the owner can finish.

    uv run python -m evals.feedback_to_item <trace_id>
    uv run python -m evals.feedback_to_item <trace_id> --out my-drafts.jsonl

The trace id is the one `python -m evals.turn_report --feedback` prints next to each thumbs-down.
The item is appended to `.turn-log/draft-items.jsonl` (local and gitignored: it may hold the
learner's text). It is shaped like a Quant golden item (`evals/golden/gre_quant.jsonl`) with
`question`, `ground_truth` and the like left for the owner, plus the trace facts that explain the
failure. Nothing is added to a golden file: a draft becomes an eval item only when the owner
completes it, checks the ground truth by hand and moves it there.

The question and the reply under review are filled in only while message capture is on
(`TURN_CAPTURE_TEXT=1`, `shared.feedback`) and the turn was captured. Otherwise, even if an older
capture file exists, the item is a stub: the trace id, the agent and the vote, and the owner
re-enters the question.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.turn_report import merge_turns
from shared.src.shared.config import (
    DEFAULT_FEEDBACK_LOG_PATH,
    DEFAULT_TURN_LOG_PATH,
    get_settings,
)
from shared.src.shared.feedback import latest_votes
from shared.src.shared.turn_meter import TurnLog

DEFAULT_OUT = ".turn-log/draft-items.jsonl"


def build_draft(
    trace_id: str,
    turn: dict[str, Any] | None,
    vote: dict[str, Any] | None,
    capture: dict[str, Any] | None,
) -> dict[str, Any]:
    """The draft item. `turn` and `vote` come from the logs; `capture` only when it was opted in."""
    mode = (turn or {}).get("mode", "")
    needs = ["ground_truth", "question_type", "tag"]
    if mode == "solve":
        needs.append("expression")
    question = str((capture or {}).get("question", ""))
    if not question:
        needs.insert(0, "question")
    rated = f"user rated {vote['vote']}" if vote else "no rating"
    comment = (vote or {}).get("comment", "")
    agent = (turn or {}).get("agent") or "?"
    version = (turn or {}).get("prompt_version") or "-"
    return {
        "id": f"fb-{trace_id[:8]}",
        "draft": True,
        "needs": needs,
        "kind": mode if mode in ("solve", "tutor") else "tutor",
        "tag": "",
        "failure_source": (
            f"{rated} on trace {trace_id} ({agent} prompt {version})"
            + (f": {comment}" if comment else "")
        ),
        "question_type": "",
        "question": question,
        "expression": "",
        "ground_truth": "",
        "reply_under_review": str((capture or {}).get("reply", "")),
        "comment": comment,
        "trace": {
            "trace_id": trace_id,
            "agent": (turn or {}).get("agent", ""),
            "mode": mode,
            "prompt_version": (turn or {}).get("prompt_version", ""),
            "outcome": (turn or {}).get("outcome", ""),
            "flags": (turn or {}).get("flags", []),
        },
        "drafted_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_draft(path: Path, item: dict[str, Any]) -> bool:
    """Append `item`, or replace the earlier draft with the same id. True when it replaced one."""
    kept: list[str] = []
    replaced = False
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                same = json.loads(line).get("id") == item["id"]
            except (ValueError, AttributeError):
                same = False
            if same:
                replaced = True
            elif line.strip():
                kept.append(line)
    kept.append(json.dumps(item, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return replaced


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.feedback_to_item",
        description="Write a draft eval item for one rated reply.",
    )
    parser.add_argument("trace_id")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"draft file (default: {DEFAULT_OUT})")
    parser.add_argument("--log", default=None, help="turn log (default: TURN_LOG_PATH)")
    parser.add_argument("--feedback-log", default=None, help="feedback log (FEEDBACK_LOG_PATH)")
    parser.add_argument("--capture-log", default=None, help="capture file (TURN_CAPTURE_PATH)")
    args = parser.parse_args(argv)

    settings = get_settings()
    turn_records = TurnLog(args.log or settings.turn_log_path or DEFAULT_TURN_LOG_PATH).read()
    turn = next((t for t in merge_turns(turn_records) if t["trace_id"] == args.trace_id), None)
    feedback_path = args.feedback_log or settings.feedback_log_path or DEFAULT_FEEDBACK_LOG_PATH
    vote = latest_votes(TurnLog(feedback_path).read()).get(args.trace_id)
    if turn is None and vote is None:
        print(
            f"unknown trace id {args.trace_id}: not in the turn log or the feedback log",
            file=sys.stderr,
        )
        return 1
    capture_path = args.capture_log or settings.turn_capture_path
    capture = (
        TurnLog(capture_path).last_with_trace(args.trace_id)
        if settings.turn_capture_text and capture_path
        else None
    )

    item = build_draft(args.trace_id, turn, vote, capture)
    out = Path(args.out)
    replaced = write_draft(out, item)
    print(f"{'updated' if replaced else 'wrote'} draft {item['id']} in {out}")
    if capture is None:
        print(
            "no captured text for this turn (TURN_CAPTURE_TEXT is off or the turn was not "
            "captured): re-enter the question by hand"
        )
    print("still to fill in: " + ", ".join(item["needs"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
