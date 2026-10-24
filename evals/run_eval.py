"""
Eval harness entry point (kept for the documented command).

This now runs the agent-in-the-loop evaluation; see evals/agent_eval/cli.py for the flags.
It defaults to a dry run that makes no calls:

  python evals/run_eval.py                         # plan only
  python evals/run_eval.py --mode offline          # pipeline check on a fake model
  python evals/run_eval.py --mode live --provider free
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.agent_eval.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
