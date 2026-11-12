"""Old entry point, kept for one release: `python -m evals feedback-to-item` is the command now."""
import sys

from evals.__main__ import legacy

if __name__ == "__main__":
    sys.exit(legacy("feedback-to-item"))
