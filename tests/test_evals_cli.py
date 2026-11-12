"""`python -m evals <command>`: the one front door, and the old entry points that still work."""
from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from evals.__main__ import COMMANDS, main
from shared.src.shared.config import PROJECT_ROOT


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
        env={"PYTHONPATH": str(PROJECT_ROOT), "PATH": ""},
    )


def test_no_command_lists_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    listing = capsys.readouterr().out
    assert all(name in listing for name in COMMANDS)


def test_unknown_command_is_an_error_that_lists_the_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["nope"]) == 2
    err = capsys.readouterr().err
    assert "unknown command 'nope'" in err
    assert all(name in err for name in COMMANDS)


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_every_command_names_a_module_with_a_main(name: str) -> None:
    assert callable(importlib.import_module(COMMANDS[name].module).main)


def test_the_rest_of_the_line_goes_to_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    from evals.reports import turn_report

    seen: list[list[str]] = []
    monkeypatch.setattr(turn_report, "main", lambda argv: seen.append(list(argv)) or 3)
    assert main(["turn-report", "--last", "3"]) == 3
    assert seen == [["--last", "3"]]


def test_a_command_runs_end_to_end_from_the_shell() -> None:
    done = _run("-m", "evals", "quant-report")  # renders the committed Quant baseline
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip()


@pytest.mark.parametrize(
    "module,command",
    [
        ("turn_report", "turn-report"),
        ("quant_report", "quant-report"),
        ("read_transcripts", "transcripts"),
        ("feedback_to_item", "feedback-to-item"),
    ],
)
def test_the_old_entry_points_still_run_and_point_at_the_new_one(
    module: str, command: str
) -> None:
    done = _run("-m", f"evals.{module}", "--help")
    assert done.returncode == 0, done.stderr
    assert f"python -m evals {command}" in done.stderr
    assert done.stdout.startswith("usage:")
