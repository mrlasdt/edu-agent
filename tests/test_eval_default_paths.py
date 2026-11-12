"""The eval tools' default folders and files resolve from the project root, not the cwd."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from shared.src.shared.config import PROJECT_ROOT

_PROBE = (
    "from evals.agent_eval.cache import DEFAULT_CACHE_DIR;"
    "from evals.reports.read_transcripts import DEFAULT_RUNS_DIR;"
    "from evals.reports.quant_report import DEFAULT_BASELINE;"
    "print(DEFAULT_CACHE_DIR);print(DEFAULT_RUNS_DIR);print(DEFAULT_BASELINE)"
)


def _defaults_from(cwd: Path) -> list[str]:
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=cwd, capture_output=True, text=True, check=True,
        env={"PYTHONPATH": str(PROJECT_ROOT), "PATH": ""},
    )
    return out.stdout.split()


def test_default_paths_are_the_same_from_any_cwd(tmp_path: Path) -> None:
    from_root = _defaults_from(PROJECT_ROOT)
    from_elsewhere = _defaults_from(tmp_path)
    assert from_elsewhere == from_root
    assert from_root == [
        str(PROJECT_ROOT / ".eval-cache"),
        str(PROJECT_ROOT / ".eval-runs"),
        str(PROJECT_ROOT / "evals/baselines/quant.json"),
    ]


def test_newest_report_and_baseline_found_from_other_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    from evals.reports.quant_report import DEFAULT_BASELINE
    assert DEFAULT_BASELINE.is_file()
