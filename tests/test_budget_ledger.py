"""
The spend ledger under concurrent writers, and its move from a JSON file to a JSON-lines log.

The IELTS mock and the evals run as separate processes and share one ledger, so `record` must be
safe when several processes write at once, and an unreadable ledger must never turn a budget
check into a 500.
"""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from typing import Any

import pytest

from shared.src.shared.budget import BudgetExceededError, Ledger

WRITERS = 6
RECORDS_EACH = 40
COST = 0.001


def _write_records(path: str, worker: int, barrier: Any) -> str:
    """Child process: wait for the others, then record RECORDS_EACH entries; return any error."""
    ledger = Ledger(Path(path), ceiling_usd=100.0)
    barrier.wait()
    try:
        for i in range(RECORDS_EACH):
            ledger.record(f"w{worker}-{i}", "gpt-4.1-mini", COST)
    except Exception as exc:  # noqa: BLE001 - the test reports whatever a writer died of
        return f"{type(exc).__name__}: {exc}"
    return ""


def test_six_processes_recording_to_one_ledger_lose_nothing(tmp_path):
    path = tmp_path / "ledger.json"
    ctx = multiprocessing.get_context("spawn")
    with ctx.Manager() as manager:
        barrier = manager.Barrier(WRITERS)
        with ctx.Pool(WRITERS) as pool:
            errors = pool.starmap(
                _write_records, [(str(path), w, barrier) for w in range(WRITERS)]
            )

    assert [e for e in errors if e] == []
    ledger = Ledger(path, ceiling_usd=100.0)
    assert ledger.spent() == pytest.approx(WRITERS * RECORDS_EACH * COST)


def test_a_torn_or_foreign_line_is_skipped_and_the_cap_still_answers(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        '{"cost_usd":4.0}\n{"cost_usd": 0.5, "ru\nnot json\n[1,2]\n'
        '{"cost_usd":"x"}\n{"cost_usd":0.5}\n'
    )
    ledger = Ledger(path, ceiling_usd=5.0)
    assert ledger.spent() == pytest.approx(4.5)
    with pytest.raises(BudgetExceededError):
        ledger.ensure_capacity(1.0)


def test_record_creates_the_directory_and_free_calls_write_nothing(tmp_path):
    ledger = Ledger(tmp_path / "deep" / "ledger.jsonl")
    ledger.record("run", "openrouter/x:free", 0.0)
    assert not (tmp_path / "deep").exists()
    ledger.record("run", "gpt-4.1-mini", 0.25)
    assert Ledger(tmp_path / "deep" / "ledger.jsonl").spent() == pytest.approx(0.25)


def _old_ledger(path: Path, costs: list[float]) -> None:
    entries = [{"ts": 1.0, "run_id": "r", "model": "m", "cost_usd": c} for c in costs]
    path.write_text(json.dumps({"entries": entries}, indent=2))


def test_an_old_json_ledger_is_migrated_once_and_its_spend_kept(tmp_path):
    old = tmp_path / "ledger.json"
    _old_ledger(old, [1.5, 2.0])
    ledger = Ledger(old, ceiling_usd=5.0)  # an old AI_LEDGER_PATH still points at the .json file

    assert ledger.spent() == pytest.approx(3.5)
    assert not old.exists() and (tmp_path / "ledger.json.migrated").exists()
    ledger.record("run", "gpt-4.1-mini", 0.5)
    assert Ledger(tmp_path / "ledger.jsonl").spent() == pytest.approx(4.0)  # the new default path
    assert Ledger(old).spent() == pytest.approx(4.0)  # and no second migration


def test_an_interrupted_migration_is_not_counted_twice(tmp_path):
    old = tmp_path / "ledger.json"
    _old_ledger(old, [1.0, 2.0])
    Ledger(old).spent()  # migrates
    # Simulate a crash after the lines were written but before the old file was renamed away.
    _old_ledger(old, [1.0, 2.0])
    assert Ledger(old).spent() == pytest.approx(3.0)
    assert not old.exists()


def test_an_unreadable_old_ledger_is_set_aside_not_fatal(tmp_path):
    old = tmp_path / "ledger.json"
    old.write_text('{"entries": [{"ts": 1.0, "cost_usd"')  # what the old concurrent writer left
    ledger = Ledger(old, ceiling_usd=5.0)
    assert ledger.spent() == 0.0
    assert (tmp_path / "ledger.json.unreadable").exists()
    ledger.record("run", "gpt-4.1-mini", 1.0)
    assert ledger.spent() == pytest.approx(1.0)


def _spent_in_child(path: str, barrier: Any) -> float:
    barrier.wait()
    return Ledger(Path(path), ceiling_usd=100.0).spent()


def test_processes_racing_to_migrate_count_the_old_spend_exactly_once(tmp_path):
    old = tmp_path / "ledger.json"
    _old_ledger(old, [1.0, 2.0, 3.0])
    ctx = multiprocessing.get_context("spawn")
    with ctx.Manager() as manager:
        barrier = manager.Barrier(WRITERS)
        with ctx.Pool(WRITERS) as pool:
            totals = pool.starmap(_spent_in_child, [(str(old), barrier)] * WRITERS)
    assert totals == [pytest.approx(6.0)] * WRITERS
    assert Ledger(old).spent() == pytest.approx(6.0)
