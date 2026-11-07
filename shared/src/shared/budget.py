"""
Spend accounting for the metered real-model calls: eval runs and the IELTS mock.

The ledger is an append-only JSON-lines log (default `.eval-ledger/ledger.jsonl`, gitignored)
that accumulates the cost of every call, summed on read. Every call is checked against the
cumulative ceiling *before* it is made, using a worst-case projection, so a run aborts
instead of overshooting. Free models (`:free`, `openrouter/free`) cost zero.

The ceiling is the owner's AI spend cap for the IELTS mock and the evals (5 USD), which share
one ledger, so there is one number to look at. Other model calls (the GRE tutor's /chat) are
not counted against it; they are measured, uncapped, in the per-turn log (`shared.turn_meter`).
`spend_ledger()` is that shared ledger as configured (`AI_LEDGER_PATH`, `AI_SPEND_CAP_USD`).
"""
from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.src.shared.config import get_settings

_log = logging.getLogger(__name__)

DEFAULT_CEILING_USD = 5.0
DEFAULT_LEDGER_PATH = Path(".eval-ledger/ledger.jsonl")

# USD per 1M tokens (input, output). Unknown paid models fall back to a deliberately
# pessimistic price so the guard errs on the side of aborting.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "o1": (15.00, 60.00),
    "typesafe/jev-1.14": (0.042, 0.0),
    # Gemma on the captain's Google AI Studio key (free tier): no charge to the ledger
    "gemma-4-26b-a4b-it": (0.0, 0.0),
    "gemma-4-31b-it": (0.0, 0.0),
}
_UNKNOWN_PRICE = (10.0, 30.0)


class BudgetExceededError(RuntimeError):
    """Raised before a call that could push cumulative spend over the ceiling."""


def _bare(model: str) -> str:
    for prefix in ("openrouter/", "openai/", "gemini/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def is_free(model: str) -> bool:
    bare = _bare(model)
    return bare.endswith(":free") or bare == "openrouter/free"


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if is_free(model):
        return 0.0
    price_in, price_out = _PRICES.get(_bare(model), _UNKNOWN_PRICE)
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _locations(path: Path) -> tuple[Path, Path | None]:
    """(the JSON-lines log, the legacy single-JSON file it replaces or None) for a configured path.

    The ledger used to be one JSON file, `ledger.json`. Configured paths ending in `.json` (an
    old `AI_LEDGER_PATH`, a test fixture) keep working: the log is the sibling `.jsonl` and the
    configured file is the legacy ledger to migrate. A path ending in `.jsonl` is the log itself.
    """
    if path.suffix == ".json":
        return path.with_suffix(".jsonl"), path
    if path.suffix == ".jsonl":
        return path, path.with_suffix(".json")
    return path, None


@dataclass
class Ledger:
    """
    Cumulative spend, kept as an append-only JSON-lines log: one line per paid call, summed on read.

    Each `record` is one `O_APPEND` write of one short line, so any number of processes (the
    IELTS mock and the evals share one ledger) can record at once without a lock and without
    losing a line. A torn or foreign line is skipped on read, never fatal, so a damaged ledger
    cannot turn a budget check into an unrelated error.

    The cap is a soft cap. `ensure_capacity` reads the total and `record` appends later, with no
    lock across the two, so calls in flight at the same moment (in this process or others) can
    each pass the check and together overshoot the ceiling by what they cost. Callers that start
    several calls at once add their own in-flight projection (see the IELTS completer).
    """

    path: Path = DEFAULT_LEDGER_PATH
    ceiling_usd: float = DEFAULT_CEILING_USD

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._log, self._legacy = _locations(self.path)

    def _lock_path(self) -> Path:
        return self._log.with_name(self._log.name + ".lock")

    def _migrate_legacy(self) -> None:
        """One time: fold an old single-JSON ledger into the log so recorded spend is not lost.

        Serialised across processes with a file lock. The migrated lines are written first and
        the old file is renamed to `<name>.migrated` after, so a crash in between is repaired on
        the next call: the log already holds `migrated` lines, so only the rename is repeated.
        A legacy file that cannot be parsed (a ledger the old writer corrupted) is set aside as
        `<name>.unreadable` with a warning; its spend is unknown and is not counted.
        """
        legacy = self._legacy
        if legacy is None or not legacy.exists():
            return
        self._log.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path().open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not legacy.exists():  # another process finished the migration while we waited
                return
            entries: list[dict[str, Any]] = []
            suffix = ".migrated"
            try:
                data = json.loads(legacy.read_text(encoding="utf-8"))
                entries = [e for e in data["entries"] if _cost_of(e) is not None]
            except (OSError, ValueError, KeyError, TypeError):
                suffix = ".unreadable"
                _log.warning("ledger: %s unreadable; set aside, its spend not counted", legacy)
            if suffix == ".migrated" and not any(e.get("migrated") for e in self._read_log()):
                lines = "".join(
                    json.dumps({**e, "migrated": True}, separators=(",", ":")) + "\n"
                    for e in entries
                )
                self._append(lines or json.dumps({"migrated": True}) + "\n")
            os.replace(legacy, legacy.with_name(legacy.name + suffix))

    def _read_log(self) -> list[dict[str, Any]]:
        try:
            text = self._log.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records

    def _append(self, text: str) -> None:
        self._log.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, text.encode("utf-8"))
        finally:
            os.close(fd)

    def spent(self) -> float:
        self._migrate_legacy()
        return sum(c for c in map(_cost_of, self._read_log()) if c is not None)

    def remaining(self) -> float:
        return max(0.0, self.ceiling_usd - self.spent())

    def ensure_capacity(self, projected_usd: float) -> None:
        """Abort when spent + projected would exceed the ceiling (free calls always pass).

        A soft cap: see the class docstring for why concurrent callers can overshoot.
        """
        if projected_usd <= 0:
            return
        spent = self.spent()
        if spent + projected_usd > self.ceiling_usd:
            raise BudgetExceededError(
                f"budget guard: spent ${spent:.4f} + projected ${projected_usd:.4f} "
                f"exceeds ceiling ${self.ceiling_usd:.2f}"
            )

    def record(self, run_id: str, model: str, cost_usd: float) -> None:
        if cost_usd <= 0:
            return
        self._migrate_legacy()
        line = json.dumps(
            {"ts": time.time(), "run_id": run_id, "model": model, "cost_usd": cost_usd},
            separators=(",", ":"),
        )
        self._append(line + "\n")


def _cost_of(entry: Any) -> float | None:
    """The entry's cost, or None for a line that is not a usable spend record."""
    if not isinstance(entry, dict):
        return None
    cost = entry.get("cost_usd")
    if isinstance(cost, bool) or not isinstance(cost, int | float) or not math.isfinite(cost):
        return None
    return float(cost)


def spend_ledger(path: str | Path | None = None, ceiling_usd: float | None = None) -> Ledger:
    """The shared spend ledger at the configured path and cap, unless overridden."""
    settings = get_settings()
    return Ledger(
        Path(path or settings.ai_ledger_path),
        settings.ai_spend_cap_usd if ceiling_usd is None else ceiling_usd,
    )
