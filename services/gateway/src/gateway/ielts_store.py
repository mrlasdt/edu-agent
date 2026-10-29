"""
Attempt store for the IELTS-style mock: one local SQLite file, one user.

The clock is server-side: starting an attempt fixes `deadline_at`; the browser countdown is
only a display. An attempt is module-agnostic (`module` column) so later modules reuse it. The
database holds the candidate's essays, so it lives in the gitignored `ielts-local/` directory.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

MODULE_WRITING_T2 = "ielts_writing_t2"
ALLOWED_S = 40 * 60
GRACE_S = 30  # network and clock slack before a submission counts as late
DRAFT_MAX_CHARS = 20_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    id TEXT PRIMARY KEY,
    module TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    deadline_at REAL,
    submitted_at REAL,
    late INTEGER NOT NULL DEFAULT 0,
    draft_text TEXT NOT NULL DEFAULT '',
    draft_saved_at REAL,
    final_text TEXT,
    report_json TEXT,
    error TEXT,
    retryable INTEGER NOT NULL DEFAULT 0,
    graded_at REAL
)
"""


class AttemptNotFoundError(LookupError):
    pass


class AttemptStateError(RuntimeError):
    """The requested transition is not allowed from the attempt's current state."""


@dataclass(frozen=True)
class Attempt:
    id: str
    module: str
    prompt_id: str
    prompt_text: str
    status: str  # created | in_progress | submitted | grading | graded | ungraded
    created_at: float
    started_at: float | None
    deadline_at: float | None
    submitted_at: float | None
    late: bool
    draft_text: str
    draft_saved_at: float | None
    final_text: str | None
    report_json: str | None
    error: str | None
    retryable: bool
    graded_at: float | None

    @property
    def essay(self) -> str:
        return self.final_text if self.final_text is not None else self.draft_text


def _row(r: sqlite3.Row) -> Attempt:
    return Attempt(
        id=r["id"], module=r["module"], prompt_id=r["prompt_id"], prompt_text=r["prompt_text"],
        status=r["status"], created_at=r["created_at"], started_at=r["started_at"],
        deadline_at=r["deadline_at"], submitted_at=r["submitted_at"], late=bool(r["late"]),
        draft_text=r["draft_text"], draft_saved_at=r["draft_saved_at"],
        final_text=r["final_text"], report_json=r["report_json"], error=r["error"],
        retryable=bool(r["retryable"]), graded_at=r["graded_at"],
    )


class AttemptStore:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self._path = path
        self.clock = clock
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self._path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with self._lock:
                yield db
                db.commit()
        finally:
            db.close()

    def get(self, attempt_id: str) -> Attempt:
        with self._connect() as db:
            row = db.execute("SELECT * FROM attempts WHERE id = ?", (attempt_id,)).fetchone()
        if row is None:
            raise AttemptNotFoundError(attempt_id)
        return _row(row)

    def recent(self, limit: int = 20, module: str = MODULE_WRITING_T2) -> list[Attempt]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM attempts WHERE module = ? AND status != 'created' "
                "ORDER BY COALESCE(started_at, created_at) DESC LIMIT ?",
                (module, limit),
            ).fetchall()
        return [_row(r) for r in rows]

    def create(self, prompt_id: str, prompt_text: str, module: str = MODULE_WRITING_T2) -> Attempt:
        attempt_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute(
                "INSERT INTO attempts (id, module, prompt_id, prompt_text, status, created_at) "
                "VALUES (?, ?, ?, ?, 'created', ?)",
                (attempt_id, module, prompt_id, prompt_text, self.clock()),
            )
        return self.get(attempt_id)

    def start(self, attempt_id: str) -> Attempt:
        """Start the clock. Idempotent: a second call returns the running attempt unchanged."""
        attempt = self.get(attempt_id)
        if attempt.status != "created":
            return attempt
        now = self.clock()
        with self._connect() as db:
            db.execute(
                "UPDATE attempts SET status = 'in_progress', started_at = ?, deadline_at = ? "
                "WHERE id = ? AND status = 'created'",
                (now, now + ALLOWED_S, attempt_id),
            )
        return self.get(attempt_id)

    def save_draft(self, attempt_id: str, text: str) -> Attempt:
        attempt = self.get(attempt_id)
        now = self.clock()
        if attempt.status != "in_progress":
            raise AttemptStateError(f"attempt is {attempt.status}, not in progress")
        assert attempt.deadline_at is not None
        if now > attempt.deadline_at + GRACE_S:
            raise AttemptStateError("time is up; submit the attempt")
        with self._connect() as db:
            db.execute(
                "UPDATE attempts SET draft_text = ?, draft_saved_at = ? WHERE id = ?",
                (text, now, attempt_id),
            )
        return self.get(attempt_id)

    def submit(self, attempt_id: str, text: str) -> Attempt:
        """
        Final text; idempotent once submitted. Past deadline + grace it is marked late and the
        text is the last autosave, so nothing written after time was up is marked.
        """
        attempt = self.get(attempt_id)
        if attempt.status != "in_progress":
            if attempt.final_text is not None:
                return attempt
            raise AttemptStateError(f"attempt is {attempt.status}, not in progress")
        assert attempt.deadline_at is not None
        now = self.clock()
        late = now > attempt.deadline_at + GRACE_S
        if late:
            text = attempt.draft_text
        with self._connect() as db:
            db.execute(
                "UPDATE attempts SET status = 'submitted', final_text = ?, draft_text = ?, "
                "submitted_at = ?, late = ? WHERE id = ? AND status = 'in_progress'",
                (text, text, now, int(late), attempt_id),
            )
        return self.get(attempt_id)

    def begin_grading(self, attempt_id: str) -> Attempt:
        """submitted -> grading; also re-opens an ungraded attempt whose failure was retryable."""
        attempt = self.get(attempt_id)
        retry = attempt.status == "ungraded" and attempt.retryable
        if attempt.status not in ("submitted", "grading") and not retry:
            raise AttemptStateError(f"attempt is {attempt.status}, cannot be marked")
        with self._connect() as db:
            db.execute(
                "UPDATE attempts SET status = 'grading', error = NULL, retryable = 0 WHERE id = ?",
                (attempt_id,),
            )
        return self.get(attempt_id)

    def finish(self, attempt_id: str, report_json: str, graded: bool) -> Attempt:
        with self._connect() as db:
            db.execute(
                "UPDATE attempts SET status = ?, report_json = ?, graded_at = ?, error = NULL, "
                "retryable = 0 WHERE id = ?",
                ("graded" if graded else "ungraded", report_json, self.clock(), attempt_id),
            )
        return self.get(attempt_id)

    def fail(self, attempt_id: str, error: str, retryable: bool = True) -> Attempt:
        with self._connect() as db:
            db.execute(
                "UPDATE attempts SET status = 'ungraded', error = ?, retryable = ? WHERE id = ?",
                (error, int(retryable), attempt_id),
            )
        return self.get(attempt_id)

    def previous_graded(self, attempt: Attempt) -> Attempt | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM attempts WHERE module = ? AND status = 'graded' AND id != ? "
                "AND submitted_at < ? ORDER BY submitted_at DESC LIMIT 1",
                (attempt.module, attempt.id, attempt.submitted_at or self.clock()),
            ).fetchone()
        return _row(row) if row else None
