"""Atomic live-request limits shared by all sessions on one app instance."""

import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


class CallLimitError(ValueError):
    """The public demo cannot issue another hosted request right now."""


@dataclass(frozen=True)
class CallLimits:
    daily: int = 100
    session_daily: int = 10
    cooldown_seconds: float = 5

    def __post_init__(self) -> None:
        if self.daily < 1 or self.session_daily < 1 or self.cooldown_seconds < 0:
            raise ValueError("Request limits must be positive; cooldown cannot be negative")


class CallBudget:
    """Reserve before a provider call; failures also consume a reservation.

    SQLite transactions make concurrent reservations atomic. Counts survive process
    restarts when the filesystem survives. A cloud rebuild can discard this file;
    this is a demo traffic limit, not a provider billing limit or user identity check.
    """

    def __init__(self, path: Path, limits: CallLimits | None = None) -> None:
        self.path = path
        self.limits = limits or CallLimits()
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path, timeout=10) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS usage "
                "(day TEXT, session TEXT, calls INTEGER, last_call REAL, "
                "PRIMARY KEY(day, session))"
            )

    def reserve(self, session: str, *, now: float | None = None) -> None:
        if not session or len(session) > 100:
            raise ValueError("Invalid demo session")
        now = time.time() if now is None else now
        day = datetime.fromtimestamp(now, UTC).date().isoformat()
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute("BEGIN IMMEDIATE")
            total = db.execute(
                "SELECT COALESCE(SUM(calls), 0) FROM usage WHERE day=?", (day,)
            ).fetchone()[0]
            row = db.execute(
                "SELECT calls, last_call FROM usage WHERE day=? AND session=?", (day, session)
            ).fetchone()
            if total >= self.limits.daily:
                raise CallLimitError(
                    "Today's shared live LLM limit has been reached. "
                    "The local decision model remains available."
                )
            if row and row[0] >= self.limits.session_daily:
                raise CallLimitError(
                    "This session has reached its daily live LLM limit. "
                    "The local decision model remains available."
                )
            if row and now - row[1] < self.limits.cooldown_seconds:
                raise CallLimitError("Please wait a few seconds before another live LLM request.")
            db.execute(
                "INSERT INTO usage VALUES (?, ?, 1, ?) "
                "ON CONFLICT(day, session) DO UPDATE SET "
                "calls=calls+1, last_call=excluded.last_call",
                (day, session, now),
            )
            db.execute("DELETE FROM usage WHERE day < ?", (day,))
