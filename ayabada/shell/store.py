"""SQLite persistence: a restart must not forget three weeks of "normal".

Stores three things, all product scope:

- **observations** — every (ts, metric, value) interval the heartbeat saw.
  On boot they are replayed through the heartbeat so the seasonal baseline,
  the chart window and the gate state are rebuilt exactly; rows older than
  the retention window are pruned.
- **incidents** — the run record and the rendered handoff per wake.
- **service_events** — up/down transitions of watched services.

One connection, one writer lock; HTTP handlers never touch the store (they
read the in-memory state the store rehydrated).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator

RETENTION_WEEKS = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    ts TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_observations_ts ON observations (ts);
CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    at TEXT NOT NULL,
    record TEXT NOT NULL,
    handoff TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS service_events (
    at TEXT NOT NULL,
    service TEXT NOT NULL,
    transition TEXT NOT NULL,
    detail TEXT NOT NULL
);
"""


class StateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def add_observations(self, ts_iso: str, observations: dict[str, float]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT INTO observations (ts, metric, value) VALUES (?, ?, ?)",
                [(ts_iso, metric, float(value)) for metric, value in observations.items()],
            )
            self._conn.commit()

    def iter_intervals(self) -> Iterator[tuple[str, dict[str, float]]]:
        """Yield (ts, {metric: value}) grouped by timestamp, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, metric, value FROM observations ORDER BY ts"
            ).fetchall()
        current_ts: str | None = None
        bucket: dict[str, float] = {}
        for ts, metric, value in rows:
            if ts != current_ts:
                if current_ts is not None:
                    yield current_ts, bucket
                current_ts, bucket = ts, {}
            bucket[metric] = value
        if current_ts is not None:
            yield current_ts, bucket

    def observation_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]

    def last_ts(self) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT MAX(ts) FROM observations").fetchone()
        return row[0]

    def prune_before(self, ts_iso: str) -> int:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM observations WHERE ts < ?", (ts_iso,))
            self._conn.commit()
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Incidents & service events
    # ------------------------------------------------------------------
    def add_incident(self, incident_id: str, at: str, record: dict[str, Any], handoff: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO incidents (id, at, record, handoff) VALUES (?, ?, ?, ?)",
                (incident_id, at, json.dumps(record), handoff),
            )
            self._conn.commit()

    def load_incidents(self) -> list[tuple[str, str, dict[str, Any], str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, at, record, handoff FROM incidents ORDER BY at"
            ).fetchall()
        return [(r[0], r[1], json.loads(r[2]), r[3]) for r in rows]

    def add_service_event(self, at: str, service: str, transition: str, detail: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO service_events (at, service, transition, detail) VALUES (?, ?, ?, ?)",
                (at, service, transition, detail),
            )
            self._conn.commit()

    def load_service_events(self, limit: int = 50) -> list[dict[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT at, service, transition, detail FROM service_events "
                "ORDER BY rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"at": r[0], "service": r[1], "transition": r[2], "detail": r[3]} for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
