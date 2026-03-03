"""可观测历史存储（SQLite）。"""

from __future__ import annotations

import os
import pickle
import sqlite3
from typing import List, Optional

from ...core.observability import ObsEventEnvelope

_SQLITE_TIMEOUT = 30.0


class SQLiteHistoryStore:
    """run 级历史事件存储（SQLite）。"""

    def __init__(self, storage_dir: str) -> None:
        self._storage_dir = os.path.abspath(storage_dir)
        self._run_id: Optional[str] = None
        self._db_path: Optional[str] = None
        self._write_conn: Optional[sqlite3.Connection] = None

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    @property
    def db_path(self) -> Optional[str]:
        return self._db_path

    def start_run(self, run_id: str) -> None:
        self.close_writer()
        os.makedirs(self._storage_dir, exist_ok=True)
        db_path = os.path.join(self._storage_dir, f"run_{run_id}.sqlite")
        conn = sqlite3.connect(db_path, timeout=_SQLITE_TIMEOUT, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "seq INTEGER PRIMARY KEY, "
            "event_type TEXT NOT NULL, "
            "data BLOB NOT NULL)"
        )
        conn.commit()
        self._run_id = run_id
        self._db_path = db_path
        self._write_conn = conn

    def append(self, event: ObsEventEnvelope) -> None:
        if self._write_conn is None:
            raise RuntimeError("History writer is not initialized")
        payload = sqlite3.Binary(pickle.dumps(event, protocol=pickle.HIGHEST_PROTOCOL))
        self._write_conn.execute(
            "INSERT INTO events (seq, event_type, data) VALUES (?, ?, ?)",
            (int(event.seq), str(event.event_type), payload),
        )
        self._write_conn.commit()

    def fetch_range(self, start_seq: int, end_seq: int, limit: int) -> List[ObsEventEnvelope]:
        if start_seq > end_seq:
            return []
        if self._db_path is None or not os.path.exists(self._db_path):
            return []
        query = (
            "SELECT data FROM events WHERE seq >= ? AND seq <= ? "
            "ORDER BY seq ASC LIMIT ?"
        )
        conn = sqlite3.connect(self._db_path, timeout=_SQLITE_TIMEOUT, check_same_thread=False)
        try:
            rows = conn.execute(query, (int(start_seq), int(end_seq), int(limit))).fetchall()
        finally:
            conn.close()
        return [pickle.loads(row[0]) for row in rows]

    def close_writer(self) -> None:
        if self._write_conn is None:
            return
        self._write_conn.commit()
        self._write_conn.close()
        self._write_conn = None

    def cleanup(self) -> None:
        self.close_writer()
        if self._db_path is None:
            return
        for suffix in ("", "-wal", "-shm"):
            path = f"{self._db_path}{suffix}"
            if os.path.exists(path):
                os.remove(path)
        self._db_path = None

    def has_history(self) -> bool:
        return bool(self._db_path and os.path.exists(self._db_path))
