"""SQLite-backed repository for runs, logs, chat, and settings overrides.

Design notes:
  * One shared connection (``check_same_thread=False``) guarded by a lock —
    simple and correct for a single-process service. WAL mode keeps readers
    from blocking the writer.
  * All methods are synchronous. Route handlers call them through
    ``asyncio.to_thread`` so the FastAPI event loop stays responsive.
  * Schema is created lazily on first connection, so merely importing the app
    never touches disk (keeps unit tests that don't need the DB clean).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

# Status lifecycle for a run.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id             TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    finished_at    TEXT,
    status         TEXT NOT NULL,
    mode           TEXT NOT NULL,
    source         TEXT NOT NULL,
    filename       TEXT NOT NULL,
    num_pages      INTEGER,
    num_questions  INTEGER,
    parser_backend TEXT,
    llm_provider   TEXT,
    vision_provider TEXT,
    markdown       TEXT,
    parse_json     TEXT,
    qa_json        TEXT,
    timings_json   TEXT,
    meta_json      TEXT,
    error          TEXT,
    duration_ms    REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs (created_at DESC);

CREATE TABLE IF NOT EXISTS run_logs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL,
    seq       INTEGER NOT NULL,
    ts        TEXT NOT NULL,
    level     TEXT NOT NULL,
    event     TEXT NOT NULL,
    data_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_logs_run ON run_logs (run_id, seq);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    sources_json TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_run ON chat_messages (run_id, id);

CREATE TABLE IF NOT EXISTS settings_overrides (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# Columns returned in list views — heavy text columns are deliberately excluded.
_SUMMARY_COLS = (
    "id, created_at, updated_at, finished_at, status, mode, source, filename, "
    "num_pages, num_questions, parser_backend, llm_provider, vision_provider, "
    "error, duration_ms"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads(value: str | None) -> Any:
    if value in (None, ""):
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


class RunRepository:
    """Thread-safe CRUD over the SQLite store."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    # -- connection / schema ------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        path = Path(self._db_path)
        if path.parent and str(path.parent) not in ("", "."):
            path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.executescript(_SCHEMA)
        conn.commit()
        self._conn = conn
        return conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # -- runs ---------------------------------------------------------------
    def create_run(
        self,
        *,
        mode: str,
        source: str,
        filename: str,
        num_questions: int | None,
        parser_backend: str,
        llm_provider: str,
        vision_provider: str,
    ) -> str:
        run_id = uuid.uuid4().hex
        now = _now()
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT INTO runs (id, created_at, updated_at, status, mode,
                       source, filename, num_questions, parser_backend,
                       llm_provider, vision_provider)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, now, now, STATUS_QUEUED, mode, source, filename,
                    num_questions, parser_backend, llm_provider, vision_provider,
                ),
            )
            conn.commit()
        return run_id

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [run_id]
        with self._lock:
            conn = self._connect()
            conn.execute(f"UPDATE runs SET {cols} WHERE id = ?", values)
            conn.commit()

    def get_run(self, run_id: str, *, include_markdown: bool = False) -> dict | None:
        with self._lock:
            conn = self._connect()
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_run(row, include_markdown=include_markdown)

    def list_runs(self, *, limit: int = 50, offset: int = 0) -> list[dict]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                f"SELECT {_SUMMARY_COLS} FROM runs "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_runs(self) -> int:
        with self._lock:
            conn = self._connect()
            return conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    def delete_run(self, run_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            cur = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.execute("DELETE FROM run_logs WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM chat_messages WHERE run_id = ?", (run_id,))
            conn.commit()
            return cur.rowcount > 0

    @staticmethod
    def _row_to_run(row: sqlite3.Row, *, include_markdown: bool) -> dict:
        run = dict(row)
        run["parse"] = _loads(run.pop("parse_json", None))
        run["qa"] = _loads(run.pop("qa_json", None))
        run["timings_ms"] = _loads(run.pop("timings_json", None)) or {}
        run["meta"] = _loads(run.pop("meta_json", None)) or {}
        markdown = run.pop("markdown", None)
        if include_markdown:
            run["markdown"] = markdown
        run["has_markdown"] = bool(markdown)
        return run

    def get_markdown(self, run_id: str) -> str | None:
        """Cheap fetch of just the parsed text — used to ground Q&A chat."""
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT markdown FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return row["markdown"] if row else None

    # -- logs ---------------------------------------------------------------
    def append_log(
        self, run_id: str, level: str, event: str, data: dict | None = None
    ) -> int:
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM run_logs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            seq = row[0]
            conn.execute(
                "INSERT INTO run_logs (run_id, seq, ts, level, event, data_json) "
                "VALUES (?,?,?,?,?,?)",
                (run_id, seq, _now(), level, event,
                 json.dumps(data, default=str) if data else None),
            )
            conn.commit()
        return seq

    def get_logs(self, run_id: str, *, after_seq: int = 0) -> list[dict]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT seq, ts, level, event, data_json FROM run_logs "
                "WHERE run_id = ? AND seq > ? ORDER BY seq",
                (run_id, after_seq),
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["data"] = _loads(item.pop("data_json", None)) or {}
            out.append(item)
        return out

    # -- chat ---------------------------------------------------------------
    def add_chat_message(
        self, run_id: str, role: str, content: str, sources: list[int] | None = None
    ) -> dict:
        now = _now()
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "INSERT INTO chat_messages (run_id, role, content, sources_json, "
                "created_at) VALUES (?,?,?,?,?)",
                (run_id, role, content,
                 json.dumps(sources) if sources else None, now),
            )
            conn.commit()
            msg_id = cur.lastrowid
        return {
            "id": msg_id, "run_id": run_id, "role": role, "content": content,
            "sources": sources or [], "created_at": now,
        }

    def get_chat(self, run_id: str) -> list[dict]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT id, role, content, sources_json, created_at "
                "FROM chat_messages WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["sources"] = _loads(item.pop("sources_json", None)) or []
            out.append(item)
        return out

    # -- settings overrides -------------------------------------------------
    def get_overrides(self) -> dict[str, Any]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute("SELECT key, value FROM settings_overrides").fetchall()
        return {r["key"]: _loads(r["value"]) for r in rows}

    def set_overrides(self, values: dict[str, Any]) -> None:
        now = _now()
        with self._lock:
            conn = self._connect()
            for key, val in values.items():
                if val is None:
                    conn.execute(
                        "DELETE FROM settings_overrides WHERE key = ?", (key,)
                    )
                else:
                    conn.execute(
                        "INSERT INTO settings_overrides (key, value, updated_at) "
                        "VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET "
                        "value = excluded.value, updated_at = excluded.updated_at",
                        (key, json.dumps(val), now),
                    )
            conn.commit()


@lru_cache(maxsize=8)
def get_repository(db_path: str) -> RunRepository:
    """Return a cached repository for a given DB path (one per path)."""
    return RunRepository(db_path)
