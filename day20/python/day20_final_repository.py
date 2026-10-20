"""Day20 最终版数据库层：多会话、WAL、事务和幂等问答。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parents[1] / "database" / "day20_final.db"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    client_id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class TurnRecord:
    turn_id: int
    request_id: str
    session_id: str
    question: str
    answer: str
    created_at: str


class FinalRepository:
    def __init__(self, path: Path = DATABASE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def initialize(self) -> None:
        connection = self.connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_client_updated
                    ON sessions(client_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_turns_session_turn
                    ON turns(session_id, turn_id);
                """
            )
            connection.commit()
        finally:
            connection.close()

    def create_session(self, session_id: str, client_id: str, title: str) -> SessionRecord:
        timestamp = now()
        connection = self.connect()
        try:
            with connection:
                connection.execute(
                    "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
                    (session_id, client_id, title, timestamp, timestamp),
                )
        finally:
            connection.close()
        return SessionRecord(session_id, client_id, title, timestamp, timestamp)

    def get_session(self, session_id: str) -> SessionRecord | None:
        connection = self.connect()
        try:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            return self._session(row) if row else None
        finally:
            connection.close()

    def list_sessions(self, client_id: str) -> list[SessionRecord]:
        connection = self.connect()
        try:
            rows = connection.execute(
                """SELECT * FROM sessions WHERE client_id = ?
                   ORDER BY updated_at DESC, session_id DESC""",
                (client_id,),
            ).fetchall()
            return [self._session(row) for row in rows]
        finally:
            connection.close()

    def load_turns(self, session_id: str, limit: int = 20) -> list[TurnRecord]:
        connection = self.connect()
        try:
            rows = connection.execute(
                """SELECT * FROM (
                       SELECT * FROM turns WHERE session_id = ?
                       ORDER BY turn_id DESC LIMIT ?
                   ) ORDER BY turn_id ASC""",
                (session_id, limit),
            ).fetchall()
            return [self._turn(row) for row in rows]
        finally:
            connection.close()

    def get_turn_by_request(self, request_id: str) -> TurnRecord | None:
        connection = self.connect()
        try:
            row = connection.execute(
                "SELECT * FROM turns WHERE request_id = ?", (request_id,)
            ).fetchone()
            return self._turn(row) if row else None
        finally:
            connection.close()

    def save_turn(
        self, request_id: str, session_id: str, question: str, answer: str
    ) -> tuple[bool, TurnRecord]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM turns WHERE request_id = ?", (request_id,)
            ).fetchone()
            if row:
                connection.commit()
                return False, self._turn(row)

            timestamp = now()
            cursor = connection.execute(
                """INSERT INTO turns
                   (request_id, session_id, question, answer, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (request_id, session_id, question, answer, timestamp),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (timestamp, session_id),
            )
            connection.commit()
            return True, TurnRecord(
                int(cursor.lastrowid), request_id, session_id,
                question, answer, timestamp
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _session(row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            row["session_id"], row["client_id"], row["title"],
            row["created_at"], row["updated_at"]
        )

    @staticmethod
    def _turn(row: sqlite3.Row) -> TurnRecord:
        return TurnRecord(
            row["turn_id"], row["request_id"], row["session_id"],
            row["question"], row["answer"], row["created_at"]
        )
