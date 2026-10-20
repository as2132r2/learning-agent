"""Day20 第五步：支持 WAL、事务和请求幂等的 SQLite Repository。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_DATABASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "day20_safe_chat.db"
)


def local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


@dataclass(frozen=True)
class PersistedTurn:
    turn_id: int
    request_id: str
    session_id: str
    question: str
    answer: str
    created_at: str


@dataclass(frozen=True)
class SaveResult:
    created: bool
    turn: PersistedTurn


class SafeChatRepository:
    """每次操作使用独立连接，并通过 request_id 保证幂等。"""

    def __init__(self, database_path: Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=10,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def initialize(self) -> str:
        """初始化表结构并启用 WAL；返回实际日志模式。"""

        connection = self._connect()
        try:
            journal_mode = connection.execute(
                "PRAGMA journal_mode = WAL"
            ).fetchone()[0]

            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id)
                        REFERENCES chat_sessions(session_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session_turn
                ON chat_turns(session_id, turn_id);
                """
            )
            connection.commit()
            return str(journal_mode)
        finally:
            connection.close()

    def create_session_if_missing(
        self,
        session_id: str,
        client_id: str,
        title: str,
    ) -> bool:
        """创建会话；已存在时不重复创建。"""

        timestamp = local_now()
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO chat_sessions (
                        session_id, client_id, title, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, client_id, title, timestamp, timestamp),
                )
                return cursor.rowcount > 0
        finally:
            connection.close()

    def save_turn_idempotent(
        self,
        request_id: str,
        session_id: str,
        question: str,
        answer: str,
    ) -> SaveResult:
        """同一个 request_id 无论提交多少次，都只保存一轮。"""

        connection = self._connect()
        try:
            # 提前取得写锁，使“先查再写”成为一个不可穿插的事务。
            connection.execute("BEGIN IMMEDIATE")

            existing = connection.execute(
                """
                SELECT turn_id, request_id, session_id,
                       question, answer, created_at
                FROM chat_turns
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()

            if existing is not None:
                connection.commit()
                return SaveResult(
                    created=False,
                    turn=self._row_to_turn(existing),
                )

            timestamp = local_now()
            cursor = connection.execute(
                """
                INSERT INTO chat_turns (
                    request_id, session_id, question, answer, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (request_id, session_id, question, answer, timestamp),
            )
            turn_id = int(cursor.lastrowid)

            connection.execute(
                """
                UPDATE chat_sessions
                SET updated_at = ?
                WHERE session_id = ?
                """,
                (timestamp, session_id),
            )

            connection.commit()
            return SaveResult(
                created=True,
                turn=PersistedTurn(
                    turn_id=turn_id,
                    request_id=request_id,
                    session_id=session_id,
                    question=question,
                    answer=answer,
                    created_at=timestamp,
                ),
            )

        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def load_turns(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[PersistedTurn]:
        if limit <= 0:
            raise ValueError("limit 必须大于 0。")

        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT turn_id, request_id, session_id,
                       question, answer, created_at
                FROM (
                    SELECT turn_id, request_id, session_id,
                           question, answer, created_at
                    FROM chat_turns
                    WHERE session_id = ?
                    ORDER BY turn_id DESC
                    LIMIT ?
                )
                ORDER BY turn_id ASC
                """,
                (session_id, limit),
            ).fetchall()
        finally:
            connection.close()

        return [self._row_to_turn(row) for row in rows]

    def count_turns(self, session_id: str) -> int:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM chat_turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return int(row["count"])
        finally:
            connection.close()

    def journal_mode(self) -> str:
        connection = self._connect()
        try:
            return str(
                connection.execute("PRAGMA journal_mode").fetchone()[0]
            )
        finally:
            connection.close()

    @staticmethod
    def _row_to_turn(row: sqlite3.Row) -> PersistedTurn:
        return PersistedTurn(
            turn_id=row["turn_id"],
            request_id=row["request_id"],
            session_id=row["session_id"],
            question=row["question"],
            answer=row["answer"],
            created_at=row["created_at"],
        )
