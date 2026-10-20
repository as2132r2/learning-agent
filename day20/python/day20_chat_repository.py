"""Day20 第二步：SQLite 聊天数据 Repository。

Repository 的职责是集中管理 SQL，让 WebSocket、HTTP 或命令行程序
只调用 Python 方法，不直接关心表结构和 SQL 细节。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_DATABASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "day20_chat.db"
)


def local_now() -> str:
    """返回带时区的 ISO 8601 时间字符串。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class ChatSession:
    session_id: str
    client_id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ChatMessage:
    message_id: int
    session_id: str
    role: str
    content: str
    created_at: str


class ChatRepository:
    """会话和消息的 SQLite 数据访问层。"""

    def __init__(self, database_path: Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """为一次数据库操作建立独立连接。"""

        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        """创建数据库表和索引。"""

        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL
                        CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id)
                        REFERENCES chat_sessions(session_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_client_updated
                ON chat_sessions(client_id, updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_messages_session_message
                ON chat_messages(session_id, message_id);
                """
            )
            connection.commit()
        finally:
            connection.close()

    def create_session(
        self,
        session_id: str,
        client_id: str,
        title: str,
    ) -> ChatSession:
        """创建一个新的聊天会话并返回领域对象。"""

        timestamp = local_now()
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    INSERT INTO chat_sessions (
                        session_id,
                        client_id,
                        title,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, client_id, title, timestamp, timestamp),
                )
        finally:
            connection.close()

        return ChatSession(
            session_id=session_id,
            client_id=client_id,
            title=title,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def get_session(self, session_id: str) -> ChatSession | None:
        """按 session_id 查询会话；不存在时返回 None。"""

        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT session_id, client_id, title, created_at, updated_at
                FROM chat_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        finally:
            connection.close()

        return self._row_to_session(row) if row is not None else None

    def list_sessions(self, client_id: str) -> list[ChatSession]:
        """列出一个客户端的全部会话，最近更新的排在前面。"""

        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT session_id, client_id, title, created_at, updated_at
                FROM chat_sessions
                WHERE client_id = ?
                ORDER BY updated_at DESC, session_id DESC
                """,
                (client_id,),
            ).fetchall()
        finally:
            connection.close()

        return [self._row_to_session(row) for row in rows]

    def rename_session(self, session_id: str, new_title: str) -> bool:
        """修改会话标题；返回是否找到并修改了会话。"""

        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE chat_sessions
                    SET title = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (new_title, local_now(), session_id),
                )
                changed = cursor.rowcount > 0
        finally:
            connection.close()

        return changed

    def append_turn(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
    ) -> tuple[ChatMessage, ChatMessage]:
        """用一个事务写入一轮完整问答。"""

        timestamp = local_now()
        connection = self._connect()
        try:
            with connection:
                user_cursor = connection.execute(
                    """
                    INSERT INTO chat_messages (
                        session_id, role, content, created_at
                    )
                    VALUES (?, 'user', ?, ?)
                    """,
                    (session_id, user_content, timestamp),
                )

                assistant_cursor = connection.execute(
                    """
                    INSERT INTO chat_messages (
                        session_id, role, content, created_at
                    )
                    VALUES (?, 'assistant', ?, ?)
                    """,
                    (session_id, assistant_content, timestamp),
                )

                connection.execute(
                    """
                    UPDATE chat_sessions
                    SET updated_at = ?
                    WHERE session_id = ?
                    """,
                    (timestamp, session_id),
                )

                if connection.total_changes < 3:
                    raise ValueError(f"会话不存在：{session_id}")

                user_message_id = int(user_cursor.lastrowid)
                assistant_message_id = int(assistant_cursor.lastrowid)
        finally:
            connection.close()

        return (
            ChatMessage(
                message_id=user_message_id,
                session_id=session_id,
                role="user",
                content=user_content,
                created_at=timestamp,
            ),
            ChatMessage(
                message_id=assistant_message_id,
                session_id=session_id,
                role="assistant",
                content=assistant_content,
                created_at=timestamp,
            ),
        )

    def load_messages(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[ChatMessage]:
        """读取最近 limit 条消息，并恢复成从旧到新的顺序。"""

        if limit <= 0:
            raise ValueError("limit 必须大于 0。")

        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT message_id, session_id, role, content, created_at
                FROM (
                    SELECT message_id, session_id, role, content, created_at
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY message_id DESC
                    LIMIT ?
                )
                ORDER BY message_id ASC
                """,
                (session_id, limit),
            ).fetchall()
        finally:
            connection.close()

        return [self._row_to_message(row) for row in rows]

    def count_messages(self, session_id: str) -> int:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT COUNT(*) AS message_count
                FROM chat_messages
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        finally:
            connection.close()

        return int(row["message_count"])

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> ChatSession:
        return ChatSession(
            session_id=row["session_id"],
            client_id=row["client_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> ChatMessage:
        return ChatMessage(
            message_id=row["message_id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
        )
