"""Day20 第一步：使用 SQLite 持久化浏览器会话和聊天消息。

运行：
    python day20/python/day20_sqlite_basic.py

本步骤不调用大模型，只学习：建表、事务写入、关闭连接、重新连接读取。
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


DAY20_DIR = Path(__file__).resolve().parents[1]
DATABASE_DIR = DAY20_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "day20_chat.db"


def utc_now() -> str:
    """返回适合写入 SQLite 的 UTC ISO 8601 时间字符串。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def open_database() -> sqlite3.Connection:
    """建立数据库连接，并启用外键和字典式行访问。"""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_tables(connection: sqlite3.Connection) -> None:
    """创建会话表和消息表；重复执行不会重复创建。"""

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
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id)
                REFERENCES chat_sessions(session_id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
        ON chat_messages(session_id, message_id);
        """
    )
    connection.commit()


def insert_demo_conversation(
    connection: sqlite3.Connection,
) -> tuple[str, str]:
    """用一个事务写入一条会话和两条消息。"""

    client_id = "browser-" + uuid.uuid4().hex[:8]
    session_id = "session-" + uuid.uuid4().hex[:12]
    timestamp = utc_now()

    # with connection：成功时 commit，发生异常时 rollback。
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
            (
                session_id,
                client_id,
                "Day20 SQLite 入门",
                timestamp,
                timestamp,
            ),
        )

        connection.executemany(
            """
            INSERT INTO chat_messages (
                session_id,
                role,
                content,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    session_id,
                    "user",
                    "我叫小林，正在学习 SQLite 持久化。",
                    timestamp,
                ),
                (
                    session_id,
                    "assistant",
                    "你好小林，这两条消息将被保存到磁盘数据库。",
                    timestamp,
                ),
            ],
        )

        connection.execute(
            """
            UPDATE chat_sessions
            SET updated_at = ?
            WHERE session_id = ?
            """,
            (timestamp, session_id),
        )

    return client_id, session_id


def read_conversation(
    connection: sqlite3.Connection,
    session_id: str,
) -> tuple[sqlite3.Row | None, list[sqlite3.Row]]:
    """读取一条会话及其全部消息。"""

    session = connection.execute(
        """
        SELECT session_id, client_id, title, created_at, updated_at
        FROM chat_sessions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()

    messages = connection.execute(
        """
        SELECT message_id, session_id, role, content, created_at
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY message_id ASC
        """,
        (session_id,),
    ).fetchall()

    return session, messages


def count_database_rows(
    connection: sqlite3.Connection,
) -> tuple[int, int]:
    """返回整个数据库中的会话数和消息数。"""

    session_count = connection.execute(
        "SELECT COUNT(*) FROM chat_sessions"
    ).fetchone()[0]
    message_count = connection.execute(
        "SELECT COUNT(*) FROM chat_messages"
    ).fetchone()[0]
    return int(session_count), int(message_count)


def main() -> None:
    print("Day20 SQLite 持久化基础练习")
    print("数据库：", DATABASE_PATH)
    print("=" * 70)

    first_connection = open_database()
    print("第一次连接类型：", type(first_connection))

    create_tables(first_connection)
    client_id, session_id = insert_demo_conversation(first_connection)

    print("已写入演示数据：")
    print("client_id：", client_id)
    print("session_id：", session_id)

    first_connection.close()
    print("\n第一次数据库连接已经关闭。")
    print("内存中的 Connection 对象不能再查询，但数据仍保存在磁盘。")

    print("\n" + "=" * 70)
    print("重新打开数据库，验证持久化：")

    second_connection = open_database()
    session, messages = read_conversation(second_connection, session_id)

    if session is None:
        raise RuntimeError("没有读到刚才保存的会话。")

    print("\n会话记录：")
    print(dict(session))
    print("会话行类型：", type(session))

    print("\n消息记录：")
    for index, message in enumerate(messages, start=1):
        print(f"第 {index} 条：", dict(message))

    session_count, message_count = count_database_rows(second_connection)
    print("\n数据库累计统计：")
    print("会话数量：", session_count)
    print("消息数量：", message_count)

    print("\n表结构：")
    table_rows = second_connection.execute(
        """
        SELECT name, type
        FROM sqlite_master
        WHERE type IN ('table', 'index')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY type DESC, name ASC
        """
    ).fetchall()
    for row in table_rows:
        print(f"- {row['type']}：{row['name']}")

    second_connection.close()

    print("\n" + "=" * 70)
    print("本步数据流：")
    print("Python tuple → 参数化 SQL → SQLite 事务 → day20_chat.db")
    print("关闭 Connection → 重新连接 → sqlite3.Row → dict → 终端显示")


if __name__ == "__main__":
    main()
