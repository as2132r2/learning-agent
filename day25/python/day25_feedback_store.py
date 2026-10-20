"""Day25 第 1 步：用户反馈数据模型与 SQLite 持久化。

运行：
    python day25/python/day25_feedback_store.py

可重复运行。程序使用 UPSERT，避免同一用户对同一回答产生重复反馈。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4


# 当前 Python 文件位于 day25/python，因此 parents[1] 是 day25 目录。
DAY25_DIR = Path(__file__).resolve().parents[1]
DATABASE_DIR = DAY25_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "day25_feedback.db"

# Literal 说明 rating 只允许这三个字符串值。
Rating = Literal["up", "down", "neutral"]


@dataclass(slots=True)
class FeedbackInput:
    """保存反馈时需要传入的数据。"""

    request_id: str
    session_id: str
    user_id: str
    question: str
    answer: str
    rating: Rating
    reason: str | None = None
    answer_version: str = "day25-v1"
    source_ids: list[str] | None = None
    pipeline_metrics: dict[str, object] | None = None


@dataclass(slots=True)
class FeedbackRecord:
    """从数据库读取出的完整反馈记录。"""

    feedback_id: str
    request_id: str
    session_id: str
    user_id: str
    question: str
    answer: str
    rating: str
    reason: str | None
    answer_version: str
    source_ids: list[str]
    pipeline_metrics: dict[str, object]
    created_at: str
    updated_at: str


def open_database() -> sqlite3.Connection:
    """打开数据库，并把查询结果设置成可以按列名读取的 sqlite3.Row。"""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    """创建反馈表和常用查询索引。"""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS answer_feedback (
            feedback_id      TEXT PRIMARY KEY,
            request_id       TEXT NOT NULL,
            session_id       TEXT NOT NULL,
            user_id          TEXT NOT NULL,
            question         TEXT NOT NULL,
            answer           TEXT NOT NULL,
            rating           TEXT NOT NULL
                             CHECK (rating IN ('up', 'down', 'neutral')),
            reason           TEXT,
            answer_version   TEXT NOT NULL,
            source_ids_json  TEXT NOT NULL DEFAULT '[]',
            metrics_json     TEXT NOT NULL DEFAULT '{}',
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            UNIQUE (request_id, user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_rating
        ON answer_feedback (rating);

        CREATE INDEX IF NOT EXISTS idx_feedback_version
        ON answer_feedback (answer_version);
        """
    )
    connection.commit()


def validate_feedback(feedback: FeedbackInput) -> None:
    """在写数据库前执行确定性校验。"""

    if feedback.rating not in {"up", "down", "neutral"}:
        raise ValueError("rating 必须是 up、down 或 neutral。")
    if not feedback.request_id.strip():
        raise ValueError("request_id 不能为空。")
    if not feedback.user_id.strip():
        raise ValueError("user_id 不能为空。")
    if not feedback.question.strip() or not feedback.answer.strip():
        raise ValueError("question 和 answer 不能为空。")


def save_feedback(
    connection: sqlite3.Connection,
    feedback: FeedbackInput,
) -> tuple[str, bool]:
    """新增或更新反馈，返回 (feedback_id, 是否为新增记录)。"""

    validate_feedback(feedback)

    old_row = connection.execute(
        """
        SELECT feedback_id
        FROM answer_feedback
        WHERE request_id = ? AND user_id = ?
        """,
        (feedback.request_id, feedback.user_id),
    ).fetchone()

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    feedback_id = old_row["feedback_id"] if old_row else f"fb-{uuid4().hex[:12]}"
    created = old_row is None

    # ON CONFLICT 实现幂等更新：同一用户可以把点赞改成点踩，记录不会重复。
    connection.execute(
        """
        INSERT INTO answer_feedback (
            feedback_id, request_id, session_id, user_id,
            question, answer, rating, reason, answer_version,
            source_ids_json, metrics_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(request_id, user_id) DO UPDATE SET
            session_id = excluded.session_id,
            question = excluded.question,
            answer = excluded.answer,
            rating = excluded.rating,
            reason = excluded.reason,
            answer_version = excluded.answer_version,
            source_ids_json = excluded.source_ids_json,
            metrics_json = excluded.metrics_json,
            updated_at = excluded.updated_at
        """,
        (
            feedback_id,
            feedback.request_id,
            feedback.session_id,
            feedback.user_id,
            feedback.question,
            feedback.answer,
            feedback.rating,
            feedback.reason,
            feedback.answer_version,
            json.dumps(feedback.source_ids or [], ensure_ascii=False),
            json.dumps(feedback.pipeline_metrics or {}, ensure_ascii=False),
            now,
            now,
        ),
    )
    connection.commit()
    return feedback_id, created


def row_to_record(row: sqlite3.Row) -> FeedbackRecord:
    """把 SQLite 行转换为带类型的 Python 对象。"""

    return FeedbackRecord(
        feedback_id=row["feedback_id"],
        request_id=row["request_id"],
        session_id=row["session_id"],
        user_id=row["user_id"],
        question=row["question"],
        answer=row["answer"],
        rating=row["rating"],
        reason=row["reason"],
        answer_version=row["answer_version"],
        source_ids=json.loads(row["source_ids_json"]),
        pipeline_metrics=json.loads(row["metrics_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def list_feedback(connection: sqlite3.Connection) -> list[FeedbackRecord]:
    """按最近更新时间倒序读取所有反馈。"""

    rows = connection.execute(
        "SELECT * FROM answer_feedback ORDER BY updated_at DESC"
    ).fetchall()
    return [row_to_record(row) for row in rows]


def rating_statistics(connection: sqlite3.Connection) -> dict[str, int]:
    """统计各类评价数量。"""

    statistics = {"up": 0, "down": 0, "neutral": 0}
    rows = connection.execute(
        """
        SELECT rating, COUNT(*) AS amount
        FROM answer_feedback
        GROUP BY rating
        """
    ).fetchall()
    for row in rows:
        statistics[row["rating"]] = row["amount"]
    return statistics


def main() -> None:
    print("Day25 用户反馈数据模型与 SQLite 持久化")
    print("数据库：", DATABASE_PATH)
    print("=" * 80)

    with open_database() as connection:
        create_schema(connection)

        # 模拟 Day24 RAG 流水线完成一次回答后，浏览器提交了一条反馈。
        sample = FeedbackInput(
            request_id="req-day25-demo-001",
            session_id="xiaolin-day25",
            user_id="user-xiaolin",
            question="什么是 RAG？",
            answer=(
                "RAG 是检索增强生成：先检索外部资料，"
                "再把资料和问题交给大语言模型。"
            ),
            rating="up",
            reason="回答简洁，而且说明了基本流程。",
            answer_version="day24-final-v1",
            source_ids=["chunk-13bb58d59e5e6030cd2c"],
            pipeline_metrics={
                "risk": "low",
                "model_calls": 2,
                "total_tokens": 2358,
                "cache_hits": 0,
            },
        )

        feedback_id, created = save_feedback(connection, sample)
        print("保存结果：", "新增" if created else "更新已有记录")
        print("feedback_id：", feedback_id)

        print("\n" + "=" * 80)
        print("数据库中的反馈：")
        records = list_feedback(connection)
        for index, record in enumerate(records, start=1):
            print(f"\n第 {index} 条")
            print(json.dumps(asdict(record), ensure_ascii=False, indent=2))

        statistics = rating_statistics(connection)
        print("\n" + "=" * 80)
        print("评价统计：")
        print(json.dumps(statistics, ensure_ascii=False, indent=2))

    print("\n本步数据流：")
    print("RAG回答 + request_id + 用户评价")
    print("→ FeedbackInput → validate_feedback()")
    print("→ save_feedback() → SQLite UPSERT")
    print("→ sqlite3.Row → FeedbackRecord → 统计或后续评测集")
    print("\n再次运行程序时，同一 request_id + user_id 会更新，不会重复新增。")


if __name__ == "__main__":
    main()
