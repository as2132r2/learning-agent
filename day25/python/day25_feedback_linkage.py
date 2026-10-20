"""Day25 第 3 步：把反馈关联到回答、引用来源和流水线指标。

运行：
    python day25/python/day25_feedback_linkage.py

本步使用规范化关系：
    answer_runs 1 ── N answer_sources
    answer_runs 1 ── N linked_feedback
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4


DAY25_DIR = Path(__file__).resolve().parents[1]
DATABASE_DIR = DAY25_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "day25_feedback.db"

Rating = Literal["up", "down", "neutral"]


@dataclass(slots=True)
class SourceTrace:
    """回答实际引用的一条证据。"""

    chunk_id: str
    source_path: str
    citation_number: int
    reranker_score: int
    excerpt: str


@dataclass(slots=True)
class AnswerRun:
    """一次已经完成的 RAG 回答及其运行信息。"""

    request_id: str
    session_id: str
    user_id: str
    question: str
    answer: str
    answer_version: str
    risk_level: str
    model_name: str
    model_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cache_hits: int
    quality_passed: bool
    sources: list[SourceTrace]


@dataclass(slots=True)
class LinkedFeedback:
    """用户评价；只保存反馈自身数据，通过 request_id 关联回答。"""

    request_id: str
    user_id: str
    rating: Rating
    reason: str | None = None


def open_database() -> sqlite3.Connection:
    """打开 SQLite，并启用外键和按列名访问。"""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    """创建回答、来源和关联反馈三张表。"""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS answer_runs (
            request_id       TEXT PRIMARY KEY,
            session_id       TEXT NOT NULL,
            user_id          TEXT NOT NULL,
            question         TEXT NOT NULL,
            answer           TEXT NOT NULL,
            answer_version   TEXT NOT NULL,
            risk_level       TEXT NOT NULL
                             CHECK (risk_level IN ('low', 'medium', 'high')),
            model_name       TEXT NOT NULL,
            model_calls      INTEGER NOT NULL CHECK (model_calls >= 0),
            input_tokens     INTEGER NOT NULL CHECK (input_tokens >= 0),
            output_tokens    INTEGER NOT NULL CHECK (output_tokens >= 0),
            latency_ms       INTEGER NOT NULL CHECK (latency_ms >= 0),
            cache_hits       INTEGER NOT NULL CHECK (cache_hits >= 0),
            quality_passed   INTEGER NOT NULL CHECK (quality_passed IN (0, 1)),
            created_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS answer_sources (
            request_id       TEXT NOT NULL,
            chunk_id         TEXT NOT NULL,
            source_path      TEXT NOT NULL,
            citation_number  INTEGER NOT NULL CHECK (citation_number > 0),
            reranker_score   INTEGER NOT NULL
                             CHECK (reranker_score BETWEEN 0 AND 100),
            excerpt          TEXT NOT NULL,
            PRIMARY KEY (request_id, citation_number),
            FOREIGN KEY (request_id)
                REFERENCES answer_runs(request_id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS linked_feedback (
            feedback_id      TEXT PRIMARY KEY,
            request_id       TEXT NOT NULL,
            user_id          TEXT NOT NULL,
            rating           TEXT NOT NULL
                             CHECK (rating IN ('up', 'down', 'neutral')),
            reason           TEXT,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            UNIQUE (request_id, user_id),
            FOREIGN KEY (request_id)
                REFERENCES answer_runs(request_id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_answer_runs_version
        ON answer_runs(answer_version);

        CREATE INDEX IF NOT EXISTS idx_linked_feedback_rating
        ON linked_feedback(rating);
        """
    )
    connection.commit()


def validate_answer_run(run: AnswerRun) -> None:
    """在写入前检查跨字段规则。"""

    if not run.request_id.strip():
        raise ValueError("request_id 不能为空。")
    if run.risk_level not in {"low", "medium", "high"}:
        raise ValueError("risk_level 必须是 low、medium 或 high。")
    if min(
        run.model_calls,
        run.input_tokens,
        run.output_tokens,
        run.latency_ms,
        run.cache_hits,
    ) < 0:
        raise ValueError("调用数、Token、耗时和缓存命中数不能为负数。")
    citation_numbers = [source.citation_number for source in run.sources]
    if len(citation_numbers) != len(set(citation_numbers)):
        raise ValueError("同一次回答中的 citation_number 不能重复。")


def save_answer_run(connection: sqlite3.Connection, run: AnswerRun) -> None:
    """在同一个事务中保存回答和它的全部引用。"""

    validate_answer_run(run)
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    # with connection 表示一个事务：回答和来源要么一起成功，要么一起回滚。
    with connection:
        connection.execute(
            """
            INSERT INTO answer_runs (
                request_id, session_id, user_id, question, answer,
                answer_version, risk_level, model_name, model_calls,
                input_tokens, output_tokens, latency_ms, cache_hits,
                quality_passed, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                session_id = excluded.session_id,
                user_id = excluded.user_id,
                question = excluded.question,
                answer = excluded.answer,
                answer_version = excluded.answer_version,
                risk_level = excluded.risk_level,
                model_name = excluded.model_name,
                model_calls = excluded.model_calls,
                input_tokens = excluded.input_tokens,
                output_tokens = excluded.output_tokens,
                latency_ms = excluded.latency_ms,
                cache_hits = excluded.cache_hits,
                quality_passed = excluded.quality_passed
            """,
            (
                run.request_id,
                run.session_id,
                run.user_id,
                run.question,
                run.answer,
                run.answer_version,
                run.risk_level,
                run.model_name,
                run.model_calls,
                run.input_tokens,
                run.output_tokens,
                run.latency_ms,
                run.cache_hits,
                int(run.quality_passed),
                now,
            ),
        )

        # 重放同一个 request_id 时，用本次完整来源集合替换旧集合。
        connection.execute(
            "DELETE FROM answer_sources WHERE request_id = ?",
            (run.request_id,),
        )
        connection.executemany(
            """
            INSERT INTO answer_sources (
                request_id, chunk_id, source_path,
                citation_number, reranker_score, excerpt
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run.request_id,
                    source.chunk_id,
                    source.source_path,
                    source.citation_number,
                    source.reranker_score,
                    source.excerpt,
                )
                for source in run.sources
            ],
        )


def save_linked_feedback(
    connection: sqlite3.Connection,
    feedback: LinkedFeedback,
) -> tuple[str, bool]:
    """保存反馈；不存在对应回答时，外键会拒绝写入。"""

    if feedback.rating not in {"up", "down", "neutral"}:
        raise ValueError("rating 必须是 up、down 或 neutral。")

    old = connection.execute(
        """
        SELECT feedback_id
        FROM linked_feedback
        WHERE request_id = ? AND user_id = ?
        """,
        (feedback.request_id, feedback.user_id),
    ).fetchone()

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    feedback_id = old["feedback_id"] if old else f"lfb-{uuid4().hex[:12]}"

    with connection:
        connection.execute(
            """
            INSERT INTO linked_feedback (
                feedback_id, request_id, user_id,
                rating, reason, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id, user_id) DO UPDATE SET
                rating = excluded.rating,
                reason = excluded.reason,
                updated_at = excluded.updated_at
            """,
            (
                feedback_id,
                feedback.request_id,
                feedback.user_id,
                feedback.rating,
                feedback.reason,
                now,
                now,
            ),
        )
    return feedback_id, old is None


def load_complete_trace(
    connection: sqlite3.Connection,
    request_id: str,
) -> dict[str, Any] | None:
    """通过 JOIN 读取回答、来源、指标和反馈的完整追踪信息。"""

    answer_row = connection.execute(
        "SELECT * FROM answer_runs WHERE request_id = ?",
        (request_id,),
    ).fetchone()
    if answer_row is None:
        return None

    source_rows = connection.execute(
        """
        SELECT chunk_id, source_path, citation_number, reranker_score, excerpt
        FROM answer_sources
        WHERE request_id = ?
        ORDER BY citation_number
        """,
        (request_id,),
    ).fetchall()
    feedback_rows = connection.execute(
        """
        SELECT feedback_id, user_id, rating, reason, created_at, updated_at
        FROM linked_feedback
        WHERE request_id = ?
        ORDER BY updated_at DESC
        """,
        (request_id,),
    ).fetchall()

    return {
        "request_id": answer_row["request_id"],
        "session_id": answer_row["session_id"],
        "question": answer_row["question"],
        "answer": answer_row["answer"],
        "answer_version": answer_row["answer_version"],
        "quality_passed": bool(answer_row["quality_passed"]),
        "metrics": {
            "risk_level": answer_row["risk_level"],
            "model_name": answer_row["model_name"],
            "model_calls": answer_row["model_calls"],
            "input_tokens": answer_row["input_tokens"],
            "output_tokens": answer_row["output_tokens"],
            "total_tokens": (
                answer_row["input_tokens"] + answer_row["output_tokens"]
            ),
            "latency_ms": answer_row["latency_ms"],
            "cache_hits": answer_row["cache_hits"],
        },
        "sources": [dict(row) for row in source_rows],
        "feedback": [dict(row) for row in feedback_rows],
    }


def main() -> None:
    print("Day25 回答—引用—指标—反馈关联练习")
    print("数据库：", DATABASE_PATH)
    print("=" * 88)

    request_id = "req-day25-linked-001"
    answer_run = AnswerRun(
        request_id=request_id,
        session_id="xiaolin-day25",
        user_id="user-xiaolin",
        question="为什么 RAG 回答需要提供引用？",
        answer=(
            "引用让用户能够核对回答依据，也让系统可以审计每条结论是否被证据支持。"
            "[资料1][资料2]"
        ),
        answer_version="day25-linked-v1",
        risk_level="medium",
        model_name="deepseek-chat",
        model_calls=3,
        input_tokens=2410,
        output_tokens=436,
        latency_ms=3280,
        cache_hits=1,
        quality_passed=True,
        sources=[
            SourceTrace(
                chunk_id="chunk-rag-citation-001",
                source_path="day21/documents/day21_sample.txt",
                citation_number=1,
                reranker_score=98,
                excerpt="RAG 回答应保留 source、title、page 等来源信息。",
            ),
            SourceTrace(
                chunk_id="chunk-rag-audit-002",
                source_path="day23/notes/LangChain-Day23-学习笔记.md",
                citation_number=2,
                reranker_score=96,
                excerpt="引用存在不等于证据支持，需要继续检查 groundedness。",
            ),
        ],
    )
    feedback = LinkedFeedback(
        request_id=request_id,
        user_id="user-xiaolin",
        rating="up",
        reason="能够看到引用，而且回答速度可以接受。",
    )

    with open_database() as connection:
        create_schema(connection)
        save_answer_run(connection, answer_run)
        feedback_id, created = save_linked_feedback(connection, feedback)
        complete_trace = load_complete_trace(connection, request_id)

    print("回答保存完成：", request_id)
    print("来源数量：", len(answer_run.sources))
    print("反馈操作：", "新增" if created else "更新")
    print("反馈编号：", feedback_id)

    print("\n" + "=" * 88)
    print("通过 request_id 还原出的完整追踪记录：")
    print(json.dumps(complete_trace, ensure_ascii=False, indent=2))

    print("\n" + "=" * 88)
    print("本步数据流：")
    print("RAG流水线结果 → AnswerRun → answer_runs")
    print("引用证据 list[SourceTrace] → answer_sources")
    print("浏览器评价 → LinkedFeedback → linked_feedback")
    print("三张表通过 request_id 关联 → load_complete_trace()")
    print("→ 可分析版本、引用、质量、成本、延迟与用户满意度之间的关系")


if __name__ == "__main__":
    main()
