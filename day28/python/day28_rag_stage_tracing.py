"""Day28 第 5 步：RAG 各阶段耗时、状态与错误追踪。

运行：
    python day28/python/day28_rag_stage_tracing.py

本步使用确定性函数模拟 RAG 阶段，重点学习可观测性，不消耗模型额度。
Trace 和 Span 会持久化到 SQLite。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

from day28_trace_spans import (
    RequestTokens,
    Span,
    SpanRecord,
    current_request_id,
    current_trace_id,
    new_id,
    reset_request_context,
)


DAY28_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = DAY28_DIR / "database" / "day28_observability.db"

T = TypeVar("T")


def utc_now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def open_database() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def create_schema() -> None:
    with open_database() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                operation_name TEXT NOT NULL,
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                duration_ms REAL,
                error_type TEXT,
                error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS spans (
                span_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                parent_span_id TEXT,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                status TEXT NOT NULL,
                attributes_json TEXT NOT NULL,
                error_type TEXT,
                error_message TEXT,
                FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
            );

            CREATE INDEX IF NOT EXISTS idx_spans_trace
            ON spans(trace_id, started_at);

            CREATE INDEX IF NOT EXISTS idx_spans_status
            ON spans(status, name);
            """
        )


def start_trace(
    trace_id: str,
    request_id: str,
    question: str,
    started_at: str,
) -> None:
    with open_database() as connection:
        connection.execute(
            """
            INSERT INTO traces (
                trace_id, request_id, operation_name, question,
                status, started_at
            ) VALUES (?, ?, 'rag_pipeline', ?, 'running', ?)
            """,
            (trace_id, request_id, question, started_at),
        )


def finish_trace(
    trace_id: str,
    status: str,
    started_counter: float,
    error: Exception | None,
) -> None:
    duration_ms = round((time.perf_counter() - started_counter) * 1000, 3)
    with open_database() as connection:
        connection.execute(
            """
            UPDATE traces
            SET status = ?, ended_at = ?, duration_ms = ?,
                error_type = ?, error_message = ?
            WHERE trace_id = ?
            """,
            (
                status,
                utc_now_text(),
                duration_ms,
                None if error is None else type(error).__name__,
                None if error is None else str(error),
                trace_id,
            ),
        )


def save_span(record: SpanRecord) -> None:
    """把已结束 Span 的完整事实写入数据库。"""

    with open_database() as connection:
        connection.execute(
            """
            INSERT INTO spans (
                span_id, trace_id, parent_span_id, name, kind,
                started_at, ended_at, duration_ms, status,
                attributes_json, error_type, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.span_id,
                record.trace_id,
                record.parent_span_id,
                record.name,
                record.kind,
                record.started_at,
                record.ended_at,
                record.duration_ms,
                record.status,
                json.dumps(record.attributes, ensure_ascii=False, default=str),
                record.error_type,
                record.error_message,
            ),
        )


class PersistentSpan(Span):
    """Span 结束后自动写入 SQLite。"""

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        suppress = super().__exit__(exc_type, exc, traceback)
        if self.record is None:
            raise RuntimeError("Span结束后没有生成SpanRecord。")
        save_span(self.record)
        return suppress


def bind_pipeline_context() -> tuple[str, str, RequestTokens]:
    request_id = new_id("req", 12)
    trace_id = new_id("trace", 16)
    tokens = RequestTokens(
        request_token=current_request_id.set(request_id),
        trace_token=current_trace_id.set(trace_id),
    )
    return request_id, trace_id, tokens


def hybrid_retrieve(question: str) -> list[dict[str, Any]]:
    """模拟向量检索和关键词检索融合。"""

    time.sleep(0.045)
    return [
        {
            "chunk_id": "chunk-rag-001",
            "text": "RAG 的中文名称是检索增强生成。",
            "rrf_score": 0.0325,
        },
        {
            "chunk_id": "chunk-rag-002",
            "text": "RAG 先检索外部资料，再把资料和问题交给大语言模型。",
            "rrf_score": 0.0318,
        },
        {
            "chunk_id": "chunk-other-003",
            "text": "LangGraph 使用图结构组织工作流。",
            "rrf_score": 0.0159,
        },
    ]


def rerank_candidates(
    question: str,
    candidates: list[dict[str, Any]],
    simulate_error: bool,
) -> list[dict[str, Any]]:
    """模拟 LLM Reranker；可主动制造一次故障。"""

    time.sleep(0.075)
    if simulate_error:
        raise TimeoutError("模拟Reranker上游请求超时。")

    keyword = "rag" in question.lower()
    reranked: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        item["relevance_score"] = (
            95 if keyword and "RAG" in item["text"] else 8
        )
        reranked.append(item)
    return sorted(
        reranked,
        key=lambda item: item["relevance_score"],
        reverse=True,
    )


def generate_answer(
    question: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    time.sleep(0.055)
    return {
        "answer": "RAG 是检索增强生成：先检索外部资料，再结合资料生成回答。",
        "question": question,
        "citations": [item["chunk_id"] for item in evidence],
        "output_tokens": 38,
    }


def evaluate_quality(
    answer: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    time.sleep(0.032)
    return {
        "passed": bool(answer["answer"] and evidence),
        "groundedness": 100,
        "relevance": 98,
    }


def observed_stage(
    name: str,
    kind: str,
    attributes: dict[str, Any],
    function: Callable[..., T],
    *args: Any,
) -> T:
    """统一包装一个 RAG 阶段，避免每个函数重复追踪样板代码。"""

    with PersistentSpan(name, kind=kind, attributes=attributes) as span:
        result = function(*args)
        if isinstance(result, list):
            span.set_attribute("output_count", len(result))
        elif isinstance(result, dict):
            span.set_attribute("output_fields", sorted(result.keys()))
        return result


def run_rag_pipeline(
    question: str,
    *,
    simulate_reranker_error: bool = False,
) -> dict[str, Any]:
    """运行一次被完整追踪的 RAG 流水线。"""

    request_id, trace_id, tokens = bind_pipeline_context()
    started_at = utc_now_text()
    started_counter = time.perf_counter()
    start_trace(trace_id, request_id, question, started_at)

    status = "success"
    caught_error: Exception | None = None
    answer: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None

    try:
        with PersistentSpan(
            "rag_pipeline",
            kind="workflow",
            attributes={
                "question_length": len(question),
                "simulate_reranker_error": simulate_reranker_error,
            },
        ) as root_span:
            candidates = observed_stage(
                "hybrid_retrieval",
                "retriever",
                {"top_k": 5, "retrieval_mode": "vector+fts+rrf"},
                hybrid_retrieve,
                question,
            )
            reranked = observed_stage(
                "llm_reranker",
                "model",
                {
                    "model": "deepseek-chat-demo",
                    "candidate_count": len(candidates),
                },
                rerank_candidates,
                question,
                candidates,
                simulate_reranker_error,
            )
            evidence = [
                item for item in reranked if item["relevance_score"] >= 90
            ][:2]
            root_span.set_attribute("qualified_evidence_count", len(evidence))

            answer = observed_stage(
                "structured_answer_generation",
                "model",
                {
                    "model": "deepseek-chat-demo",
                    "evidence_count": len(evidence),
                },
                generate_answer,
                question,
                evidence,
            )
            quality = observed_stage(
                "answer_quality_evaluation",
                "evaluator",
                {
                    "claim_count": len(answer["citations"]),
                    "evidence_count": len(evidence),
                },
                evaluate_quality,
                answer,
                evidence,
            )
            root_span.set_attribute("quality_passed", quality["passed"])
    except Exception as error:
        status = "failed"
        caught_error = error
    finally:
        finish_trace(trace_id, status, started_counter, caught_error)
        reset_request_context(tokens)

    return {
        "request_id": request_id,
        "trace_id": trace_id,
        "status": status,
        "answer": None if answer is None else answer["answer"],
        "quality": quality,
        "error_type": (
            None if caught_error is None else type(caught_error).__name__
        ),
        "error_message": None if caught_error is None else str(caught_error),
    }


def load_trace(trace_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open_database() as connection:
        trace_row = connection.execute(
            "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        span_rows = connection.execute(
            """
            SELECT * FROM spans
            WHERE trace_id = ?
            ORDER BY started_at, rowid
            """,
            (trace_id,),
        ).fetchall()

    if trace_row is None:
        raise LookupError(f"没有找到Trace：{trace_id}")
    trace = dict(trace_row)
    spans: list[dict[str, Any]] = []
    for row in span_rows:
        item = dict(row)
        item["attributes"] = json.loads(item.pop("attributes_json"))
        spans.append(item)
    return trace, spans


def render_database_tree(spans: list[dict[str, Any]]) -> list[str]:
    children: dict[str | None, list[dict[str, Any]]] = {}
    for span in spans:
        children.setdefault(span["parent_span_id"], []).append(span)

    lines: list[str] = []

    def visit(parent_id: str | None, depth: int) -> None:
        for span in children.get(parent_id, []):
            error = (
                ""
                if span["error_type"] is None
                else f" error={span['error_type']}"
            )
            lines.append(
                f"{'  ' * depth}├─ {span['name']} [{span['kind']}] "
                f"{span['duration_ms']:.3f}ms status={span['status']}{error}"
            )
            visit(span["span_id"], depth + 1)

    visit(None, 0)
    return lines


def stage_diagnostics(spans: list[dict[str, Any]]) -> dict[str, Any]:
    child_spans = [span for span in spans if span["parent_span_id"] is not None]
    failed = [span for span in child_spans if span["status"] == "error"]
    slowest = max(child_spans, key=lambda span: span["duration_ms"], default=None)
    return {
        "slowest_stage": None if slowest is None else slowest["name"],
        "slowest_duration_ms": (
            None if slowest is None else slowest["duration_ms"]
        ),
        "failed_stage_count": len(failed),
        "failed_stages": [
            {
                "name": span["name"],
                "error_type": span["error_type"],
                "error_message": span["error_message"],
            }
            for span in failed
        ],
    }


def print_trace_report(result: dict[str, Any]) -> None:
    trace, spans = load_trace(result["trace_id"])
    print("\n" + "=" * 104)
    print("Trace：", trace["trace_id"])
    print("request_id：", trace["request_id"])
    print("状态：", trace["status"])
    print("总耗时：", trace["duration_ms"], "ms")
    if trace["error_type"]:
        print("Trace错误：", trace["error_type"], trace["error_message"])
    print("调用树：")
    for line in render_database_tree(spans):
        print(line)
    print("阶段诊断：")
    print(json.dumps(stage_diagnostics(spans), ensure_ascii=False, indent=2))


def main() -> None:
    create_schema()

    print("Day28 RAG 各阶段耗时、状态与错误追踪")
    print("数据库：", DATABASE_PATH)
    print("说明：使用确定性阶段模拟，不调用真实模型。")

    successful = run_rag_pipeline("什么是 RAG？")
    failed = run_rag_pipeline(
        "RAG 的工作流程是什么？",
        simulate_reranker_error=True,
    )

    print_trace_report(successful)
    print_trace_report(failed)

    print("\n" + "=" * 104)
    print("两个请求结果：")
    print(json.dumps([successful, failed], ensure_ascii=False, indent=2))

    print("\n本步数据流：")
    print(
        "问题 → Trace(rag_pipeline)\n"
        "→ Span(hybrid_retrieval)\n"
        "→ Span(llm_reranker)\n"
        "→ Span(structured_answer_generation)\n"
        "→ Span(answer_quality_evaluation)\n"
        "→ 每个Span结束时写入SQLite\n"
        "→ 按trace_id重建调用树\n"
        "→ 找出最慢阶段和准确失败位置"
    )


if __name__ == "__main__":
    main()
