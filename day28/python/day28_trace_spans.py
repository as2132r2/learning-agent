"""Day28 第 2 步：Trace、Span 与父子调用关系。

运行：
    python day28/python/day28_trace_spans.py

输出日志：
    day28/logs/day28_trace_spans.jsonl
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DAY28_DIR = Path(__file__).resolve().parents[1]
LOG_PATH = DAY28_DIR / "logs" / "day28_trace_spans.jsonl"
SERVICE_NAME = "day28-rag-service"

current_request_id: ContextVar[str] = ContextVar(
    "current_request_id", default="-"
)
current_trace_id: ContextVar[str] = ContextVar(
    "current_trace_id", default="-"
)
current_span_id: ContextVar[str] = ContextVar(
    "current_span_id", default="-"
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str, length: int = 16) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:length]}"


@dataclass(frozen=True)
class RequestTokens:
    """恢复 request_id 和 trace_id 时需要的 ContextVar Token。"""

    request_token: Token[str]
    trace_token: Token[str]


@dataclass
class SpanRecord:
    """一个已经结束、可以分析的 Span。"""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str
    started_at: str
    ended_at: str = ""
    duration_ms: float = 0.0
    status: str = "running"
    attributes: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


# 教学中先保存在内存，下一步会关注异步任务中的上下文传播。
finished_spans: list[SpanRecord] = []


class TraceFilter(logging.Filter):
    """自动把当前 Trace/Span 上下文注入每条日志。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.service = SERVICE_NAME
        record.request_id = current_request_id.get()
        record.trace_id = current_trace_id.get()
        record.span_id = current_span_id.get()
        return True


class JsonFormatter(logging.Formatter):
    """把 LogRecord 格式化成单行 JSON。"""

    RESERVED = set(logging.makeLogRecord({}).__dict__) | {
        "message",
        "asctime",
        "service",
        "request_id",
        "trace_id",
        "span_id",
        "event",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": utc_now().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "service": record.service,
            "event": getattr(record, "event", "application_log"),
            "message": record.getMessage(),
            "request_id": record.request_id,
            "trace_id": record.trace_id,
            "span_id": record.span_id,
        }
        details = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self.RESERVED and not key.startswith("_")
        }
        if details:
            payload["details"] = details
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def create_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("day28.trace")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = JsonFormatter()
    trace_filter = TraceFilter()
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        handler.addFilter(trace_filter)
        logger.addHandler(handler)
    return logger


logger = create_logger()


class Span:
    """同步上下文管理器：进入时开始 Span，退出时结束 Span。"""

    def __init__(
        self,
        name: str,
        *,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.attributes = dict(attributes or {})
        self.span_id = new_id("span")
        parent = current_span_id.get()
        self.parent_span_id = None if parent == "-" else parent
        self._span_token: Token[str] | None = None
        self._start_counter = 0.0
        self.record: SpanRecord | None = None

    def __enter__(self) -> Span:
        self._start_counter = time.perf_counter()
        self._span_token = current_span_id.set(self.span_id)
        self.record = SpanRecord(
            trace_id=current_trace_id.get(),
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            kind=self.kind,
            started_at=utc_now().isoformat(timespec="milliseconds"),
            attributes=self.attributes,
        )
        logger.info(
            f"Span开始：{self.name}",
            extra={
                "event": "span_started",
                "span_name": self.name,
                "span_kind": self.kind,
                "parent_span_id": self.parent_span_id,
                **self.attributes,
            },
        )
        return self

    def set_attribute(self, key: str, value: Any) -> None:
        """在运行过程中给 Span 增加结果属性。"""

        if self.record is None:
            raise RuntimeError("Span尚未开始。")
        self.record.attributes[key] = value

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.record is None or self._span_token is None:
            raise RuntimeError("Span没有正确进入上下文。")

        self.record.ended_at = utc_now().isoformat(timespec="milliseconds")
        self.record.duration_ms = round(
            (time.perf_counter() - self._start_counter) * 1000,
            3,
        )

        if exc is None:
            self.record.status = "ok"
            logger.info(
                f"Span结束：{self.name}",
                extra={
                    "event": "span_completed",
                    "span_name": self.name,
                    "span_kind": self.kind,
                    "parent_span_id": self.parent_span_id,
                    "duration_ms": self.record.duration_ms,
                    "status": self.record.status,
                    **self.record.attributes,
                },
            )
        else:
            self.record.status = "error"
            self.record.error_type = type(exc).__name__
            self.record.error_message = str(exc)
            logger.error(
                f"Span失败：{self.name}",
                extra={
                    "event": "span_failed",
                    "span_name": self.name,
                    "span_kind": self.kind,
                    "parent_span_id": self.parent_span_id,
                    "duration_ms": self.record.duration_ms,
                    "status": self.record.status,
                    "error_type": self.record.error_type,
                    "error_message": self.record.error_message,
                },
            )

        finished_spans.append(self.record)
        current_span_id.reset(self._span_token)

        # False 表示不吞掉业务异常，交给外层继续处理。
        return False


def bind_request_context() -> tuple[str, str, RequestTokens]:
    request_id = new_id("req", 12)
    trace_id = new_id("trace")
    tokens = RequestTokens(
        request_token=current_request_id.set(request_id),
        trace_token=current_trace_id.set(trace_id),
    )
    return request_id, trace_id, tokens


def reset_request_context(tokens: RequestTokens) -> None:
    current_trace_id.reset(tokens.trace_token)
    current_request_id.reset(tokens.request_token)


def create_query_embedding(question: str) -> list[float]:
    """模拟检索内部的子操作。"""

    with Span(
        "create_query_embedding",
        kind="model",
        attributes={"model": "bge-small-zh", "text_length": len(question)},
    ) as span:
        time.sleep(0.025)
        vector = [0.12, 0.34, 0.56]
        span.set_attribute("vector_dimensions", len(vector))
        return vector


def search_database(question: str) -> list[str]:
    """检索 Span 内部再包含 Embedding 子 Span。"""

    with Span(
        "retrieve_documents",
        kind="retriever",
        attributes={"top_k": 2},
    ) as span:
        vector = create_query_embedding(question)
        time.sleep(0.018)
        documents = [
            "RAG 的中文名称是检索增强生成。",
            "RAG 先检索资料，再把资料和问题交给模型。",
        ]
        span.set_attribute("query_vector_dimensions", len(vector))
        span.set_attribute("document_count", len(documents))
        return documents


def call_model(question: str, documents: list[str]) -> str:
    with Span(
        "generate_answer",
        kind="model",
        attributes={
            "model": "deepseek-chat-demo",
            "evidence_count": len(documents),
        },
    ) as span:
        time.sleep(0.035)
        answer = "RAG 是检索增强生成，它先检索资料，再结合资料生成回答。"
        span.set_attribute("answer_length", len(answer))
        span.set_attribute("question_length", len(question))
        return answer


def handle_request(question: str) -> dict[str, str]:
    request_id, trace_id, tokens = bind_request_context()
    try:
        with Span(
            "POST /ask",
            kind="server",
            attributes={"http.method": "POST", "http.route": "/ask"},
        ) as root_span:
            documents = search_database(question)
            answer = call_model(question, documents)
            root_span.set_attribute("http.status_code", 200)
            return {
                "request_id": request_id,
                "trace_id": trace_id,
                "root_span_id": root_span.span_id,
                "answer": answer,
            }
    finally:
        reset_request_context(tokens)


def build_span_tree(trace_id: str) -> list[str]:
    """根据 parent_span_id 递归生成可读调用树。"""

    trace_spans = [span for span in finished_spans if span.trace_id == trace_id]
    children: dict[str | None, list[SpanRecord]] = {}
    for span in trace_spans:
        children.setdefault(span.parent_span_id, []).append(span)

    lines: list[str] = []

    def visit(parent_id: str | None, depth: int) -> None:
        for span in children.get(parent_id, []):
            lines.append(
                f"{'  ' * depth}├─ {span.name} "
                f"[{span.kind}] {span.duration_ms:.3f}ms "
                f"status={span.status} span_id={span.span_id}"
            )
            visit(span.span_id, depth + 1)

    visit(None, 0)
    return lines


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    finished_spans.clear()

    print("Day28 Trace、Span 与父子调用关系")
    print("日志文件：", LOG_PATH)
    print("=" * 96)

    result = handle_request("什么是 RAG？")

    print("\n请求结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 96)
    print("Span 调用树：")
    for line in build_span_tree(result["trace_id"]):
        print(line)

    trace_spans = [
        span for span in finished_spans if span.trace_id == result["trace_id"]
    ]
    root = next(span for span in trace_spans if span.parent_span_id is None)

    print("\n" + "=" * 96)
    print("关键观察：")
    print("Trace 数量：1")
    print("Span 数量：", len(trace_spans))
    print("根 Span：", root.name, root.span_id)
    print("根 Span 总耗时：", root.duration_ms, "ms")
    print(
        "所有 Span 的 trace_id 是否一致：",
        all(span.trace_id == result["trace_id"] for span in trace_spans),
    )
    print("请求结束后的 span_id：", current_span_id.get())

    print("\n" + "=" * 96)
    print("本步数据流：")
    print(
        "用户请求 → 创建一个trace_id\n"
        "→ 根Span：POST /ask\n"
        "   ├→ 检索Span：retrieve_documents\n"
        "   │  └→ 子Span：create_query_embedding\n"
        "   └→ 模型Span：generate_answer\n"
        "→ 每个Span记录span_id、parent_span_id、耗时、状态和属性\n"
        "→ parent_span_id还原整棵调用树"
    )


if __name__ == "__main__":
    main()
