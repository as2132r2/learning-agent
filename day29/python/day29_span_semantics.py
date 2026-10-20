"""Day29 第 2 步：标准 Span Kind、状态与语义属性。

运行：
    python day29/python/day29_span_semantics.py

本练习实现一个轻量教学模型，不依赖 OpenTelemetry SDK，但遵循其核心概念。
HTTP 属性使用当前稳定命名；RAG 业务字段使用本课程自己的 rag.* 命名空间。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from day29_traceparent_basic import TraceContext, create_child_context, create_root_context


DAY29_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = DAY29_DIR / "logs" / "day29_semantic_spans.jsonl"


class SpanKind(StrEnum):
    """OpenTelemetry 定义的五种 Span Kind。"""

    INTERNAL = "INTERNAL"
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


class StatusCode(StrEnum):
    """Span 最终状态；正常 HTTP Span 通常保持 UNSET。"""

    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


class SemanticValidationError(ValueError):
    """Span Kind 或语义属性组合不符合本练习规则。"""


@dataclass
class SemanticSpan:
    """一个具有标准 Kind、Status 和语义属性的教学 Span。"""

    name: str
    kind: SpanKind
    context: TraceContext
    parent_span_id: str | None
    started_at: str
    ended_at: str = ""
    duration_ms: float = 0.0
    status_code: StatusCode = StatusCode.UNSET
    status_description: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "name": name,
                "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "attributes": dict(attributes or {}),
            }
        )

    def set_status(
        self,
        code: StatusCode,
        description: str | None = None,
    ) -> None:
        self.status_code = code
        self.status_description = description

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["status_code"] = self.status_code.value
        data["trace_id"] = self.context.trace_id
        data["span_id"] = self.context.span_id
        data["sampled"] = self.context.sampled
        del data["context"]
        return data


class SpanBuilder:
    """负责创建、计时、校验并结束一个 SemanticSpan。"""

    def __init__(
        self,
        name: str,
        kind: SpanKind,
        parent_context: TraceContext,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self._started_counter = 0.0
        self.span = SemanticSpan(
            name=name,
            kind=kind,
            context=create_child_context(parent_context),
            parent_span_id=parent_context.span_id,
            started_at="",
            attributes=dict(attributes or {}),
        )

    def __enter__(self) -> SemanticSpan:
        self._started_counter = time.perf_counter()
        self.span.started_at = datetime.now(UTC).isoformat(timespec="milliseconds")
        return self.span

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.span.ended_at = datetime.now(UTC).isoformat(timespec="milliseconds")
        self.span.duration_ms = round(
            (time.perf_counter() - self._started_counter) * 1000,
            3,
        )
        if exc is not None:
            self.span.set_status(StatusCode.ERROR, str(exc))
            self.span.set_attribute("error.type", type(exc).__name__)
            self.span.add_event(
                "exception",
                {
                    "exception.type": type(exc).__name__,
                    "exception.message": str(exc),
                },
            )
        validate_span(self.span)
        finished_spans.append(self.span)
        return False


finished_spans: list[SemanticSpan] = []


def require_attributes(span: SemanticSpan, names: set[str]) -> None:
    missing = sorted(name for name in names if name not in span.attributes)
    if missing:
        raise SemanticValidationError(
            f"{span.name}({span.kind}) 缺少属性：{', '.join(missing)}"
        )


def validate_span(span: SemanticSpan) -> None:
    """验证本练习使用到的稳定 HTTP 属性和自定义 RAG 约定。"""

    if not span.name.strip():
        raise SemanticValidationError("Span名称不能为空。")

    if span.kind is SpanKind.SERVER and "http.request.method" in span.attributes:
        require_attributes(
            span,
            {"http.request.method", "url.scheme", "url.path"},
        )

    if span.kind is SpanKind.CLIENT and "http.request.method" in span.attributes:
        require_attributes(
            span,
            {"http.request.method", "server.address", "url.full"},
        )

    if span.kind is SpanKind.CLIENT and "db.system.name" in span.attributes:
        require_attributes(span, {"db.system.name", "db.operation.name"})

    if span.kind in {SpanKind.PRODUCER, SpanKind.CONSUMER}:
        require_attributes(
            span,
            {"messaging.system", "messaging.destination.name"},
        )

    if span.status_code is StatusCode.ERROR and "error.type" not in span.attributes:
        raise SemanticValidationError(
            f"错误Span {span.name} 必须包含低基数error.type。"
        )


def apply_http_status(span: SemanticSpan, status_code: int) -> None:
    """按 HTTP Span 的基本语义设置状态和属性。"""

    span.set_attribute("http.response.status_code", status_code)

    # 对本练习：5xx 对 SERVER/CLIENT 都视为错误；
    # 4xx 对 CLIENT 视为错误，而 SERVER 保持 UNSET。
    is_error = status_code >= 500 or (
        span.kind is SpanKind.CLIENT and 400 <= status_code < 500
    )
    if is_error:
        span.set_status(StatusCode.ERROR)
        span.set_attribute("error.type", str(status_code))


def simulate_trace() -> list[SemanticSpan]:
    """构造包含五种 Span Kind 的一条教学调用链。"""

    finished_spans.clear()
    root_context = create_root_context(sampled=True)

    # 1. SERVER：处理外部同步请求。这里显式保持根 Span 打开，
    # 直到所有同步子操作完成后再结束它。
    server_builder = SpanBuilder(
        "POST /ask",
        SpanKind.SERVER,
        root_context,
        {
            "http.request.method": "POST",
            "http.route": "/ask",
            "url.scheme": "http",
            "url.path": "/ask",
            "server.address": "127.0.0.1",
            "server.port": 8043,
        },
    )
    server_span = server_builder.__enter__()
    server_span.add_event("request.validated", {"request.valid": True})
    time.sleep(0.008)

    # 后续 Span 在同一 Trace 中，以 SERVER Span 为父亲。
    server_context = server_span.context

    # 2. INTERNAL：当前进程内部 RAG 编排，不跨网络。
    with SpanBuilder(
        "rag.retrieve",
        SpanKind.INTERNAL,
        server_context,
        {
            "rag.operation.name": "retrieve",
            "rag.retrieval.mode": "hybrid",
            "rag.top_k": 5,
        },
    ) as retrieval_span:
        time.sleep(0.012)
        retrieval_span.set_attribute("rag.document.count", 3)

    # 3. CLIENT：当前服务同步等待远程模型 API。
    with SpanBuilder(
        "POST model.example/v1/chat/completions",
        SpanKind.CLIENT,
        server_context,
        {
            "http.request.method": "POST",
            "server.address": "model.example",
            "server.port": 443,
            "url.full": "https://model.example/v1/chat/completions",
            # GenAI语义仍在演进，教学中先使用明确的rag.*业务属性。
            "rag.model.name": "deepseek-chat-demo",
        },
    ) as model_client_span:
        time.sleep(0.016)
        model_client_span.set_attribute("rag.input_tokens", 320)
        model_client_span.set_attribute("rag.output_tokens", 86)
        apply_http_status(model_client_span, 200)

    # 额外数据库 CLIENT Span：展示远程数据库调用的 Kind。
    with SpanBuilder(
        "SELECT knowledge_chunks",
        SpanKind.CLIENT,
        retrieval_span.context,
        {
            "db.system.name": "sqlite",
            "db.operation.name": "SELECT",
            "db.namespace": "day29_knowledge",
            # 不记录完整SQL和用户问题，降低敏感数据与高基数风险。
        },
    ) as database_span:
        time.sleep(0.006)
        database_span.set_attribute("db.response.returned_rows", 3)

    # 4. PRODUCER：投递异步审计任务，生产者不会等待消费者完成。
    with SpanBuilder(
        "audit-events publish",
        SpanKind.PRODUCER,
        server_context,
        {
            "messaging.system": "demo_queue",
            "messaging.destination.name": "audit-events",
            "messaging.operation.name": "publish",
        },
    ) as producer_span:
        time.sleep(0.004)
        producer_span.set_attribute("messaging.batch.message_count", 1)

    # 5. CONSUMER：稍后由后台 Worker 处理异步消息。
    with SpanBuilder(
        "audit-events process",
        SpanKind.CONSUMER,
        producer_span.context,
        {
            "messaging.system": "demo_queue",
            "messaging.destination.name": "audit-events",
            "messaging.operation.name": "process",
        },
    ) as consumer_span:
        time.sleep(0.005)
        consumer_span.add_event("audit.persisted", {"record.count": 1})

    apply_http_status(server_span, 200)
    server_builder.__exit__(None, None, None)
    return list(finished_spans)


def demonstrate_error_status(parent_context: TraceContext) -> SemanticSpan:
    """演示 CLIENT 收到 503 时的标准错误表示。"""

    with SpanBuilder(
        "POST model.example/v1/chat/completions",
        SpanKind.CLIENT,
        parent_context,
        {
            "http.request.method": "POST",
            "server.address": "model.example",
            "url.full": "https://model.example/v1/chat/completions",
        },
    ) as failed_span:
        time.sleep(0.003)
        apply_http_status(failed_span, 503)
        failed_span.add_event("retry.scheduled", {"retry.delay_ms": 500})
    return failed_span


def write_jsonl(spans: list[SemanticSpan]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for span in spans:
            file.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    spans = simulate_trace()
    error_span = demonstrate_error_status(spans[0].context)
    all_spans = [*spans, error_span]
    write_jsonl(all_spans)

    print("Day29 标准 Span Kind、状态与语义属性")
    print("输出文件：", OUTPUT_PATH)
    print("=" * 108)

    print("五种 Span Kind 示例：")
    for span in spans:
        print(
            f"- {span.kind.value:<8} | {span.name:<42} | "
            f"status={span.status_code.value:<5} | {span.duration_ms:.3f}ms"
        )

    print("\n" + "=" * 108)
    print("HTTP成功与失败状态：")
    http_success = next(
        span for span in spans if span.name.startswith("POST model.example")
    )
    print("HTTP 200 CLIENT Span：")
    print("status_code：", http_success.status_code.value)
    print("http.response.status_code：", http_success.attributes["http.response.status_code"])
    print("error.type：", http_success.attributes.get("error.type"))
    print()
    print("HTTP 503 CLIENT Span：")
    print("status_code：", error_span.status_code.value)
    print("http.response.status_code：", error_span.attributes["http.response.status_code"])
    print("error.type：", error_span.attributes["error.type"])

    print("\n" + "=" * 108)
    print("完整语义 Span JSON 示例：")
    print(json.dumps(error_span.to_dict(), ensure_ascii=False, indent=2))

    print("\n" + "=" * 108)
    print("Kind选择规则：")
    print("SERVER   ：接收远程同步请求，对方等待结果。")
    print("CLIENT   ：调用远程同步服务，当前方等待结果。")
    print("PRODUCER ：发布或调度异步工作，不等待消费者完成。")
    print("CONSUMER ：处理先前由生产者发布的异步工作。")
    print("INTERNAL ：进程内部操作，不表示远程通信边界。")

    print("\n本步数据流：")
    print(
        "操作类型 → 选择SpanKind\n"
        "→ 使用低基数Span名称\n"
        "→ 写入对应语义属性\n"
        "→ 正常情况Status保持UNSET\n"
        "→ 错误时Status=ERROR + error.type\n"
        "→ validate_span检查必要字段\n"
        "→ 单行JSON持久化"
    )


if __name__ == "__main__":
    main()
