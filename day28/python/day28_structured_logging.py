"""Day28 第 1 步：结构化 JSON 日志与关联 ID。

运行：
    python day28/python/day28_structured_logging.py

本程序会同时把日志输出到终端，并写入：
    day28/logs/day28_app.jsonl
"""

from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DAY28_DIR = Path(__file__).resolve().parents[1]
LOG_PATH = DAY28_DIR / "logs" / "day28_app.jsonl"
SERVICE_NAME = "day28-rag-service"

# ContextVar 保存“当前执行上下文”的关联 ID。
# 后续普通函数只需要记录日志，不必层层手工传递这些 ID。
current_request_id: ContextVar[str] = ContextVar(
    "current_request_id", default="-"
)
current_trace_id: ContextVar[str] = ContextVar(
    "current_trace_id", default="-"
)


@dataclass(frozen=True)
class RequestContext:
    """一次请求的关联上下文。"""

    request_id: str
    trace_id: str


@dataclass(frozen=True)
class ContextTokens:
    """保存 ContextVar.set() 返回的令牌，用于恢复旧上下文。"""

    request_token: Token[str]
    trace_token: Token[str]


class CorrelationFilter(logging.Filter):
    """在每条 LogRecord 上补充当前 request_id 和 trace_id。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id.get()
        record.trace_id = current_trace_id.get()
        record.service = SERVICE_NAME
        return True


class JsonFormatter(logging.Formatter):
    """把 logging.LogRecord 转成每行一个 JSON 对象。"""

    # logging 自带的字段不重复放入 details。
    RESERVED_FIELDS = set(logging.makeLogRecord({}).__dict__) | {
        "message",
        "asctime",
        "request_id",
        "trace_id",
        "service",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "service": record.service,
            "logger": record.name,
            "event": getattr(record, "event", "application_log"),
            "message": record.getMessage(),
            "request_id": record.request_id,
            "trace_id": record.trace_id,
        }

        details = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self.RESERVED_FIELDS
            and key != "event"
            and not key.startswith("_")
        }
        if details:
            payload["details"] = details

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def create_logger() -> logging.Logger:
    """创建同时写终端和 JSONL 文件的 Logger。"""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("day28")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = JsonFormatter()
    correlation_filter = CorrelationFilter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(correlation_filter)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(correlation_filter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


logger = create_logger()


def new_request_context() -> RequestContext:
    """为一条新请求创建独立 request_id 和 trace_id。"""

    return RequestContext(
        request_id=f"req-{uuid.uuid4().hex[:12]}",
        trace_id=f"trace-{uuid.uuid4().hex[:16]}",
    )


def bind_context(context: RequestContext) -> ContextTokens:
    """把关联 ID 绑定到当前执行上下文。"""

    return ContextTokens(
        request_token=current_request_id.set(context.request_id),
        trace_token=current_trace_id.set(context.trace_id),
    )


def reset_context(tokens: ContextTokens) -> None:
    """恢复绑定前的上下文，防止污染下一条请求。"""

    current_trace_id.reset(tokens.trace_token)
    current_request_id.reset(tokens.request_token)


def retrieve_documents(question: str) -> list[str]:
    """模拟知识库检索。"""

    logger.info(
        "开始检索知识库。",
        extra={
            "event": "retrieval_started",
            "question_length": len(question),
            "top_k": 2,
        },
    )
    documents = [
        "RAG 的中文名称是检索增强生成。",
        "RAG 会先检索资料，再让模型根据资料生成回答。",
    ]
    logger.info(
        "知识库检索完成。",
        extra={
            "event": "retrieval_completed",
            "document_count": len(documents),
        },
    )
    return documents


def generate_answer(question: str, documents: list[str]) -> str:
    """模拟根据问题和资料生成答案。"""

    logger.info(
        "开始生成回答。",
        extra={
            "event": "generation_started",
            "question": question,
            "evidence_count": len(documents),
            "model": "deepseek-chat-demo",
        },
    )
    answer = "RAG 是检索增强生成：先检索外部资料，再结合资料生成回答。"
    logger.info(
        "回答生成完成。",
        extra={
            "event": "generation_completed",
            "answer_length": len(answer),
        },
    )
    return answer


def handle_request(question: str) -> dict[str, str]:
    """处理一次请求，并保证上下文最终被清理。"""

    context = new_request_context()
    tokens = bind_context(context)

    try:
        logger.info(
            "收到用户问题。",
            extra={
                "event": "request_started",
                "method": "POST",
                "path": "/ask",
            },
        )
        documents = retrieve_documents(question)
        answer = generate_answer(question, documents)
        logger.info(
            "请求处理成功。",
            extra={
                "event": "request_completed",
                "status_code": 200,
            },
        )
        return {
            "request_id": context.request_id,
            "trace_id": context.trace_id,
            "answer": answer,
        }
    except Exception:
        logger.exception(
            "请求处理失败。",
            extra={"event": "request_failed", "status_code": 500},
        )
        raise
    finally:
        reset_context(tokens)


def read_logs_for_trace(trace_id: str) -> list[dict[str, Any]]:
    """从 JSONL 日志中筛选属于指定 trace_id 的所有记录。"""

    matched: list[dict[str, Any]] = []
    with LOG_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record.get("trace_id") == trace_id:
                matched.append(record)
    return matched


def main() -> None:
    # 每次演示前清空旧日志，输出更容易观察。
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    print("Day28 结构化 JSON 日志与关联 ID")
    print("日志文件：", LOG_PATH)
    print("=" * 88)

    first = handle_request("什么是 RAG？")
    second = handle_request("RAG 的工作流程是什么？")

    print("\n" + "=" * 88)
    print("两次请求的关联 ID：")
    print("请求 1：", first["request_id"], first["trace_id"])
    print("请求 2：", second["request_id"], second["trace_id"])
    print("两个 trace_id 不同，说明两次请求的日志不会混在同一条链路中。")

    matched = read_logs_for_trace(first["trace_id"])
    print("\n" + "=" * 88)
    print("只查询第 1 次请求的日志：")
    for index, record in enumerate(matched, start=1):
        print(
            f"{index}. event={record['event']} | "
            f"request_id={record['request_id']} | "
            f"message={record['message']}"
        )

    print("\n匹配日志数量：", len(matched))
    print("上下文清理后的 request_id：", current_request_id.get())
    print("上下文清理后的 trace_id：", current_trace_id.get())

    print("\n" + "=" * 88)
    print("本步数据流：")
    print(
        "新请求 → 生成 request_id + trace_id → ContextVar绑定\n"
        "→ 普通业务函数写 logging.LogRecord\n"
        "→ CorrelationFilter自动补充关联ID\n"
        "→ JsonFormatter转成单行JSON\n"
        "→ 终端 + JSONL文件\n"
        "→ 按trace_id查询整条请求日志\n"
        "→ finally恢复ContextVar，避免污染下一请求"
    )


if __name__ == "__main__":
    main()
