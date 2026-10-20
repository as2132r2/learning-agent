"""Day28 第 3 步：跨异步任务传播 Trace 上下文。

运行：
    python day28/python/day28_async_context.py

本文件复用第 2 步已经实现的 Span、JSON Formatter 和 ContextVar。
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from day28_trace_spans import (
    JsonFormatter,
    Span,
    TraceFilter,
    bind_request_context,
    build_span_tree,
    current_request_id,
    current_span_id,
    current_trace_id,
    finished_spans,
    logger,
    reset_request_context,
)


DAY28_DIR = Path(__file__).resolve().parents[1]
LOG_PATH = DAY28_DIR / "logs" / "day28_async_context.jsonl"


def configure_step3_log_file() -> None:
    """把第 2 步 Logger 的文件输出切换到第 3 步独立文件。"""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    file_handler.addFilter(TraceFilter())
    logger.addHandler(file_handler)


def log_task_context(event: str, message: str) -> None:
    """把当前 Task 名称及其 Trace 上下文写入日志。"""

    task = asyncio.current_task()
    logger.info(
        message,
        extra={
            "event": event,
            "task_name": None if task is None else task.get_name(),
        },
    )


async def create_query_embedding(question: str) -> list[float]:
    """异步模拟 Embedding 网络调用。"""

    with Span(
        "create_query_embedding",
        kind="model",
        attributes={"model": "bge-small-zh", "text_length": len(question)},
    ) as span:
        log_task_context("embedding_context", "Embedding任务读取当前上下文。")
        await asyncio.sleep(0.06)
        vector = [0.12, 0.34, 0.56]
        span.set_attribute("vector_dimensions", len(vector))
        return vector


async def retrieve_documents(question: str) -> list[str]:
    """检索任务内部再创建一个 Embedding Task。"""

    with Span(
        "retrieve_documents",
        kind="retriever",
        attributes={"top_k": 2},
    ) as span:
        log_task_context("retrieval_context", "检索任务读取当前上下文。")

        # create_task() 会复制“创建这一刻”的 ContextVar 上下文。
        embedding_task = asyncio.create_task(
            create_query_embedding(question),
            name=f"embedding-{current_request_id.get()}",
        )
        vector = await embedding_task
        await asyncio.sleep(0.04)

        documents = [
            "RAG 的中文名称是检索增强生成。",
            "RAG 先检索外部资料，再结合资料生成回答。",
        ]
        span.set_attribute("query_vector_dimensions", len(vector))
        span.set_attribute("document_count", len(documents))
        return documents


async def check_request_policy(question: str) -> bool:
    """与检索并行执行的策略检查。"""

    with Span(
        "check_request_policy",
        kind="guardrail",
        attributes={"policy_version": "day28-v1"},
    ) as span:
        log_task_context("policy_context", "策略任务读取当前上下文。")
        await asyncio.sleep(0.035)
        allowed = bool(question.strip())
        span.set_attribute("allowed", allowed)
        return allowed


async def generate_answer(question: str, documents: list[str]) -> str:
    with Span(
        "generate_answer",
        kind="model",
        attributes={
            "model": "deepseek-chat-demo",
            "evidence_count": len(documents),
        },
    ) as span:
        log_task_context("generation_context", "回答任务读取当前上下文。")
        await asyncio.sleep(0.05)
        answer = f"关于“{question}”：RAG 会先检索资料，再结合资料生成回答。"
        span.set_attribute("answer_length", len(answer))
        return answer


async def handle_request(question: str, request_number: int) -> dict[str, Any]:
    """处理一条异步请求，并在 finally 中恢复本 Task 的上下文。"""

    request_id, trace_id, tokens = bind_request_context()
    try:
        with Span(
            "POST /ask",
            kind="server",
            attributes={
                "http.method": "POST",
                "http.route": "/ask",
                "request_number": request_number,
            },
        ) as root_span:
            log_task_context("request_context", "请求Task已绑定独立上下文。")

            # 两个 Task 都在根 Span 上下文中创建，因此都继承根 Span 为父亲。
            retrieval_task = asyncio.create_task(
                retrieve_documents(question),
                name=f"retrieval-request-{request_number}",
            )
            policy_task = asyncio.create_task(
                check_request_policy(question),
                name=f"policy-request-{request_number}",
            )

            documents, allowed = await asyncio.gather(
                retrieval_task,
                policy_task,
            )
            if not allowed:
                raise ValueError("问题没有通过策略检查。")

            answer = await generate_answer(question, documents)
            root_span.set_attribute("http.status_code", 200)
            return {
                "request_number": request_number,
                "request_id": request_id,
                "trace_id": trace_id,
                "root_span_id": root_span.span_id,
                "answer": answer,
            }
    finally:
        reset_request_context(tokens)


def validate_isolation(results: list[dict[str, Any]]) -> dict[str, Any]:
    """检查每个 Trace 的 Span 是否独立，没有错误父子关系。"""

    trace_ids = {result["trace_id"] for result in results}
    request_ids = {result["request_id"] for result in results}
    spans_by_trace = {
        trace_id: [span for span in finished_spans if span.trace_id == trace_id]
        for trace_id in trace_ids
    }

    wrong_trace_spans = [
        span.span_id
        for span in finished_spans
        if span.trace_id not in trace_ids
    ]

    return {
        "trace_count": len(trace_ids),
        "request_count": len(request_ids),
        "span_count_by_trace": {
            trace_id: len(spans) for trace_id, spans in spans_by_trace.items()
        },
        "each_trace_has_one_root": all(
            sum(span.parent_span_id is None for span in spans) == 1
            for spans in spans_by_trace.values()
        ),
        "all_parent_ids_stay_inside_trace": all(
            all(
                span.parent_span_id is None
                or span.parent_span_id in {item.span_id for item in spans}
                for span in spans
            )
            for spans in spans_by_trace.values()
        ),
        "unexpected_span_ids": wrong_trace_spans,
    }


async def main() -> None:
    configure_step3_log_file()
    LOG_PATH.write_text("", encoding="utf-8")
    finished_spans.clear()

    print("Day28 跨异步任务传播 Trace 上下文")
    print("日志文件：", LOG_PATH)
    print("=" * 100)

    # 在尚未绑定任何请求上下文时创建两个请求 Task。
    request_tasks = [
        asyncio.create_task(
            handle_request("什么是 RAG？", 1),
            name="http-request-1",
        ),
        asyncio.create_task(
            handle_request("RAG 为什么需要检索？", 2),
            name="http-request-2",
        ),
    ]

    print("创建的顶层 Task：")
    for task in request_tasks:
        print(f"- {task.get_name()} | 类型={type(task)}")

    results = await asyncio.gather(*request_tasks)

    print("\n" + "=" * 100)
    print("两条并发请求的结果：")
    for result in results:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 100)
    for result in results:
        print(
            f"请求 {result['request_number']} 调用树 "
            f"| trace_id={result['trace_id']}"
        )
        for line in build_span_tree(result["trace_id"]):
            print(line)
        print()

    report = validate_isolation(results)
    print("=" * 100)
    print("异步上下文隔离检查：")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n主协程当前上下文：")
    print("request_id：", current_request_id.get())
    print("trace_id：", current_trace_id.get())
    print("span_id：", current_span_id.get())

    print("\n" + "=" * 100)
    print("本步数据流：")
    print(
        "main创建2个请求Task\n"
        "├→ 请求Task 1绑定request_id-1、trace_id-1、root-span-1\n"
        "│  ├→ retrieval Task继承Trace 1和root-span-1\n"
        "│  └→ policy Task继承Trace 1和root-span-1\n"
        "└→ 请求Task 2绑定request_id-2、trace_id-2、root-span-2\n"
        "   ├→ retrieval Task继承Trace 2和root-span-2\n"
        "   └→ policy Task继承Trace 2和root-span-2\n"
        "每个Task修改自己的Context副本，不会覆盖其他并发Task"
    )


if __name__ == "__main__":
    asyncio.run(main())
