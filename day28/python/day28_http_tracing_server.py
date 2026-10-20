"""Day28 第 4 步：带请求链路追踪的 aiohttp 服务端。

运行服务端：
    python day28/python/day28_http_tracing_server.py

然后在另一个终端运行：
    python day28/python/day28_http_tracing_client.py
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from aiohttp import web

from day28_trace_spans import (
    JsonFormatter,
    RequestTokens,
    Span,
    TraceFilter,
    current_request_id,
    current_trace_id,
    logger,
    new_id,
    reset_request_context,
)


HOST = "127.0.0.1"
PORT = 8041
DAY28_DIR = Path(__file__).resolve().parents[1]
LOG_PATH = DAY28_DIR / "logs" / "day28_http_tracing.jsonl"

REQUEST_ID_HEADER = "X-Request-ID"
TRACE_ID_HEADER = "X-Trace-ID"
SPAN_ID_HEADER = "X-Span-ID"

# 只接受简单、安全、长度有限的外部 ID。
VALID_ID = re.compile(r"^[A-Za-z0-9._:-]{1,80}$")


def configure_http_log_file() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    file_handler.addFilter(TraceFilter())
    logger.addHandler(file_handler)


def accepted_or_generated(
    raw_value: str | None,
    prefix: str,
    length: int,
) -> tuple[str, bool]:
    """校验外部关联 ID；无效时生成新值。"""

    if raw_value is not None and VALID_ID.fullmatch(raw_value):
        return raw_value, True
    return new_id(prefix, length), False


def bind_http_context(request: web.Request) -> tuple[str, str, RequestTokens]:
    request_id, accepted_request_id = accepted_or_generated(
        request.headers.get(REQUEST_ID_HEADER), "req", 12
    )
    trace_id, accepted_trace_id = accepted_or_generated(
        request.headers.get(TRACE_ID_HEADER), "trace", 16
    )

    tokens = RequestTokens(
        request_token=current_request_id.set(request_id),
        trace_token=current_trace_id.set(trace_id),
    )
    logger.info(
        "HTTP关联上下文已经建立。",
        extra={
            "event": "http_context_bound",
            "accepted_client_request_id": accepted_request_id,
            "accepted_client_trace_id": accepted_trace_id,
        },
    )
    return request_id, trace_id, tokens


def json_error(message: str, status: int) -> web.Response:
    return web.json_response(
        {
            "ok": False,
            "error": message,
            "request_id": current_request_id.get(),
            "trace_id": current_trace_id.get(),
        },
        status=status,
    )


@web.middleware
async def tracing_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """为每个 HTTP 请求创建根 Span，并在响应头返回关联 ID。"""

    request_id, trace_id, tokens = bind_http_context(request)
    response: web.StreamResponse

    try:
        with Span(
            f"{request.method} {request.path}",
            kind="server",
            attributes={
                "http.method": request.method,
                "http.route": request.path,
                "http.client": request.remote or "unknown",
            },
        ) as root_span:
            try:
                response = await handler(request)
            except web.HTTPException as error:
                response = json_error(error.text or error.reason, error.status)
            except Exception as error:
                logger.exception(
                    "HTTP请求处理失败。",
                    extra={
                        "event": "http_request_failed",
                        "error_type": type(error).__name__,
                    },
                )
                response = json_error("服务器内部错误。", 500)

            root_span.set_attribute("http.status_code", response.status)
            if response.status >= 400:
                root_span.set_attribute("http.error", True)

            response.headers[REQUEST_ID_HEADER] = request_id
            response.headers[TRACE_ID_HEADER] = trace_id
            response.headers[SPAN_ID_HEADER] = root_span.span_id
            return response
    finally:
        reset_request_context(tokens)


async def retrieve_documents(question: str) -> list[str]:
    with Span(
        "retrieve_documents",
        kind="retriever",
        attributes={"top_k": 2, "question_length": len(question)},
    ) as span:
        await asyncio.sleep(0.08)
        documents = [
            "RAG 的中文名称是检索增强生成。",
            "RAG 先检索外部资料，再把资料和问题交给模型。",
        ]
        span.set_attribute("document_count", len(documents))
        return documents


async def generate_answer(question: str, documents: list[str]) -> str:
    with Span(
        "generate_answer",
        kind="model",
        attributes={
            "model": "deepseek-chat-demo",
            "evidence_count": len(documents),
        },
    ) as span:
        await asyncio.sleep(0.07)
        answer = f"关于“{question}”：RAG 会先检索外部资料，再结合资料生成回答。"
        span.set_attribute("answer_length", len(answer))
        return answer


async def health_handler(request: web.Request) -> web.Response:
    del request
    return web.json_response(
        {
            "status": "ok",
            "service": "day28-http-tracing-server",
            "request_id": current_request_id.get(),
            "trace_id": current_trace_id.get(),
        }
    )


async def ask_handler(request: web.Request) -> web.Response:
    try:
        body: Any = await request.json()
    except Exception as error:
        raise web.HTTPBadRequest(text="请求正文必须是 JSON。") from error

    question = body.get("question") if isinstance(body, dict) else None
    if not isinstance(question, str) or not question.strip():
        raise web.HTTPBadRequest(text="question 必须是非空字符串。")

    question = question.strip()
    documents = await retrieve_documents(question)
    answer = await generate_answer(question, documents)
    logger.info(
        "HTTP问答处理成功。",
        extra={
            "event": "ask_completed",
            "question_length": len(question),
            "answer_length": len(answer),
        },
    )
    return web.json_response(
        {
            "ok": True,
            "request_id": current_request_id.get(),
            "trace_id": current_trace_id.get(),
            "question": question,
            "answer": answer,
            "source_count": len(documents),
        }
    )


def create_app() -> web.Application:
    app = web.Application(middlewares=[tracing_middleware])
    app.router.add_get("/health", health_handler)
    app.router.add_post("/ask", ask_handler)
    return app


def main() -> None:
    configure_http_log_file()
    LOG_PATH.write_text("", encoding="utf-8")
    print("Day28 HTTP Trace 服务已启动。")
    print(f"服务地址：http://{HOST}:{PORT}")
    print("健康检查：GET /health")
    print("模型问答：POST /ask")
    print("日志文件：", LOG_PATH)
    print("请在另一个终端运行：")
    print("python day28/python/day28_http_tracing_client.py")
    print("按 Ctrl+C 停止服务。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
