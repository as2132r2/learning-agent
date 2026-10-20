"""Day17 第三步（服务端）：通过 SSE 推送模型流。

运行方式：
    python day17/python/day17_sse_server.py

服务地址：
    http://127.0.0.1:8005

接口：
    GET /health
    GET /stream?question=什么是SSE

SSE（Server-Sent Events）特点：
    1. 基于普通 HTTP；
    2. 服务端通过一个持续响应不断向客户端推送事件；
    3. Content-Type 必须是 text/event-stream；
    4. 每个事件以一个空行结束。
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any

from aiohttp import web
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()


HOST = "127.0.0.1"
PORT = 8005


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


def encode_sse(
    *,
    event: str,
    data: dict[str, Any],
    event_id: str | None = None,
) -> bytes:
    """把 Python 数据编码成一个完整 SSE 事件。

    SSE 文本格式示例：
        id: request-001:1
        event: token
        data: {"text": "你好"}

    最后的两个换行符非常重要，它们表示一个事件结束。
    """

    lines: list[str] = []

    if event_id is not None:
        lines.append(f"id: {event_id}")

    lines.append(f"event: {event}")
    lines.append(
        "data: " + json.dumps(data, ensure_ascii=False)
    )

    return ("\n".join(lines) + "\n\n").encode("utf-8")


async def send_event(
    response: web.StreamResponse,
    *,
    event: str,
    data: dict[str, Any],
    event_id: str | None = None,
) -> None:
    """编码并立即写出一个 SSE 事件。"""

    payload = encode_sse(
        event=event,
        data=data,
        event_id=event_id,
    )
    await response.write(payload)


async def health_handler(request: web.Request) -> web.Response:
    """普通 JSON 健康检查，不使用流式响应。"""

    return web.json_response(
        {
            "status": "ok",
            "service": "day17-sse-server",
            "model": llm.model_name,
        }
    )


async def stream_handler(request: web.Request) -> web.StreamResponse:
    """接收问题，把模型的异步流转换成 SSE 事件流。"""

    question = request.query.get("question", "").strip()

    if not question:
        return web.json_response(
            {"error": "查询参数 question 不能为空。"},
            status=400,
        )

    request_id = "req-" + uuid.uuid4().hex[:12]

    # StreamResponse 不会一次性生成完整 body，可以连续 write()。
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 禁止某些反向代理缓存整个响应后再统一发送。
            "X-Accel-Buffering": "no",
        },
    )

    # prepare() 发送 HTTP 状态行和响应头，之后才能逐块写 body。
    await response.prepare(request)

    print(f"[SSE 开始] request_id={request_id} question={question!r}")

    chunk_count = 0
    complete_parts: list[str] = []

    try:
        await send_event(
            response,
            event="start",
            event_id=f"{request_id}:0",
            data={
                "request_id": request_id,
                "model": llm.model_name,
                "question": question,
            },
        )

        messages = [
            SystemMessage(
                content="你是一名简洁、友好的编程学习助手。"
            ),
            HumanMessage(content=question),
        ]

        async for chunk in llm.astream(messages):
            if not isinstance(chunk.content, str):
                text = str(chunk.content)
            else:
                text = chunk.content

            # 空块常用于开始、结束或元数据，不必作为正文事件发送。
            if not text:
                continue

            chunk_count += 1
            complete_parts.append(text)

            await send_event(
                response,
                event="token",
                event_id=f"{request_id}:{chunk_count}",
                data={
                    "request_id": request_id,
                    "index": chunk_count,
                    "text": text,
                },
            )

        complete_answer = "".join(complete_parts)

        await send_event(
            response,
            event="done",
            event_id=f"{request_id}:{chunk_count + 1}",
            data={
                "request_id": request_id,
                "chunk_count": chunk_count,
                "answer_length": len(complete_answer),
            },
        )

        print(
            f"[SSE 完成] request_id={request_id} "
            f"chunks={chunk_count} length={len(complete_answer)}"
        )

    except (ConnectionResetError, asyncio.CancelledError):
        # 客户端关闭页面或中断请求时，停止继续写入网络连接。
        print(f"[SSE 断开] request_id={request_id} 客户端已经离开")

    except Exception as error:
        print(
            f"[SSE 错误] request_id={request_id} "
            f"type={type(error).__name__} error={error}"
        )

        # 如果连接仍然可写，就把错误作为 SSE 事件告诉客户端。
        try:
            await send_event(
                response,
                event="error",
                data={
                    "request_id": request_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
        except (ConnectionResetError, RuntimeError):
            pass

    finally:
        # write_eof() 表示 HTTP 响应体已经结束。
        try:
            await response.write_eof()
        except (ConnectionResetError, RuntimeError):
            pass

    return response


@web.middleware
async def logging_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    """记录请求方法和路径，再交给对应路由处理函数。"""

    print(f"[HTTP] {request.method} {request.path_qs}")
    return await handler(request)


def create_app() -> web.Application:
    """创建 aiohttp 应用并注册 HTTP 路由。"""

    app = web.Application(middlewares=[logging_middleware])
    app.router.add_get("/health", health_handler)
    app.router.add_get("/stream", stream_handler)
    return app


def main() -> None:
    """启动 SSE HTTP Server。"""

    print("Day17 SSE 模型服务正在启动。")
    print(f"服务地址：http://{HOST}:{PORT}")
    print("健康检查：GET /health")
    print("模型流：GET /stream?question=你的问题")
    print("按 Ctrl+C 停止服务。")

    web.run_app(
        create_app(),
        host=HOST,
        port=PORT,
        print=None,
    )


if __name__ == "__main__":
    main()
