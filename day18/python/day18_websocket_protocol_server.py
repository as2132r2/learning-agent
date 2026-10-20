"""Day18 第三步（服务端）：结构化 WebSocket 消息协议。

运行方式：
    python day18/python/day18_websocket_protocol_server.py

WebSocket 地址：ws://127.0.0.1:8012/ws

客户端 ask 消息：
    {
        "version": "1.0",
        "type": "ask",
        "request_id": "req-...",
        "payload": {"question": "..."}
    }

服务端所有响应都保留相同的 version 和 request_id。
"""

from __future__ import annotations

import os
from typing import Any, Literal

from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError


load_dotenv()


HOST = "127.0.0.1"
PORT = 8012
PROTOCOL_VERSION = "1.0"


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


class AskPayload(BaseModel):
    """ask 消息 payload 的数据结构。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)


class AskMessage(BaseModel):
    """客户端允许发送的 ask 协议消息。"""

    model_config = ConfigDict(extra="forbid")

    version: Literal["1.0"]
    type: Literal["ask"]
    request_id: str = Field(min_length=5, max_length=100)
    payload: AskPayload


async def send_message(
    websocket: web.WebSocketResponse,
    *,
    message_type: str,
    request_id: str | None,
    payload: dict[str, Any],
) -> None:
    """按照统一协议包装并发送服务端消息。"""

    await websocket.send_json(
        {
            "version": PROTOCOL_VERSION,
            "type": message_type,
            "request_id": request_id,
            "payload": payload,
        }
    )


async def stream_answer(
    websocket: web.WebSocketResponse,
    request_id: str,
    question: str,
) -> None:
    """生成模型回答，并让每条流式消息携带 request_id。"""

    await send_message(
        websocket,
        message_type="start",
        request_id=request_id,
        payload={
            "model": llm.model_name,
            "question": question,
        },
    )

    messages = [
        SystemMessage(
            content="你是一名简洁、准确的 Python 编程学习助手。"
        ),
        HumanMessage(content=question),
    ]

    chunk_count = 0
    answer_length = 0

    try:
        async for chunk in llm.astream(messages):
            text = (
                chunk.content
                if isinstance(chunk.content, str)
                else str(chunk.content)
            )

            if not text:
                continue

            chunk_count += 1
            answer_length += len(text)

            await send_message(
                websocket,
                message_type="token",
                request_id=request_id,
                payload={
                    "index": chunk_count,
                    "text": text,
                },
            )

        await send_message(
            websocket,
            message_type="done",
            request_id=request_id,
            payload={
                "chunk_count": chunk_count,
                "answer_length": answer_length,
            },
        )

    except Exception as error:
        await send_message(
            websocket,
            message_type="error",
            request_id=request_id,
            payload={
                "code": "model_error",
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """校验协议消息并处理模型请求。"""

    websocket = web.WebSocketResponse(heartbeat=20)
    await websocket.prepare(request)

    await send_message(
        websocket,
        message_type="connected",
        request_id=None,
        payload={
            "model": llm.model_name,
            "supported_types": ["ask"],
        },
    )

    try:
        async for raw_message in websocket:
            if raw_message.type != WSMsgType.TEXT:
                if raw_message.type == WSMsgType.ERROR:
                    print("[WebSocket 错误]", websocket.exception())
                continue

            try:
                ask = AskMessage.model_validate_json(raw_message.data)
            except ValidationError as error:
                await send_message(
                    websocket,
                    message_type="error",
                    request_id=None,
                    payload={
                        "code": "invalid_protocol_message",
                        "message": "消息不符合协议 Schema。",
                        "details": error.errors(
                            include_url=False,
                            include_input=False,
                        ),
                    },
                )
                continue

            question = ask.payload.question.strip()
            if not question:
                await send_message(
                    websocket,
                    message_type="error",
                    request_id=ask.request_id,
                    payload={
                        "code": "empty_question",
                        "message": "question 去除空格后不能为空。",
                    },
                )
                continue

            print(
                f"[收到 ask] request_id={ask.request_id} "
                f"question={question!r}"
            )

            await stream_answer(
                websocket,
                ask.request_id,
                question,
            )

    finally:
        if not websocket.closed:
            await websocket.close()

    return websocket


def create_app() -> web.Application:
    """创建 WebSocket 协议服务。"""

    app = web.Application()
    app.router.add_get("/ws", websocket_handler)
    return app


def main() -> None:
    """启动服务。"""

    print("Day18 WebSocket 协议服务正在启动。")
    print(f"地址：ws://{HOST}:{PORT}/ws")
    print("协议版本：", PROTOCOL_VERSION)
    print("按 Ctrl+C 停止服务。")

    web.run_app(
        create_app(),
        host=HOST,
        port=PORT,
        print=None,
    )


if __name__ == "__main__":
    main()
