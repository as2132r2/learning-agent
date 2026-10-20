"""Day18 第六步（最终服务端）：可靠的 WebSocket 流式对话服务。

运行：python day18/python/day18_final_websocket_server.py
地址：ws://127.0.0.1:8015/ws

整合能力：统一协议、Pydantic 校验、每连接记忆、模型流、全局并发限制、
后台 Task、cancel/status 指令、写入锁和断线清理。
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError


load_dotenv()

HOST = "127.0.0.1"
PORT = 8015
PROTOCOL_VERSION = "1.0"
MAX_MODEL_CONCURRENCY = 2

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

model_semaphore = asyncio.Semaphore(MAX_MODEL_CONCURRENCY)


class Payload(BaseModel):
    """客户端消息负载；不同 type 使用不同字段。"""

    model_config = ConfigDict(extra="forbid")
    question: str | None = Field(default=None, max_length=4000)


class ClientMessage(BaseModel):
    """客户端统一协议消息。"""

    model_config = ConfigDict(extra="forbid")
    version: Literal["1.0"]
    type: Literal["ask", "cancel", "status"]
    request_id: str | None
    payload: Payload = Field(default_factory=Payload)


@dataclass
class ConnectionState:
    """只属于一条 WebSocket 的状态。"""

    connection_id: str
    history: list[BaseMessage] = field(default_factory=list)
    active_request_id: str | None = None
    active_task: asyncio.Task[None] | None = None
    completed_requests: int = 0
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def send(
    websocket: web.WebSocketResponse,
    state: ConnectionState,
    message_type: str,
    request_id: str | None,
    payload: dict[str, Any],
) -> None:
    """串行化同一连接上的所有写操作。"""

    async with state.send_lock:
        if not websocket.closed:
            await websocket.send_json(
                {
                    "version": PROTOCOL_VERSION,
                    "type": message_type,
                    "connection_id": state.connection_id,
                    "request_id": request_id,
                    "payload": payload,
                }
            )


async def run_model(
    websocket: web.WebSocketResponse,
    state: ConnectionState,
    request_id: str,
    question: str,
) -> None:
    """等待全局许可、流式回答，并只在成功后提交本轮历史。"""

    text_parts: list[str] = []
    acquired = False

    try:
        await send(
            websocket,
            state,
            "queued",
            request_id,
            {"max_model_concurrency": MAX_MODEL_CONCURRENCY},
        )

        await model_semaphore.acquire()
        acquired = True

        await send(
            websocket,
            state,
            "start",
            request_id,
            {"model": llm.model_name},
        )

        model_messages: list[BaseMessage] = [
            SystemMessage(
                content="你是一名简洁、准确、能记住当前对话的 Python 学习助手。"
            ),
            *state.history,
            HumanMessage(content=question),
        ]

        chunk_count = 0

        async for chunk in llm.astream(model_messages):
            text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            if not text:
                continue

            chunk_count += 1
            text_parts.append(text)
            await send(
                websocket,
                state,
                "token",
                request_id,
                {"index": chunk_count, "text": text},
            )

        complete_answer = "".join(text_parts)

        # 事务式记忆：只有完整成功的一轮才写入历史。
        state.history.extend(
            [HumanMessage(content=question), AIMessage(content=complete_answer)]
        )
        state.completed_requests += 1

        await send(
            websocket,
            state,
            "done",
            request_id,
            {
                "chunk_count": chunk_count,
                "answer_length": len(complete_answer),
                "history_messages": len(state.history),
            },
        )

    except asyncio.CancelledError:
        await send(
            websocket,
            state,
            "cancelled",
            request_id,
            {
                "partial_answer": "".join(text_parts),
                "saved_to_history": False,
            },
        )
        return

    except Exception as error:
        await send(
            websocket,
            state,
            "error",
            request_id,
            {
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )

    finally:
        if acquired:
            model_semaphore.release()
        if state.active_request_id == request_id:
            state.active_request_id = None
            state.active_task = None


async def handle_client_message(
    websocket: web.WebSocketResponse,
    state: ConnectionState,
    raw_text: str,
) -> None:
    """校验并路由 ask、cancel、status。"""

    try:
        message = ClientMessage.model_validate_json(raw_text)
    except ValidationError as error:
        await send(
            websocket,
            state,
            "error",
            None,
            {
                "code": "invalid_message",
                "details": error.errors(include_url=False, include_input=False),
            },
        )
        return

    if message.type == "ask":
        if not message.request_id:
            await send(websocket, state, "error", None, {"message": "ask 需要 request_id。"})
            return

        question = (message.payload.question or "").strip()
        if not question:
            await send(websocket, state, "error", message.request_id, {"message": "问题不能为空。"})
            return

        if state.active_task is not None and not state.active_task.done():
            await send(
                websocket,
                state,
                "busy",
                message.request_id,
                {"active_request_id": state.active_request_id},
            )
            return

        state.active_request_id = message.request_id
        state.active_task = asyncio.create_task(
            run_model(websocket, state, message.request_id, question),
            name=f"model-{message.request_id}",
        )

    elif message.type == "cancel":
        if (
            state.active_task is None
            or state.active_task.done()
            or message.request_id != state.active_request_id
        ):
            await send(websocket, state, "cancel_not_found", message.request_id, {})
            return

        await send(websocket, state, "cancel_requested", message.request_id, {})
        state.active_task.cancel()

    elif message.type == "status":
        await send(
            websocket,
            state,
            "status_result",
            message.request_id,
            {
                "active_request_id": state.active_request_id,
                "history_messages": len(state.history),
                "completed_requests": state.completed_requests,
            },
        )


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """一条连接的主接收循环与生命周期管理。"""

    websocket = web.WebSocketResponse(heartbeat=20, receive_timeout=300)
    await websocket.prepare(request)

    state = ConnectionState(connection_id="conn-" + uuid.uuid4().hex[:10])

    await send(
        websocket,
        state,
        "connected",
        None,
        {
            "model": llm.model_name,
            "supported_types": ["ask", "cancel", "status"],
        },
    )

    try:
        async for raw_message in websocket:
            if raw_message.type == WSMsgType.TEXT:
                await handle_client_message(websocket, state, raw_message.data)
            elif raw_message.type == WSMsgType.ERROR:
                break
    finally:
        if state.active_task is not None and not state.active_task.done():
            state.active_task.cancel()
            await asyncio.gather(state.active_task, return_exceptions=True)
        if not websocket.closed:
            await websocket.close()

    return websocket


def create_app() -> web.Application:
    """创建最终 WebSocket 应用。"""

    app = web.Application()
    app.router.add_get("/ws", websocket_handler)
    return app


def main() -> None:
    print("Day18 最终 WebSocket 对话服务正在启动。")
    print(f"地址：ws://{HOST}:{PORT}/ws")
    print("模型并发上限：", MAX_MODEL_CONCURRENCY)
    print("按 Ctrl+C 停止服务。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
