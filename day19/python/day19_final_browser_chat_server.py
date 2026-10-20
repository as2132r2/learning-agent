"""Day19 第六步：可靠的浏览器 WebSocket 流式聊天服务。

运行：python day19/python/day19_final_browser_chat_server.py
访问：http://127.0.0.1:8021
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError


load_dotenv()

HOST = "127.0.0.1"
PORT = 8021
PROTOCOL_VERSION = "1.0"
MAX_MODEL_CONCURRENCY = 2
MAX_HISTORY_MESSAGES = 12

DAY19_DIR = Path(__file__).resolve().parents[1]
HTML_FILE = DAY19_DIR / "html" / "day19_final_browser_chat.html"
JAVASCRIPT_FILE = DAY19_DIR / "javascript" / "day19_final_browser_chat.js"
CSS_FILE = DAY19_DIR / "css" / "day19_final_browser_chat.css"

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

model_semaphore = asyncio.Semaphore(MAX_MODEL_CONCURRENCY)


class CommandPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str | None = Field(default=None, max_length=4000)


class ClientCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["1.0"]
    type: Literal["ask", "cancel", "status", "ping"]
    request_id: str | None = Field(default=None, max_length=100)
    payload: CommandPayload = Field(default_factory=CommandPayload)


@dataclass
class SessionState:
    """由 client_id 标识，可跨 WebSocket 重连复用。"""

    history: list[BaseMessage] = field(default_factory=list)
    completed_requests: int = 0
    cancelled_requests: int = 0


@dataclass
class ConnectionState:
    """只属于当前 WebSocket，断线后销毁。"""

    connection_id: str
    client_id: str
    session: SessionState
    active_request_id: str | None = None
    active_task: asyncio.Task[None] | None = None
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# 教学版保存于服务进程内存；重启服务后清空。
sessions: dict[str, SessionState] = {}


def serialize_history(messages: list[BaseMessage]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for message in messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        result.append({"role": role, "content": str(message.content)})
    return result


async def send_event(
    websocket: web.WebSocketResponse,
    state: ConnectionState,
    event_type: str,
    request_id: str | None,
    payload: dict[str, Any],
) -> None:
    async with state.send_lock:
        if not websocket.closed:
            await websocket.send_json(
                {
                    "version": PROTOCOL_VERSION,
                    "type": event_type,
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
    text_parts: list[str] = []
    acquired = False

    try:
        await send_event(
            websocket,
            state,
            "queued",
            request_id,
            {"max_model_concurrency": MAX_MODEL_CONCURRENCY},
        )

        await model_semaphore.acquire()
        acquired = True
        await send_event(
            websocket,
            state,
            "start",
            request_id,
            {"model": llm.model_name},
        )

        model_messages: list[BaseMessage] = [
            SystemMessage(
                content="你是一名简洁、准确、能记住当前会话的编程学习助手。"
            ),
            *state.session.history,
            HumanMessage(content=question),
        ]

        chunk_count = 0
        async for chunk in llm.astream(model_messages):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if not text:
                continue

            chunk_count += 1
            text_parts.append(text)
            await send_event(
                websocket,
                state,
                "token",
                request_id,
                {"index": chunk_count, "text": text},
            )

        complete_answer = "".join(text_parts)

        # 事务式提交：只有完整成功后才把这一问一答写入会话历史。
        state.session.history.extend(
            [HumanMessage(content=question), AIMessage(content=complete_answer)]
        )
        state.session.history = state.session.history[-MAX_HISTORY_MESSAGES:]
        state.session.completed_requests += 1

        await send_event(
            websocket,
            state,
            "done",
            request_id,
            {
                "chunk_count": chunk_count,
                "answer_length": len(complete_answer),
                "history_messages": len(state.session.history),
            },
        )

    except asyncio.CancelledError:
        state.session.cancelled_requests += 1
        await send_event(
            websocket,
            state,
            "cancelled",
            request_id,
            {
                "partial_answer": "".join(text_parts),
                "saved_to_history": False,
            },
        )

    except Exception as error:
        await send_event(
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


async def handle_command(
    websocket: web.WebSocketResponse,
    state: ConnectionState,
    raw_text: str,
) -> None:
    try:
        command = ClientCommand.model_validate_json(raw_text)
    except ValidationError as error:
        await send_event(
            websocket,
            state,
            "error",
            None,
            {
                "code": "invalid_command",
                "details": error.errors(include_url=False, include_input=False),
            },
        )
        return

    if command.type == "ask":
        question = (command.payload.question or "").strip()
        if not command.request_id or not question:
            await send_event(
                websocket,
                state,
                "error",
                command.request_id,
                {"message": "ask 需要 request_id 和非空 question。"},
            )
            return

        if state.active_task is not None and not state.active_task.done():
            await send_event(
                websocket,
                state,
                "busy",
                command.request_id,
                {"active_request_id": state.active_request_id},
            )
            return

        state.active_request_id = command.request_id
        state.active_task = asyncio.create_task(
            run_model(websocket, state, command.request_id, question),
            name=f"browser-model-{command.request_id}",
        )

    elif command.type == "cancel":
        if (
            state.active_task is None
            or state.active_task.done()
            or command.request_id != state.active_request_id
        ):
            await send_event(
                websocket, state, "cancel_not_found", command.request_id, {}
            )
            return

        await send_event(
            websocket, state, "cancel_requested", command.request_id, {}
        )
        state.active_task.cancel()

    elif command.type == "status":
        await send_event(
            websocket,
            state,
            "status_result",
            command.request_id,
            {
                "client_id": state.client_id,
                "active_request_id": state.active_request_id,
                "history_messages": len(state.session.history),
                "completed_requests": state.session.completed_requests,
                "cancelled_requests": state.session.cancelled_requests,
                "known_sessions": len(sessions),
            },
        )

    elif command.type == "ping":
        await send_event(websocket, state, "pong", command.request_id, {})


def normalized_client_id(request: web.Request) -> str:
    candidate = request.query.get("client_id", "").strip()
    if 1 <= len(candidate) <= 80 and all(
        character.isalnum() or character in "-_" for character in candidate
    ):
        return candidate
    return "client-" + uuid.uuid4().hex


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    websocket = web.WebSocketResponse(heartbeat=20, receive_timeout=300)
    await websocket.prepare(request)

    client_id = normalized_client_id(request)
    session = sessions.setdefault(client_id, SessionState())
    state = ConnectionState(
        connection_id="conn-" + uuid.uuid4().hex[:10],
        client_id=client_id,
        session=session,
    )

    print(f"[连接建立] {state.connection_id} | client={client_id}")
    await send_event(
        websocket,
        state,
        "connected",
        None,
        {
            "client_id": client_id,
            "model": llm.model_name,
            "history": serialize_history(session.history),
        },
    )

    try:
        async for raw_message in websocket:
            if raw_message.type == WSMsgType.TEXT:
                await handle_command(websocket, state, raw_message.data)
            elif raw_message.type == WSMsgType.ERROR:
                break
    finally:
        if state.active_task is not None and not state.active_task.done():
            state.active_task.cancel()
            await asyncio.gather(state.active_task, return_exceptions=True)
        print(f"[连接清理] {state.connection_id} | client={client_id}")

    return websocket


async def index_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(HTML_FILE)


async def javascript_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(JAVASCRIPT_FILE)


async def css_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(CSS_FILE)


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/app.js", javascript_handler)
    app.router.add_get("/app.css", css_handler)
    app.router.add_get("/ws", websocket_handler)
    return app


def main() -> None:
    print("Day19 最终浏览器聊天服务已启动。")
    print(f"浏览器页面：http://{HOST}:{PORT}")
    print(f"模型并发上限：{MAX_MODEL_CONCURRENCY}")
    print("按 Ctrl+C 停止服务。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
