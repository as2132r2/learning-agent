"""Day20 第四步：一个客户端创建、列出并切换多个持久化会话。

运行：python day20/python/day20_multi_session_server.py
地址：ws://127.0.0.1:8023/ws?client_id=xiaolin
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from day20_chat_repository import ChatMessage, ChatRepository


load_dotenv()

HOST = "127.0.0.1"
PORT = 8023

repository = ChatRepository()
repository.initialize()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


@dataclass
class ConnectionState:
    client_id: str
    current_session_id: str | None = None


def valid_identifier(value: str) -> bool:
    return 1 <= len(value) <= 80 and all(
        character.isalnum() or character in "-_" for character in value
    )


def to_model_history(messages: list[ChatMessage]) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for message in messages:
        if message.role == "user":
            result.append(HumanMessage(content=message.content))
        else:
            result.append(AIMessage(content=message.content))
    return result


async def send(
    websocket: web.WebSocketResponse,
    event_type: str,
    request_id: str | None,
    payload: dict[str, Any],
) -> None:
    if not websocket.closed:
        await websocket.send_json(
            {
                "type": event_type,
                "request_id": request_id,
                "payload": payload,
            }
        )


async def session_snapshot(
    client_id: str,
) -> list[dict[str, Any]]:
    sessions = await asyncio.to_thread(repository.list_sessions, client_id)
    return [asdict(session) for session in sessions]


async def history_snapshot(session_id: str) -> list[dict[str, Any]]:
    messages = await asyncio.to_thread(repository.load_messages, session_id, 30)
    return [asdict(message) for message in messages]


async def answer_question(
    websocket: web.WebSocketResponse,
    state: ConnectionState,
    request_id: str,
    question: str,
) -> None:
    session_id = state.current_session_id
    if session_id is None:
        await send(
            websocket,
            "error",
            request_id,
            {"message": "请先使用 /new 创建会话，或使用 /switch 切换会话。"},
        )
        return

    stored_messages = await asyncio.to_thread(
        repository.load_messages,
        session_id,
        12,
    )
    model_messages: list[BaseMessage] = [
        SystemMessage(content="你是一名简洁、友好的编程学习助手。"),
        *to_model_history(stored_messages),
        HumanMessage(content=question),
    ]

    await send(
        websocket,
        "start",
        request_id,
        {
            "session_id": session_id,
            "loaded_history_messages": len(stored_messages),
        },
    )

    parts: list[str] = []
    try:
        async for chunk in llm.astream(model_messages):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                parts.append(text)
                await send(
                    websocket,
                    "token",
                    request_id,
                    {"session_id": session_id, "text": text},
                )

        saved = await asyncio.to_thread(
            repository.append_turn,
            session_id,
            question,
            "".join(parts),
        )
        await send(
            websocket,
            "done",
            request_id,
            {
                "session_id": session_id,
                "saved_message_ids": [item.message_id for item in saved],
            },
        )
    except Exception as error:
        await send(
            websocket,
            "error",
            request_id,
            {"error_type": type(error).__name__, "message": str(error)},
        )


async def handle_command(
    websocket: web.WebSocketResponse,
    state: ConnectionState,
    data: dict[str, Any],
) -> None:
    command_type = str(data.get("type", ""))
    request_id_value = data.get("request_id")
    request_id = str(request_id_value) if request_id_value is not None else None
    payload = data.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}

    if command_type == "list_sessions":
        await send(
            websocket,
            "sessions",
            request_id,
            {
                "current_session_id": state.current_session_id,
                "sessions": await session_snapshot(state.client_id),
            },
        )

    elif command_type == "create_session":
        title = str(payload.get("title", "新对话")).strip() or "新对话"
        title = title[:100]
        session_id = "session-" + uuid.uuid4().hex[:12]
        session = await asyncio.to_thread(
            repository.create_session,
            session_id,
            state.client_id,
            title,
        )
        state.current_session_id = session_id
        await send(
            websocket,
            "session_created",
            request_id,
            {"session": asdict(session), "history": []},
        )

    elif command_type == "switch_session":
        session_id = str(payload.get("session_id", "")).strip()
        session = await asyncio.to_thread(repository.get_session, session_id)
        if session is None:
            await send(
                websocket,
                "error",
                request_id,
                {"message": "会话不存在。"},
            )
        elif session.client_id != state.client_id:
            await send(
                websocket,
                "error",
                request_id,
                {"message": "不能切换到其他客户端的会话。"},
            )
        else:
            state.current_session_id = session_id
            await send(
                websocket,
                "session_switched",
                request_id,
                {
                    "session": asdict(session),
                    "history": await history_snapshot(session_id),
                },
            )

    elif command_type == "ask":
        question = str(payload.get("question", "")).strip()
        if not request_id or not question:
            await send(
                websocket,
                "error",
                request_id,
                {"message": "ask 需要 request_id 和非空问题。"},
            )
        else:
            await answer_question(websocket, state, request_id, question)

    else:
        await send(
            websocket,
            "error",
            request_id,
            {"message": f"未知命令：{command_type!r}"},
        )


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    client_id = request.query.get("client_id", "").strip()
    websocket = web.WebSocketResponse(heartbeat=20)
    await websocket.prepare(request)

    if not valid_identifier(client_id):
        await send(websocket, "error", None, {"message": "client_id 格式无效。"})
        await websocket.close(code=1008, message=b"invalid client id")
        return websocket

    existing_sessions = await asyncio.to_thread(repository.list_sessions, client_id)
    state = ConnectionState(
        client_id=client_id,
        current_session_id=(
            existing_sessions[0].session_id if existing_sessions else None
        ),
    )

    initial_history: list[dict[str, Any]] = []
    if state.current_session_id is not None:
        initial_history = await history_snapshot(state.current_session_id)

    await send(
        websocket,
        "connected",
        None,
        {
            "client_id": client_id,
            "current_session_id": state.current_session_id,
            "sessions": [asdict(session) for session in existing_sessions],
            "history": initial_history,
        },
    )
    print(
        f"[连接] client={client_id} sessions={len(existing_sessions)} "
        f"current={state.current_session_id}"
    )

    try:
        async for raw_message in websocket:
            if raw_message.type == WSMsgType.TEXT:
                try:
                    data = raw_message.json()
                    if not isinstance(data, dict):
                        raise ValueError("消息必须是 JSON 对象。")
                    await handle_command(websocket, state, data)
                except (ValueError, TypeError) as error:
                    await send(websocket, "error", None, {"message": str(error)})
            elif raw_message.type == WSMsgType.ERROR:
                break
    finally:
        print(f"[断开] client={client_id}")

    return websocket


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/ws", websocket_handler)
    return app


def main() -> None:
    print("Day20 多会话 WebSocket 服务已启动。")
    print(f"地址：ws://{HOST}:{PORT}/ws?client_id=xiaolin")
    print("数据库：", repository.database_path)
    print("按 Ctrl+C 停止服务。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
