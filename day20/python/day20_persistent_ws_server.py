"""Day20 第三步：从 SQLite 恢复历史并持久化模型回答的 WebSocket 服务。

运行：python day20/python/day20_persistent_ws_server.py
地址：ws://127.0.0.1:8022/ws?client_id=...&session_id=...
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict
from typing import Any

from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from day20_chat_repository import ChatMessage, ChatRepository


load_dotenv()

HOST = "127.0.0.1"
PORT = 8022

repository = ChatRepository()
repository.initialize()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


def valid_identifier(value: str) -> bool:
    """限制客户端提供的标识符字符和长度。"""

    return 1 <= len(value) <= 80 and all(
        character.isalnum() or character in "-_" for character in value
    )


def to_langchain_messages(
    stored_messages: list[ChatMessage],
) -> list[BaseMessage]:
    """将数据库领域对象转换成 LangChain 消息对象。"""

    result: list[BaseMessage] = []
    for message in stored_messages:
        if message.role == "user":
            result.append(HumanMessage(content=message.content))
        else:
            result.append(AIMessage(content=message.content))
    return result


async def send_event(
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


async def ensure_owned_session(
    client_id: str,
    session_id: str,
) -> tuple[bool, str]:
    """创建缺失会话，或检查已有会话是否属于当前客户端。"""

    session = await asyncio.to_thread(repository.get_session, session_id)

    if session is None:
        await asyncio.to_thread(
            repository.create_session,
            session_id,
            client_id,
            "Day20 持久化对话",
        )
        return True, "created"

    if session.client_id != client_id:
        return False, "session_owner_mismatch"

    return True, "restored"


async def answer_question(
    websocket: web.WebSocketResponse,
    session_id: str,
    request_id: str,
    question: str,
) -> None:
    """从数据库加载历史、流式回答，并在成功后持久化完整问答。"""

    stored_history = await asyncio.to_thread(
        repository.load_messages,
        session_id,
        12,
    )

    model_messages: list[BaseMessage] = [
        SystemMessage(
            content="你是一名简洁、准确、能记住当前会话的编程学习助手。"
        ),
        *to_langchain_messages(stored_history),
        HumanMessage(content=question),
    ]

    await send_event(
        websocket,
        "start",
        request_id,
        {
            "model": llm.model_name,
            "loaded_history_messages": len(stored_history),
        },
    )

    text_parts: list[str] = []
    chunk_count = 0

    try:
        async for chunk in llm.astream(model_messages):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if not text:
                continue

            chunk_count += 1
            text_parts.append(text)
            await send_event(
                websocket,
                "token",
                request_id,
                {"index": chunk_count, "text": text},
            )

        complete_answer = "".join(text_parts)

        # 同步 SQLite 操作在线程中执行；一问一答在同一事务中提交。
        saved_messages = await asyncio.to_thread(
            repository.append_turn,
            session_id,
            question,
            complete_answer,
        )

        await send_event(
            websocket,
            "done",
            request_id,
            {
                "chunk_count": chunk_count,
                "saved_message_ids": [
                    message.message_id for message in saved_messages
                ],
                "database_message_count": await asyncio.to_thread(
                    repository.count_messages,
                    session_id,
                ),
            },
        )

    except Exception as error:
        await send_event(
            websocket,
            "error",
            request_id,
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "saved_to_database": False,
            },
        )


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    client_id = request.query.get("client_id", "").strip()
    session_id = request.query.get("session_id", "").strip()

    websocket = web.WebSocketResponse(heartbeat=20)
    await websocket.prepare(request)

    if not valid_identifier(client_id) or not valid_identifier(session_id):
        await send_event(
            websocket,
            "error",
            None,
            {"message": "client_id 或 session_id 格式无效。"},
        )
        await websocket.close(code=1008, message=b"invalid identifier")
        return websocket

    owned, action = await ensure_owned_session(client_id, session_id)
    if not owned:
        await send_event(
            websocket,
            "error",
            None,
            {"message": "该 session_id 属于其他 client_id。"},
        )
        await websocket.close(code=1008, message=b"owner mismatch")
        return websocket

    history = await asyncio.to_thread(repository.load_messages, session_id, 20)
    await send_event(
        websocket,
        "connected",
        None,
        {
            "client_id": client_id,
            "session_id": session_id,
            "session_action": action,
            "history": [asdict(message) for message in history],
            "database": str(repository.database_path),
        },
    )

    print(
        f"[连接] client={client_id} session={session_id} "
        f"action={action} history={len(history)}"
    )

    try:
        async for raw_message in websocket:
            if raw_message.type == WSMsgType.TEXT:
                try:
                    data = raw_message.json()
                    request_id = str(data.get("request_id", "")).strip()
                    question = str(data.get("question", "")).strip()
                    if not request_id or not question:
                        raise ValueError("request_id 和 question 不能为空。")

                    # 本步骤一条连接同一时间只处理一个问题。
                    await answer_question(
                        websocket,
                        session_id,
                        request_id,
                        question,
                    )
                except (ValueError, TypeError) as error:
                    await send_event(
                        websocket,
                        "error",
                        None,
                        {"message": str(error), "saved_to_database": False},
                    )
            elif raw_message.type == WSMsgType.ERROR:
                break
    finally:
        print(f"[断开] client={client_id} session={session_id}")

    return websocket


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/ws", websocket_handler)
    return app


def main() -> None:
    print("Day20 SQLite 持久化 WebSocket 服务已启动。")
    print(f"地址：ws://{HOST}:{PORT}/ws")
    print("数据库：", repository.database_path)
    print("按 Ctrl+C 停止服务。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
