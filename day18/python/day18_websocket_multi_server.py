"""Day18 第四步（服务端）：多 WebSocket 客户端与连接状态隔离。

运行方式：
    python day18/python/day18_websocket_multi_server.py

地址：ws://127.0.0.1:8013/ws

每个 websocket_handler() 调用都拥有独立的：
    connection_id、client_name、request_count。
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any

from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()


HOST = "127.0.0.1"
PORT = 8013
PROTOCOL_VERSION = "1.0"


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


active_connections = 0
connections_lock = asyncio.Lock()


async def change_connections(delta: int) -> int:
    """在 Lock 中修改活动连接数并返回新值。"""

    global active_connections

    async with connections_lock:
        active_connections += delta
        return active_connections


async def send_message(
    websocket: web.WebSocketResponse,
    *,
    message_type: str,
    connection_id: str,
    request_id: str | None,
    payload: dict[str, Any],
) -> None:
    """发送包含连接编号和请求编号的统一消息。"""

    await websocket.send_json(
        {
            "version": PROTOCOL_VERSION,
            "type": message_type,
            "connection_id": connection_id,
            "request_id": request_id,
            "payload": payload,
        }
    )


async def stream_answer(
    websocket: web.WebSocketResponse,
    *,
    connection_id: str,
    client_name: str,
    request_id: str,
    question: str,
    request_number: int,
) -> None:
    """为当前连接生成一条独立模型流。"""

    await send_message(
        websocket,
        message_type="start",
        connection_id=connection_id,
        request_id=request_id,
        payload={
            "client_name": client_name,
            "request_number": request_number,
        },
    )

    messages = [
        SystemMessage(
            content=(
                f"你正在服务客户端 {client_name}。"
                "请使用简洁、准确的 Python 术语回答。"
            )
        ),
        HumanMessage(content=question),
    ]

    chunk_count = 0

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
            await send_message(
                websocket,
                message_type="token",
                connection_id=connection_id,
                request_id=request_id,
                payload={"index": chunk_count, "text": text},
            )

        await send_message(
            websocket,
            message_type="done",
            connection_id=connection_id,
            request_id=request_id,
            payload={
                "client_name": client_name,
                "request_number": request_number,
                "chunk_count": chunk_count,
            },
        )

    except Exception as error:
        await send_message(
            websocket,
            message_type="error",
            connection_id=connection_id,
            request_id=request_id,
            payload={
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """为一条连接维护独立局部状态。"""

    connection_id = "conn-" + uuid.uuid4().hex[:10]
    client_name = "anonymous"
    request_count = 0

    websocket = web.WebSocketResponse(heartbeat=20)
    await websocket.prepare(request)

    current_connections = await change_connections(1)

    print(
        f"[连接建立] {connection_id} "
        f"active={current_connections}"
    )

    await send_message(
        websocket,
        message_type="connected",
        connection_id=connection_id,
        request_id=None,
        payload={"active_connections": current_connections},
    )

    try:
        async for raw_message in websocket:
            if raw_message.type != WSMsgType.TEXT:
                continue

            try:
                data: Any = json.loads(raw_message.data)
            except json.JSONDecodeError:
                await send_message(
                    websocket,
                    message_type="error",
                    connection_id=connection_id,
                    request_id=None,
                    payload={"message": "消息必须是 JSON。"},
                )
                continue

            if not isinstance(data, dict):
                continue

            message_type = data.get("type")
            request_id = data.get("request_id")
            payload = data.get("payload", {})

            if not isinstance(payload, dict):
                continue

            if message_type == "set_name":
                proposed_name = str(payload.get("name", "")).strip()
                if proposed_name:
                    client_name = proposed_name

                await send_message(
                    websocket,
                    message_type="name_set",
                    connection_id=connection_id,
                    request_id=None,
                    payload={"client_name": client_name},
                )

            elif message_type == "ask" and isinstance(request_id, str):
                question = str(payload.get("question", "")).strip()
                if not question:
                    continue

                request_count += 1

                await stream_answer(
                    websocket,
                    connection_id=connection_id,
                    client_name=client_name,
                    request_id=request_id,
                    question=question,
                    request_number=request_count,
                )

    finally:
        current_connections = await change_connections(-1)
        print(
            f"[连接结束] {connection_id} "
            f"name={client_name!r} requests={request_count} "
            f"active={current_connections}"
        )

        if not websocket.closed:
            await websocket.close()

    return websocket


def create_app() -> web.Application:
    """创建多客户端 WebSocket 服务。"""

    app = web.Application()
    app.router.add_get("/ws", websocket_handler)
    return app


def main() -> None:
    """启动服务。"""

    print("Day18 多客户端 WebSocket 服务正在启动。")
    print(f"地址：ws://{HOST}:{PORT}/ws")
    print("按 Ctrl+C 停止服务。")

    web.run_app(
        create_app(),
        host=HOST,
        port=PORT,
        print=None,
    )


if __name__ == "__main__":
    main()
