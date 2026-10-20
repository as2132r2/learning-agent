"""Day20 第四步：创建、列出和切换多个持久化对话的终端客户端。

先启动服务端，再运行：
    python day20/python/day20_multi_session_client.py xiaolin

命令：/new 标题、/list、/switch session_id、/exit
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from typing import Any

from aiohttp import ClientSession, WSMsgType


HOST = "127.0.0.1"
PORT = 8023


def show_history(history: list[dict[str, Any]]) -> None:
    print("\n当前会话历史：")
    if not history:
        print("（暂无消息）")
    for message in history:
        role = "你" if message["role"] == "user" else "助手"
        print(f"{role}：{message['content']}")


def show_sessions(payload: dict[str, Any]) -> None:
    current = payload.get("current_session_id")
    sessions = payload.get("sessions", [])
    print("\n会话列表：")
    if not sessions:
        print("（暂无会话，请使用 /new 标题 创建）")
    for session in sessions:
        marker = "*" if session["session_id"] == current else " "
        print(f"{marker} {session['session_id']} | {session['title']}")


async def receive_until_terminal(websocket: Any) -> dict[str, Any]:
    """读取事件，直到收到一个命令的终止事件。"""

    while True:
        raw = await websocket.receive()
        if raw.type != WSMsgType.TEXT:
            raise ConnectionError("WebSocket 已断开。")

        message: dict[str, Any] = raw.json()
        event_type = message["type"]
        payload = message.get("payload", {})

        if event_type == "start":
            print(
                f"\n[会话 {payload['session_id']}，"
                f"加载历史 {payload['loaded_history_messages']} 条]\n助手：",
                end="",
                flush=True,
            )
        elif event_type == "token":
            print(payload.get("text", ""), end="", flush=True)
        elif event_type == "done":
            print("\n[回答已持久化]")
            return message
        elif event_type in {
            "sessions",
            "session_created",
            "session_switched",
            "error",
        }:
            return message


async def send_command(
    websocket: Any,
    command_type: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_id = "req-" + uuid.uuid4().hex[:10]
    await websocket.send_json(
        {
            "type": command_type,
            "request_id": request_id,
            "payload": payload or {},
        }
    )
    return await receive_until_terminal(websocket)


async def main() -> None:
    client_id = sys.argv[1] if len(sys.argv) > 1 else "xiaolin"
    url = f"ws://{HOST}:{PORT}/ws?client_id={client_id}"

    print("Day20 多会话客户端")
    print("client_id：", client_id)
    print("命令：/new 标题、/list、/switch session_id、/exit")

    async with ClientSession() as session:
        async with session.ws_connect(url, heartbeat=20) as websocket:
            connected: dict[str, Any] = await websocket.receive_json()
            connected_payload = connected["payload"]
            show_sessions(connected_payload)
            show_history(connected_payload["history"])

            while True:
                user_input = (await asyncio.to_thread(input, "\n你：")).strip()
                if not user_input:
                    continue
                if user_input == "/exit":
                    break

                if user_input == "/list":
                    response = await send_command(websocket, "list_sessions")
                    show_sessions(response["payload"])
                    continue

                if user_input.startswith("/new"):
                    title = user_input.removeprefix("/new").strip() or "新对话"
                    response = await send_command(
                        websocket,
                        "create_session",
                        {"title": title},
                    )
                    if response["type"] == "session_created":
                        created = response["payload"]["session"]
                        print(
                            f"已创建并切换：{created['session_id']} | "
                            f"{created['title']}"
                        )
                        show_history(response["payload"]["history"])
                    else:
                        print("创建失败：", response["payload"])
                    continue

                if user_input.startswith("/switch "):
                    target = user_input.removeprefix("/switch ").strip()
                    response = await send_command(
                        websocket,
                        "switch_session",
                        {"session_id": target},
                    )
                    if response["type"] == "session_switched":
                        switched = response["payload"]["session"]
                        print(f"已切换：{switched['title']}")
                        show_history(response["payload"]["history"])
                    else:
                        print("切换失败：", response["payload"])
                    continue

                response = await send_command(
                    websocket,
                    "ask",
                    {"question": user_input},
                )
                if response["type"] == "error":
                    print("错误：", response["payload"])

    print("\n客户端已退出，会话和消息仍保存在 SQLite。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n用户终止客户端。")
