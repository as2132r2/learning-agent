"""Day18 第六步（最终客户端）：交互式 WebSocket 流式聊天。

先启动：python day18/python/day18_final_websocket_server.py
再运行：python day18/python/day18_final_websocket_client.py

命令：/cancel 取消当前回答，/status 查看状态，/exit 退出。
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from aiohttp import ClientSession, ClientWebSocketResponse, WSMsgType


WEBSOCKET_URL = "ws://127.0.0.1:8015/ws"


@dataclass
class ClientState:
    """客户端本地界面状态。"""

    active_request_id: str | None = None
    text_parts: list[str] = field(default_factory=list)
    running: bool = True


def envelope(
    message_type: str,
    request_id: str | None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """创建符合 1.0 协议的客户端消息。"""

    return {
        "version": "1.0",
        "type": message_type,
        "request_id": request_id,
        "payload": payload or {},
    }


async def receive_loop(
    websocket: ClientWebSocketResponse,
    state: ClientState,
) -> None:
    """唯一的接收协程：显示服务端所有异步消息。"""

    async for raw_message in websocket:
        if raw_message.type != WSMsgType.TEXT:
            if raw_message.type == WSMsgType.ERROR:
                print("\n[WebSocket 错误]", websocket.exception())
            continue

        message: Any = raw_message.json()
        message_type = message.get("type")
        request_id = message.get("request_id")
        payload = message.get("payload", {})

        if message_type == "queued":
            print("\n[请求已进入模型队列]")

        elif message_type == "start":
            print("[模型开始回答]\n助手：", end="", flush=True)

        elif message_type == "token":
            text = str(payload.get("text", ""))
            state.text_parts.append(text)
            print(text, end="", flush=True)

        elif message_type == "done":
            print("\n[回答完成]", payload)
            state.active_request_id = None
            state.text_parts = []

        elif message_type == "cancel_requested":
            print("\n[服务端收到取消指令]")

        elif message_type == "cancelled":
            print("[模型任务已取消]", payload)
            state.active_request_id = None
            state.text_parts = []

        elif message_type == "status_result":
            print("\n[服务端状态]", payload)

        elif message_type == "busy":
            print("\n[当前仍有请求运行]", payload)

        elif message_type in ("error", "cancel_not_found"):
            print(f"\n[{message_type}] request_id={request_id} {payload}")
            if request_id == state.active_request_id:
                state.active_request_id = None

    state.running = False


async def input_loop(
    websocket: ClientWebSocketResponse,
    state: ClientState,
) -> None:
    """读取终端输入，并在模型输出期间仍可发送控制命令。"""

    while state.running:
        user_input = await asyncio.to_thread(input, "\n你：")
        user_input = user_input.strip()

        if not user_input:
            continue

        if user_input == "/exit":
            if state.active_request_id is not None:
                await websocket.send_json(
                    envelope("cancel", state.active_request_id)
                )
                await asyncio.sleep(0.1)
            state.running = False
            await websocket.close()
            return

        if user_input == "/cancel":
            if state.active_request_id is None:
                print("当前没有运行中的请求。")
            else:
                await websocket.send_json(
                    envelope("cancel", state.active_request_id)
                )
            continue

        if user_input == "/status":
            await websocket.send_json(
                envelope(
                    "status",
                    "status-" + uuid.uuid4().hex[:8],
                )
            )
            continue

        if state.active_request_id is not None:
            print("当前回答尚未结束；可输入 /cancel 或等待完成。")
            continue

        request_id = "req-" + uuid.uuid4().hex[:12]
        state.active_request_id = request_id
        state.text_parts = []

        await websocket.send_json(
            envelope(
                "ask",
                request_id,
                {"question": user_input},
            )
        )


async def main() -> None:
    """建立连接，并发运行输入协程和接收协程。"""

    print("Day18 最终 WebSocket 流式聊天客户端")
    print("命令：/cancel、/status、/exit")
    print("地址：", WEBSOCKET_URL)

    state = ClientState()

    async with ClientSession() as session:
        async with session.ws_connect(
            WEBSOCKET_URL,
            heartbeat=20,
        ) as websocket:
            connected: Any = await websocket.receive_json()
            print("连接成功：", connected)

            receiver = asyncio.create_task(
                receive_loop(websocket, state),
                name="websocket-receiver",
            )
            keyboard = asyncio.create_task(
                input_loop(websocket, state),
                name="keyboard-input",
            )

            done, pending = await asyncio.wait(
                {receiver, keyboard},
                return_when=asyncio.FIRST_COMPLETED,
            )

            state.running = False
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await asyncio.gather(*done, return_exceptions=True)

    print("\n聊天结束。")
    print("客户端数据流：终端输入与 WebSocket 接收由两个独立 Task 并发处理。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n用户终止客户端。")
