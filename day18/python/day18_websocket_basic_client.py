"""Day18 第一步（客户端）：在一条 WebSocket 上连续收发消息。

先启动服务端：
    python day18/python/day18_websocket_basic_server.py

再运行客户端：
    python day18/python/day18_websocket_basic_client.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from aiohttp import ClientSession, ClientTimeout, WSMsgType


WEBSOCKET_URL = "ws://127.0.0.1:8010/ws"


async def receive_json(websocket: Any) -> dict[str, Any]:
    """接收一条 WebSocket 消息，并验证它是 JSON object。"""

    message = await websocket.receive()

    if message.type == WSMsgType.TEXT:
        value = message.json()
        if not isinstance(value, dict):
            raise ValueError("服务端 JSON 顶层不是 object。")
        return value

    if message.type in (
        WSMsgType.CLOSE,
        WSMsgType.CLOSED,
        WSMsgType.CLOSING,
    ):
        raise ConnectionError("服务端已经关闭 WebSocket。")

    if message.type == WSMsgType.ERROR:
        raise ConnectionError(f"WebSocket 错误：{websocket.exception()}")

    raise ValueError(f"收到未处理的消息类型：{message.type}")


async def main() -> None:
    """建立一次连接，依次执行 echo、status 和 close。"""

    print("Day18 WebSocket 基础客户端")
    print("服务地址：", WEBSOCKET_URL)
    print("=" * 72)

    timeout = ClientTimeout(total=30)

    async with ClientSession(timeout=timeout) as session:
        # ws_connect() 先发起 HTTP Upgrade，成功后返回 WebSocket 对象。
        async with session.ws_connect(
            WEBSOCKET_URL,
            heartbeat=20,
        ) as websocket:
            connected = await receive_json(websocket)
            print("连接消息：", connected)

            messages_to_send = [
                {
                    "type": "echo",
                    "text": "hello websocket",
                },
                {
                    "type": "echo",
                    "text": "客户端和服务端可以双向通信",
                },
                {
                    "type": "status",
                },
            ]

            for index, outgoing in enumerate(
                messages_to_send,
                start=1,
            ):
                print(f"\n发送消息 {index}：", outgoing)
                await websocket.send_json(outgoing)

                incoming = await receive_json(websocket)
                print(f"收到回复 {index}：", incoming)

            print("\n发送关闭指令。")
            await websocket.send_json({"type": "close"})

            closing = await receive_json(websocket)
            print("关闭消息：", closing)

            # 接收服务端的 WebSocket Close Frame。
            close_frame = await websocket.receive()
            print("最终帧类型：", close_frame.type)
            print("WebSocket 是否关闭：", websocket.closed)

    print("\n" + "=" * 72)
    print("本步数据流：")
    print("HTTP GET /ws + Upgrade: websocket")
    print("→ 服务端返回 101 Switching Protocols")
    print("→ 建立一条持久 WebSocket")
    print("→ 客户端 send_json(echo)")
    print("→ 服务端 receive() + send_json(echo_result)")
    print("→ 同一连接继续发送 status")
    print("→ 客户端发送 close")
    print("→ 服务端发送 Close Frame")


if __name__ == "__main__":
    asyncio.run(main())
