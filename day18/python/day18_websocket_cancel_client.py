"""Day18 第五步（客户端）：接收模型流期间发送 cancel。

先启动服务端：
    python day18/python/day18_websocket_cancel_server.py

再运行客户端：
    python day18/python/day18_websocket_cancel_client.py
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from aiohttp import ClientSession, WSMsgType


WEBSOCKET_URL = "ws://127.0.0.1:8014/ws"
CANCEL_AFTER_TOKENS = 12

QUESTION = (
    "请分十点详细讲解 Python 异步任务，每一点都给出原理和代码示例。"
)


async def main() -> None:
    """发送 ask，收到固定数量 token 后沿同一连接发送 cancel。"""

    request_id = "req-" + uuid.uuid4().hex[:12]
    token_count = 0
    text_parts: list[str] = []
    cancel_sent = False
    cancel_requested_received = False
    cancelled_received = False

    print("Day18 WebSocket 模型取消客户端")
    print("地址：", WEBSOCKET_URL)
    print("request_id：", request_id)
    print("取消阈值：", CANCEL_AFTER_TOKENS, "个 token 消息")
    print("=" * 72)

    async with ClientSession() as session:
        async with session.ws_connect(
            WEBSOCKET_URL,
            heartbeat=20,
        ) as websocket:
            connected: Any = await websocket.receive_json()
            print("连接消息：", connected)

            await websocket.send_json(
                {
                    "version": "1.0",
                    "type": "ask",
                    "request_id": request_id,
                    "payload": {"question": QUESTION},
                }
            )

            async for raw_message in websocket:
                if raw_message.type != WSMsgType.TEXT:
                    continue

                message: Any = raw_message.json()
                if message.get("request_id") != request_id:
                    continue

                message_type = message.get("type")
                payload = message.get("payload", {})

                if message_type == "start":
                    print("\n助手：", end="", flush=True)

                elif message_type == "token":
                    token_count += 1
                    text = str(payload.get("text", ""))
                    text_parts.append(text)
                    print(text, end="", flush=True)

                    if (
                        token_count >= CANCEL_AFTER_TOKENS
                        and not cancel_sent
                    ):
                        cancel_sent = True

                        print("\n\n[客户端发送 cancel]")
                        await websocket.send_json(
                            {
                                "version": "1.0",
                                "type": "cancel",
                                "request_id": request_id,
                                "payload": {},
                            }
                        )

                elif message_type == "cancel_requested":
                    cancel_requested_received = True
                    print("[服务端确认收到取消指令]")

                elif message_type == "cancelled":
                    cancelled_received = True
                    print("[模型 Task 已取消]")
                    print("取消统计：", payload)
                    break

                elif message_type == "done":
                    print("[模型在 cancel 前已经完成]")
                    break

                elif message_type == "error":
                    print("[服务端错误]", payload)
                    break

            await websocket.close()

    print("\n" + "=" * 72)
    print("取消结果")
    print("已发送 cancel：", cancel_sent)
    print("收到 cancel_requested：", cancel_requested_received)
    print("收到 cancelled：", cancelled_received)
    print("取消前 token 数：", token_count)
    print("取消前部分回答：")
    print("".join(text_parts))

    print("\n本步数据流：")
    print("ask(request_id)")
    print("→ 服务端 create_task(model request)")
    print("→ 主接收循环立即恢复")
    print("→ token 持续返回")
    print("→ 客户端在同一连接发送 cancel(request_id)")
    print("→ active_tasks[request_id].cancel()")
    print("→ 模型 Task 捕获 CancelledError")
    print("→ 返回 cancelled 并清理字典")


if __name__ == "__main__":
    asyncio.run(main())
