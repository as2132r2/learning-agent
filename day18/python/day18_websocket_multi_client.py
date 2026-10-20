"""Day18 第四步（客户端）：同时建立三条独立 WebSocket 连接。

先启动服务端：
    python day18/python/day18_websocket_multi_server.py

再运行客户端：
    python day18/python/day18_websocket_multi_client.py
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientSession, WSMsgType


WEBSOCKET_URL = "ws://127.0.0.1:8013/ws"

CLIENT_JOBS = [
    ("xiaolin", "请用一句话解释协程。"),
    ("leo", "请用一句话解释事件循环。"),
    ("student-3", "请用一句话解释异步迭代器。"),
]


@dataclass
class ClientResult:
    """一条客户端连接的最终结果。"""

    client_name: str
    connection_id: str
    request_id: str
    answer: str
    request_number: int
    total_seconds: float


async def run_client(
    session: ClientSession,
    client_name: str,
    question: str,
) -> ClientResult:
    """建立独立连接、设置名称、提问并接收自己的回答。"""

    started_at = time.perf_counter()
    request_id = "req-" + uuid.uuid4().hex[:10]
    connection_id = ""
    text_parts: list[str] = []
    request_number = 0

    print(f"[{client_name}] 正在连接")

    async with session.ws_connect(
        WEBSOCKET_URL,
        heartbeat=20,
    ) as websocket:
        connected: Any = await websocket.receive_json()
        connection_id = str(connected["connection_id"])

        print(f"[{client_name}] connection_id={connection_id}")

        await websocket.send_json(
            {
                "version": "1.0",
                "type": "set_name",
                "request_id": None,
                "payload": {"name": client_name},
            }
        )

        name_set: Any = await websocket.receive_json()
        print(
            f"[{client_name}] 服务端保存名称="
            f"{name_set['payload']['client_name']}"
        )

        await websocket.send_json(
            {
                "version": "1.0",
                "type": "ask",
                "request_id": request_id,
                "payload": {"question": question},
            }
        )

        async for raw_message in websocket:
            if raw_message.type != WSMsgType.TEXT:
                continue

            message: Any = raw_message.json()

            if message.get("connection_id") != connection_id:
                raise RuntimeError("收到其他连接的消息。")

            incoming_request_id = message.get("request_id")
            if (
                incoming_request_id is not None
                and incoming_request_id != request_id
            ):
                raise RuntimeError("收到其他请求的消息。")

            message_type = message.get("type")
            payload = message.get("payload", {})

            if message_type == "token":
                text_parts.append(str(payload.get("text", "")))

            elif message_type == "done":
                request_number = int(payload["request_number"])
                break

            elif message_type == "error":
                raise RuntimeError(str(payload))

        await websocket.close()

    elapsed = time.perf_counter() - started_at
    print(f"[{client_name}] 完成，耗时={elapsed:.3f} 秒")

    return ClientResult(
        client_name=client_name,
        connection_id=connection_id,
        request_id=request_id,
        answer="".join(text_parts),
        request_number=request_number,
        total_seconds=elapsed,
    )


async def main() -> None:
    """并发运行三个独立客户端。"""

    print("Day18 WebSocket 多客户端隔离测试")
    print("地址：", WEBSOCKET_URL)
    print("客户端数量：", len(CLIENT_JOBS))
    print("=" * 72)

    overall_started_at = time.perf_counter()

    async with ClientSession() as session:
        tasks = [
            asyncio.create_task(
                run_client(session, name, question),
                name=f"ws-client-{name}",
            )
            for name, question in CLIENT_JOBS
        ]

        results: list[ClientResult] = await asyncio.gather(*tasks)

    overall_seconds = time.perf_counter() - overall_started_at

    print("\n" + "=" * 72)
    print("隔离结果：")

    for result in results:
        print("\n客户端：", result.client_name)
        print("connection_id：", result.connection_id)
        print("request_id：", result.request_id)
        print("该连接内请求序号：", result.request_number)
        print("回答：", result.answer)
        print("耗时：", round(result.total_seconds, 3), "秒")

    connection_ids = {result.connection_id for result in results}
    request_ids = {result.request_id for result in results}

    print("\n连接 ID 是否全部不同：", len(connection_ids) == 3)
    print("请求 ID 是否全部不同：", len(request_ids) == 3)
    print("整体并发耗时：", round(overall_seconds, 3), "秒")

    print("\n本步数据流：")
    print("3 个 asyncio.Task")
    print("→ 3 条独立 WebSocket")
    print("→ 3 个 websocket_handler() 协程")
    print("→ 各自独立 connection_id/client_name/request_count")
    print("→ 各自独立 request_id 和模型流")


if __name__ == "__main__":
    asyncio.run(main())
