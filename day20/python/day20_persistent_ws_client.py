"""Day20 第三步：测试 SQLite 持久化 WebSocket 服务的终端客户端。

先启动服务端，再运行：
    python day20/python/day20_persistent_ws_client.py

也可指定标识：
    python day20/python/day20_persistent_ws_client.py xiaolin day20-main
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from typing import Any

from aiohttp import ClientSession, WSMsgType


HOST = "127.0.0.1"
PORT = 8022


async def receive_one_answer(websocket: Any) -> None:
    """持续读取一轮 start/token/done 或 error。"""

    while True:
        raw_message = await websocket.receive()
        if raw_message.type != WSMsgType.TEXT:
            raise ConnectionError("WebSocket 已关闭或发生错误。")

        message = raw_message.json()
        event_type = message.get("type")
        payload = message.get("payload", {})

        if event_type == "start":
            print(
                f"[模型开始，加载历史 {payload['loaded_history_messages']} 条]\n"
                "助手：",
                end="",
                flush=True,
            )
        elif event_type == "token":
            print(payload.get("text", ""), end="", flush=True)
        elif event_type == "done":
            print("\n")
            print("[已写入数据库]")
            print("消息ID：", payload["saved_message_ids"])
            print("数据库消息总数：", payload["database_message_count"])
            return
        elif event_type == "error":
            print("\n[错误]", payload)
            return


async def main() -> None:
    client_id = sys.argv[1] if len(sys.argv) > 1 else "xiaolin"
    session_id = sys.argv[2] if len(sys.argv) > 2 else "day20-main"
    url = (
        f"ws://{HOST}:{PORT}/ws"
        f"?client_id={client_id}&session_id={session_id}"
    )

    print("Day20 持久化聊天客户端")
    print("client_id：", client_id)
    print("session_id：", session_id)
    print("输入 exit 结束。")

    async with ClientSession() as http_session:
        async with http_session.ws_connect(url, heartbeat=20) as websocket:
            connected: dict[str, Any] = await websocket.receive_json()
            payload = connected["payload"]

            print("\n连接结果：", payload["session_action"])
            print("数据库：", payload["database"])
            print("恢复的历史消息：")
            if not payload["history"]:
                print("（暂无历史）")
            for message in payload["history"]:
                role = "你" if message["role"] == "user" else "助手"
                print(f"{role}：{message['content']}")

            while True:
                question = (await asyncio.to_thread(input, "\n你：")).strip()
                if question.lower() == "exit":
                    break
                if not question:
                    continue

                request_id = "req-" + uuid.uuid4().hex[:12]
                await websocket.send_json(
                    {
                        "request_id": request_id,
                        "question": question,
                    }
                )
                await receive_one_answer(websocket)

    print("\n聊天结束。再次连接同一 session_id 可恢复历史。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n用户终止客户端。")
