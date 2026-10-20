"""Day19 第二步：向浏览器分块发送文本，练习 DOM 流式更新。

运行：
    python day19/python/day19_dom_stream_server.py

浏览器访问：
    http://127.0.0.1:8017
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from aiohttp import WSMsgType, web


HOST = "127.0.0.1"
PORT = 8017
DAY19_DIR = Path(__file__).resolve().parents[1]
HTML_FILE = DAY19_DIR / "html" / "day19_dom_stream.html"


async def index_handler(request: web.Request) -> web.FileResponse:
    """返回本步骤的完整 HTML 页面。"""

    return web.FileResponse(HTML_FILE)


async def send_simulated_stream(
    websocket: web.WebSocketResponse,
    request_id: str,
    question: str,
) -> None:
    """把模拟回答拆成多个 token 事件逐块发送。"""

    answer_parts = [
        "我已经收到你的问题：",
        f"“{question}”。",
        "\n\n这一步还没有调用大模型，",
        "而是由 Python 服务端分块发送固定回答，",
        "浏览器收到每个文本块后，",
        "持续更新同一个 DOM 元素。",
    ]

    await websocket.send_json(
        {
            "type": "start",
            "request_id": request_id,
            "payload": {},
        }
    )

    for index, text in enumerate(answer_parts, start=1):
        await asyncio.sleep(0.35)
        await websocket.send_json(
            {
                "type": "token",
                "request_id": request_id,
                "payload": {
                    "index": index,
                    "text": text,
                },
            }
        )

    await websocket.send_json(
        {
            "type": "done",
            "request_id": request_id,
            "payload": {
                "chunk_count": len(answer_parts),
            },
        }
    )


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """接收浏览器问题，并按 start/token/done 协议返回模拟流。"""

    websocket = web.WebSocketResponse(heartbeat=20)
    await websocket.prepare(request)

    connection_id = "conn-" + uuid.uuid4().hex[:10]
    print(f"[连接建立] {connection_id}")

    await websocket.send_json(
        {
            "type": "connected",
            "request_id": None,
            "payload": {"connection_id": connection_id},
        }
    )

    try:
        async for message in websocket:
            if message.type == WSMsgType.TEXT:
                try:
                    data = message.json()
                except ValueError:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "request_id": None,
                            "payload": {"message": "消息必须是 JSON。"},
                        }
                    )
                    continue

                request_id = str(data.get("request_id", "")).strip()
                question = str(data.get("question", "")).strip()

                if not request_id or not question:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "request_id": request_id or None,
                            "payload": {"message": "request_id 和 question 不能为空。"},
                        }
                    )
                    continue

                print(f"[开始模拟流] {request_id} | {question!r}")
                await send_simulated_stream(websocket, request_id, question)
                print(f"[模拟流完成] {request_id}")

            elif message.type == WSMsgType.ERROR:
                print("[WebSocket 错误]", websocket.exception())
                break
    finally:
        print(f"[连接断开] {connection_id}")

    return websocket


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", websocket_handler)
    return app


def main() -> None:
    print("Day19 DOM 流式更新服务已启动。")
    print(f"浏览器页面：http://{HOST}:{PORT}")
    print(f"WebSocket：ws://{HOST}:{PORT}/ws")
    print("按 Ctrl+C 停止服务。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
