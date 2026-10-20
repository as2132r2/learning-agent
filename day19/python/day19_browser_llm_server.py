"""Day19 第三步：把真实大模型流通过 WebSocket 发送给浏览器。

运行：
    python day19/python/day19_browser_llm_server.py

浏览器访问：
    http://127.0.0.1:8018
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()

HOST = "127.0.0.1"
PORT = 8018
DAY19_DIR = Path(__file__).resolve().parents[1]
HTML_FILE = DAY19_DIR / "html" / "day19_browser_llm.html"

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


async def index_handler(request: web.Request) -> web.FileResponse:
    """向浏览器返回本步骤的页面。"""

    return web.FileResponse(HTML_FILE)


async def stream_model_answer(
    websocket: web.WebSocketResponse,
    request_id: str,
    question: str,
) -> None:
    """调用模型，并把 AIMessageChunk 转成 WebSocket token 事件。"""

    started_at = time.perf_counter()
    text_parts: list[str] = []
    chunk_count = 0

    await websocket.send_json(
        {
            "type": "start",
            "request_id": request_id,
            "payload": {"model": llm.model_name},
        }
    )

    try:
        messages = [
            SystemMessage(
                content="你是一名简洁、友好的 Python 和大模型应用学习助手。"
            ),
            HumanMessage(content=question),
        ]

        async for chunk in llm.astream(messages):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if not text:
                continue

            chunk_count += 1
            text_parts.append(text)

            await websocket.send_json(
                {
                    "type": "token",
                    "request_id": request_id,
                    "payload": {
                        "index": chunk_count,
                        "text": text,
                    },
                }
            )

        complete_answer = "".join(text_parts)
        elapsed = time.perf_counter() - started_at

        await websocket.send_json(
            {
                "type": "done",
                "request_id": request_id,
                "payload": {
                    "chunk_count": chunk_count,
                    "answer_length": len(complete_answer),
                    "elapsed_seconds": round(elapsed, 3),
                },
            }
        )

        print(
            f"[模型完成] {request_id} | "
            f"chunks={chunk_count} | elapsed={elapsed:.3f}s"
        )

    except Exception as error:
        await websocket.send_json(
            {
                "type": "error",
                "request_id": request_id,
                "payload": {
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        print(f"[模型失败] {request_id} | {type(error).__name__}: {error}")


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """管理一条浏览器连接，并处理其模型问题。"""

    websocket = web.WebSocketResponse(heartbeat=20)
    await websocket.prepare(request)

    connection_id = "conn-" + uuid.uuid4().hex[:10]
    print(f"[浏览器已连接] {connection_id}")

    await websocket.send_json(
        {
            "type": "connected",
            "request_id": None,
            "payload": {
                "connection_id": connection_id,
                "model": llm.model_name,
            },
        }
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

                    print(f"[收到问题] {request_id} | {question!r}")
                    await stream_model_answer(
                        websocket,
                        request_id,
                        question,
                    )

                except (ValueError, TypeError) as error:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "request_id": None,
                            "payload": {
                                "error_type": type(error).__name__,
                                "message": str(error),
                            },
                        }
                    )

            elif raw_message.type == WSMsgType.ERROR:
                print("[WebSocket 错误]", websocket.exception())
                break
    finally:
        print(f"[浏览器已断开] {connection_id}")

    return websocket


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", websocket_handler)
    return app


def main() -> None:
    print("Day19 浏览器真实模型流服务已启动。")
    print(f"浏览器页面：http://{HOST}:{PORT}")
    print(f"模型：{llm.model_name}")
    print("按 Ctrl+C 停止服务。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
