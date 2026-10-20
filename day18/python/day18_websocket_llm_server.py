"""Day18 第二步（服务端）：通过 WebSocket 推送模型流。

运行方式：
    python day18/python/day18_websocket_llm_server.py

WebSocket 地址：
    ws://127.0.0.1:8011/ws

客户端消息：
    {"type": "ask", "question": "什么是 WebSocket？"}

服务端消息：
    {"type": "start", ...}
    {"type": "token", "text": "..."}
    {"type": "done", ...}
    {"type": "error", ...}
"""

from __future__ import annotations

import json
import os
from typing import Any

from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()


HOST = "127.0.0.1"
PORT = 8011


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


async def stream_answer(
    websocket: web.WebSocketResponse,
    question: str,
) -> None:
    """调用模型并把每个文本块转换成 WebSocket JSON 消息。"""

    await websocket.send_json(
        {
            "type": "start",
            "model": llm.model_name,
            "question": question,
        }
    )

    messages = [
        SystemMessage(
            content="你是一名简洁、准确的 Python 编程学习助手。"
        ),
        HumanMessage(content=question),
    ]

    chunk_count = 0
    answer_length = 0

    try:
        async for chunk in llm.astream(messages):
            if isinstance(chunk.content, str):
                text = chunk.content
            else:
                text = str(chunk.content)

            if not text:
                continue

            chunk_count += 1
            answer_length += len(text)

            await websocket.send_json(
                {
                    "type": "token",
                    "index": chunk_count,
                    "text": text,
                }
            )

        await websocket.send_json(
            {
                "type": "done",
                "chunk_count": chunk_count,
                "answer_length": answer_length,
            }
        )

    except Exception as error:
        await websocket.send_json(
            {
                "type": "error",
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """接收 ask 消息并在当前连接中发送模型流。"""

    websocket = web.WebSocketResponse(
        heartbeat=20,
        receive_timeout=120,
    )
    await websocket.prepare(request)

    print("[WebSocket 已连接]")

    await websocket.send_json(
        {
            "type": "connected",
            "model": llm.model_name,
            "message": "请发送 ask 消息。",
        }
    )

    try:
        async for message in websocket:
            if message.type == WSMsgType.TEXT:
                try:
                    data: Any = json.loads(message.data)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error_type": "InvalidJSON",
                            "message": "消息必须是 JSON。",
                        }
                    )
                    continue

                if not isinstance(data, dict):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error_type": "InvalidMessage",
                            "message": "JSON 顶层必须是 object。",
                        }
                    )
                    continue

                if data.get("type") != "ask":
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error_type": "UnknownType",
                            "message": "当前只支持 ask 消息。",
                        }
                    )
                    continue

                question = str(data.get("question", "")).strip()
                if not question:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error_type": "InvalidQuestion",
                            "message": "question 不能为空。",
                        }
                    )
                    continue

                print(f"[开始模型流] question={question!r}")
                await stream_answer(websocket, question)
                print("[模型流结束]")

            elif message.type == WSMsgType.ERROR:
                print("[WebSocket 错误]", websocket.exception())
                break

    finally:
        print("[WebSocket 已断开]")
        if not websocket.closed:
            await websocket.close()

    return websocket


async def health_handler(request: web.Request) -> web.Response:
    """返回服务健康状态。"""

    return web.json_response(
        {
            "status": "ok",
            "service": "day18-websocket-llm-server",
            "model": llm.model_name,
        }
    )


def create_app() -> web.Application:
    """创建 aiohttp 应用。"""

    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/ws", websocket_handler)
    return app


def main() -> None:
    """启动模型流 WebSocket Server。"""

    print("Day18 WebSocket 模型流服务正在启动。")
    print(f"WebSocket 地址：ws://{HOST}:{PORT}/ws")
    print("按 Ctrl+C 停止服务。")

    web.run_app(
        create_app(),
        host=HOST,
        port=PORT,
        print=None,
    )


if __name__ == "__main__":
    main()
