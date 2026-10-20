"""Day18 第五步（服务端）：同一 WebSocket 上接收 cancel 并取消模型流。

运行方式：
    python day18/python/day18_websocket_cancel_server.py

地址：ws://127.0.0.1:8014/ws

关键改变：
    旧版：await stream_answer() 会阻塞当前连接的接收循环。
    本版：create_task(stream_answer()) 后立即返回接收循环，因而可以继续
          接收 cancel 消息。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import Any

from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()


HOST = "127.0.0.1"
PORT = 8014
PROTOCOL_VERSION = "1.0"


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


async def send_message(
    websocket: web.WebSocketResponse,
    send_lock: asyncio.Lock,
    *,
    message_type: str,
    request_id: str | None,
    payload: dict[str, Any],
) -> None:
    """使用 Lock 避免多个模型 Task 同时写同一 WebSocket。"""

    async with send_lock:
        if websocket.closed:
            return

        await websocket.send_json(
            {
                "version": PROTOCOL_VERSION,
                "type": message_type,
                "request_id": request_id,
                "payload": payload,
            }
        )


async def run_model_request(
    websocket: web.WebSocketResponse,
    send_lock: asyncio.Lock,
    active_tasks: dict[str, asyncio.Task[None]],
    *,
    request_id: str,
    question: str,
) -> None:
    """作为独立 Task 运行模型流，并响应取消。"""

    chunk_count = 0
    text_length = 0

    try:
        await send_message(
            websocket,
            send_lock,
            message_type="start",
            request_id=request_id,
            payload={"model": llm.model_name},
        )

        messages = [
            SystemMessage(
                content="你是一名简洁、准确的 Python 编程学习助手。"
            ),
            HumanMessage(content=question),
        ]

        async for chunk in llm.astream(messages):
            text = (
                chunk.content
                if isinstance(chunk.content, str)
                else str(chunk.content)
            )
            if not text:
                continue

            chunk_count += 1
            text_length += len(text)

            await send_message(
                websocket,
                send_lock,
                message_type="token",
                request_id=request_id,
                payload={
                    "index": chunk_count,
                    "text": text,
                },
            )

        await send_message(
            websocket,
            send_lock,
            message_type="done",
            request_id=request_id,
            payload={
                "chunk_count": chunk_count,
                "answer_length": text_length,
            },
        )

    except asyncio.CancelledError:
        print(
            f"[模型任务取消] request_id={request_id} "
            f"chunks={chunk_count}"
        )

        # 告诉客户端取消已经真正作用到模型 Task，而不只是收到指令。
        await send_message(
            websocket,
            send_lock,
            message_type="cancelled",
            request_id=request_id,
            payload={
                "chunk_count_before_cancel": chunk_count,
                "answer_length_before_cancel": text_length,
            },
        )

        # 此处不再重新抛出，让 Task 以已清理的正常状态结束。
        return

    except Exception as error:
        await send_message(
            websocket,
            send_lock,
            message_type="error",
            request_id=request_id,
            payload={
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )

    finally:
        active_tasks.pop(request_id, None)
        print(
            f"[任务清理] request_id={request_id} "
            f"remaining={len(active_tasks)}"
        )


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """持续接收 ask/cancel，模型任务在后台独立运行。"""

    websocket = web.WebSocketResponse(heartbeat=20)
    await websocket.prepare(request)

    send_lock = asyncio.Lock()
    active_tasks: dict[str, asyncio.Task[None]] = {}

    await send_message(
        websocket,
        send_lock,
        message_type="connected",
        request_id=None,
        payload={"supported_types": ["ask", "cancel"]},
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
                    send_lock,
                    message_type="error",
                    request_id=None,
                    payload={"message": "消息必须是 JSON。"},
                )
                continue

            if not isinstance(data, dict):
                continue

            message_type = data.get("type")
            request_id = data.get("request_id")
            payload = data.get("payload", {})

            if not isinstance(request_id, str):
                await send_message(
                    websocket,
                    send_lock,
                    message_type="error",
                    request_id=None,
                    payload={"message": "request_id 必须是字符串。"},
                )
                continue

            if not isinstance(payload, dict):
                continue

            if message_type == "ask":
                question = str(payload.get("question", "")).strip()

                if not question:
                    await send_message(
                        websocket,
                        send_lock,
                        message_type="error",
                        request_id=request_id,
                        payload={"message": "question 不能为空。"},
                    )
                    continue

                if request_id in active_tasks:
                    await send_message(
                        websocket,
                        send_lock,
                        message_type="error",
                        request_id=request_id,
                        payload={"message": "request_id 正在运行。"},
                    )
                    continue

                task = asyncio.create_task(
                    run_model_request(
                        websocket,
                        send_lock,
                        active_tasks,
                        request_id=request_id,
                        question=question,
                    ),
                    name=f"model-{request_id}",
                )
                active_tasks[request_id] = task

                print(
                    f"[创建模型任务] request_id={request_id} "
                    f"active={len(active_tasks)}"
                )

            elif message_type == "cancel":
                task = active_tasks.get(request_id)

                if task is None or task.done():
                    await send_message(
                        websocket,
                        send_lock,
                        message_type="cancel_not_found",
                        request_id=request_id,
                        payload={
                            "message": "没有找到正在运行的请求。"
                        },
                    )
                    continue

                await send_message(
                    websocket,
                    send_lock,
                    message_type="cancel_requested",
                    request_id=request_id,
                    payload={"message": "已收到取消指令。"},
                )

                task.cancel()

            else:
                await send_message(
                    websocket,
                    send_lock,
                    message_type="error",
                    request_id=request_id,
                    payload={"message": "只支持 ask 和 cancel。"},
                )

    finally:
        print(f"[连接关闭，取消任务] count={len(active_tasks)}")

        tasks_to_stop = list(active_tasks.values())
        for task in tasks_to_stop:
            task.cancel()

        if tasks_to_stop:
            await asyncio.gather(
                *tasks_to_stop,
                return_exceptions=True,
            )

        if not websocket.closed:
            await websocket.close()

    return websocket


def create_app() -> web.Application:
    """创建可取消模型流服务。"""

    app = web.Application()
    app.router.add_get("/ws", websocket_handler)
    return app


def main() -> None:
    """启动服务。"""

    print("Day18 可取消 WebSocket 模型服务正在启动。")
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
