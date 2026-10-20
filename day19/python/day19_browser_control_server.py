"""Day19 第四步：支持 ask、cancel、status 的浏览器模型流服务。

运行：python day19/python/day19_browser_control_server.py
访问：http://127.0.0.1:8019
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()

HOST = "127.0.0.1"
PORT = 8019
DAY19_DIR = Path(__file__).resolve().parents[1]
HTML_FILE = DAY19_DIR / "html" / "day19_browser_control.html"

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


@dataclass
class ConnectionState:
    """一条浏览器连接独享的运行状态。"""

    connection_id: str
    active_request_id: str | None = None
    active_task: asyncio.Task[None] | None = None
    completed_requests: int = 0
    cancelled_requests: int = 0
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def send_event(
    websocket: web.WebSocketResponse,
    state: ConnectionState,
    event_type: str,
    request_id: str | None,
    payload: dict[str, Any],
) -> None:
    """给同一连接串行发送统一格式的事件。"""

    async with state.send_lock:
        if not websocket.closed:
            await websocket.send_json(
                {
                    "type": event_type,
                    "connection_id": state.connection_id,
                    "request_id": request_id,
                    "payload": payload,
                }
            )


async def stream_model_answer(
    websocket: web.WebSocketResponse,
    state: ConnectionState,
    request_id: str,
    question: str,
) -> None:
    """后台运行模型流；支持被 active_task.cancel() 取消。"""

    started_at = time.perf_counter()
    text_parts: list[str] = []
    chunk_count = 0

    try:
        await send_event(
            websocket,
            state,
            "start",
            request_id,
            {"model": llm.model_name},
        )

        messages = [
            SystemMessage(content="你是一名简洁、友好的编程学习助手。"),
            HumanMessage(content=question),
        ]

        async for chunk in llm.astream(messages):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if not text:
                continue

            chunk_count += 1
            text_parts.append(text)
            await send_event(
                websocket,
                state,
                "token",
                request_id,
                {"index": chunk_count, "text": text},
            )

        complete_answer = "".join(text_parts)
        state.completed_requests += 1

        await send_event(
            websocket,
            state,
            "done",
            request_id,
            {
                "chunk_count": chunk_count,
                "answer_length": len(complete_answer),
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            },
        )

    except asyncio.CancelledError:
        state.cancelled_requests += 1
        await send_event(
            websocket,
            state,
            "cancelled",
            request_id,
            {
                "partial_answer": "".join(text_parts),
                "chunk_count": chunk_count,
            },
        )

    except Exception as error:
        await send_event(
            websocket,
            state,
            "error",
            request_id,
            {
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )

    finally:
        if state.active_request_id == request_id:
            state.active_request_id = None
            state.active_task = None


async def handle_command(
    websocket: web.WebSocketResponse,
    state: ConnectionState,
    data: dict[str, Any],
) -> None:
    """根据 type 路由 ask、cancel、status 三种浏览器命令。"""

    command_type = str(data.get("type", ""))
    request_id_value = data.get("request_id")
    request_id = str(request_id_value) if request_id_value is not None else None
    payload = data.get("payload", {})

    if not isinstance(payload, dict):
        await send_event(
            websocket, state, "error", request_id, {"message": "payload 必须是对象。"}
        )
        return

    if command_type == "ask":
        question = str(payload.get("question", "")).strip()

        if not request_id or not question:
            await send_event(
                websocket,
                state,
                "error",
                request_id,
                {"message": "ask 需要 request_id 和非空 question。"},
            )
            return

        if state.active_task is not None and not state.active_task.done():
            await send_event(
                websocket,
                state,
                "busy",
                request_id,
                {"active_request_id": state.active_request_id},
            )
            return

        state.active_request_id = request_id
        state.active_task = asyncio.create_task(
            stream_model_answer(websocket, state, request_id, question),
            name=f"browser-model-{request_id}",
        )

    elif command_type == "cancel":
        if (
            state.active_task is None
            or state.active_task.done()
            or request_id != state.active_request_id
        ):
            await send_event(websocket, state, "cancel_not_found", request_id, {})
            return

        await send_event(websocket, state, "cancel_requested", request_id, {})
        state.active_task.cancel()

    elif command_type == "status":
        await send_event(
            websocket,
            state,
            "status_result",
            request_id,
            {
                "active_request_id": state.active_request_id,
                "completed_requests": state.completed_requests,
                "cancelled_requests": state.cancelled_requests,
            },
        )

    else:
        await send_event(
            websocket,
            state,
            "error",
            request_id,
            {"message": f"未知命令类型：{command_type!r}"},
        )


async def index_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(HTML_FILE)


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    websocket = web.WebSocketResponse(heartbeat=20)
    await websocket.prepare(request)

    state = ConnectionState(connection_id="conn-" + uuid.uuid4().hex[:10])
    print(f"[浏览器已连接] {state.connection_id}")

    await send_event(
        websocket,
        state,
        "connected",
        None,
        {"model": llm.model_name},
    )

    try:
        async for raw_message in websocket:
            if raw_message.type == WSMsgType.TEXT:
                try:
                    data = raw_message.json()
                    if not isinstance(data, dict):
                        raise ValueError("消息最外层必须是 JSON 对象。")
                    await handle_command(websocket, state, data)
                except (ValueError, TypeError) as error:
                    await send_event(
                        websocket,
                        state,
                        "error",
                        None,
                        {"message": str(error)},
                    )
            elif raw_message.type == WSMsgType.ERROR:
                break
    finally:
        if state.active_task is not None and not state.active_task.done():
            state.active_task.cancel()
            await asyncio.gather(state.active_task, return_exceptions=True)
        print(f"[浏览器已断开] {state.connection_id}")

    return websocket


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", websocket_handler)
    return app


def main() -> None:
    print("Day19 浏览器控制版服务已启动。")
    print(f"浏览器页面：http://{HOST}:{PORT}")
    print("支持：提问、取消、状态查询。")
    print("按 Ctrl+C 停止服务。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
