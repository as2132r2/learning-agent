"""Day19 第五步：支持心跳、断线清理与浏览器自动重连的服务端。

运行：python day19/python/day19_browser_reconnect_server.py
访问：http://127.0.0.1:8020
"""

from __future__ import annotations

import asyncio
import os
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
PORT = 8020
HTML_FILE = (
    Path(__file__).resolve().parents[1]
    / "html"
    / "day19_browser_reconnect.html"
)

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


@dataclass
class ConnectionState:
    connection_id: str
    active_request_id: str | None = None
    active_task: asyncio.Task[None] | None = None
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def send_event(
    websocket: web.WebSocketResponse,
    state: ConnectionState,
    event_type: str,
    request_id: str | None,
    payload: dict[str, Any],
) -> None:
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


async def run_model(
    websocket: web.WebSocketResponse,
    state: ConnectionState,
    request_id: str,
    question: str,
) -> None:
    text_parts: list[str] = []

    try:
        await send_event(
            websocket, state, "start", request_id, {"model": llm.model_name}
        )

        messages = [
            SystemMessage(content="你是一名简洁、友好的编程学习助手。"),
            HumanMessage(content=question),
        ]

        async for chunk in llm.astream(messages):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if not text:
                continue

            text_parts.append(text)
            await send_event(
                websocket,
                state,
                "token",
                request_id,
                {"text": text},
            )

        await send_event(
            websocket,
            state,
            "done",
            request_id,
            {"answer_length": len("".join(text_parts))},
        )

    except asyncio.CancelledError:
        await send_event(
            websocket,
            state,
            "cancelled",
            request_id,
            {"partial_answer": "".join(text_parts)},
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
) -> bool:
    """处理命令；返回 False 表示主动结束当前连接。"""

    command_type = str(data.get("type", ""))
    request_id = data.get("request_id")
    payload = data.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}

    if command_type == "ask":
        question = str(payload.get("question", "")).strip()
        if not isinstance(request_id, str) or not request_id or not question:
            await send_event(
                websocket, state, "error", None, {"message": "问题格式错误。"}
            )
        elif state.active_task is not None and not state.active_task.done():
            await send_event(
                websocket,
                state,
                "busy",
                request_id,
                {"active_request_id": state.active_request_id},
            )
        else:
            state.active_request_id = request_id
            state.active_task = asyncio.create_task(
                run_model(websocket, state, request_id, question),
                name=f"reconnect-model-{request_id}",
            )

    elif command_type == "cancel":
        if state.active_task is not None and not state.active_task.done():
            state.active_task.cancel()
        else:
            await send_event(websocket, state, "cancel_not_found", request_id, {})

    elif command_type == "ping":
        await send_event(
            websocket,
            state,
            "pong",
            request_id if isinstance(request_id, str) else None,
            {},
        )

    elif command_type == "disconnect":
        await send_event(websocket, state, "disconnecting", None, {})
        await websocket.close(code=1000, message=b"simulated disconnect")
        return False

    else:
        await send_event(
            websocket,
            state,
            "error",
            request_id if isinstance(request_id, str) else None,
            {"message": f"未知命令：{command_type!r}"},
        )

    return True


async def index_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(HTML_FILE)


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    websocket = web.WebSocketResponse(heartbeat=20)
    await websocket.prepare(request)

    state = ConnectionState(connection_id="conn-" + uuid.uuid4().hex[:10])
    print(f"[连接建立] {state.connection_id}")
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
                        raise ValueError("消息必须是 JSON 对象。")
                    should_continue = await handle_command(websocket, state, data)
                    if not should_continue:
                        break
                except (ValueError, TypeError) as error:
                    await send_event(
                        websocket, state, "error", None, {"message": str(error)}
                    )
            elif raw_message.type == WSMsgType.ERROR:
                break
    finally:
        if state.active_task is not None and not state.active_task.done():
            state.active_task.cancel()
            await asyncio.gather(state.active_task, return_exceptions=True)
        print(f"[连接清理完成] {state.connection_id}")

    return websocket


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", websocket_handler)
    return app


def main() -> None:
    print("Day19 自动重连练习服务已启动。")
    print(f"浏览器页面：http://{HOST}:{PORT}")
    print("按 Ctrl+C 停止服务。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
