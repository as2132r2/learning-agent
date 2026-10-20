"""Day20 最终版：SQLite 持久化、多会话浏览器流式聊天。"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from day20_final_repository import FinalRepository, TurnRecord


load_dotenv()
HOST, PORT = "127.0.0.1", 8024
MAX_CONCURRENCY = 2
ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "html" / "day20_final.html"
JS = ROOT / "javascript" / "day20_final.js"
CSS = ROOT / "css" / "day20_final.css"

repository = FinalRepository()
repository.initialize()
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"), api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"), temperature=0,
)


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str | None = Field(None, max_length=4000)
    title: str | None = Field(None, max_length=100)
    session_id: str | None = Field(None, max_length=80)


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["1.0"]
    type: Literal["create_session", "switch_session", "list_sessions",
                  "ask", "cancel", "status", "ping"]
    request_id: str | None = None
    payload: Payload = Field(default_factory=Payload)


@dataclass
class ConnectionState:
    client_id: str
    current_session_id: str | None = None
    active_request_id: str | None = None
    active_task: asyncio.Task[None] | None = None
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def valid_id(value: str) -> bool:
    return 1 <= len(value) <= 80 and all(c.isalnum() or c in "-_" for c in value)


def model_history(turns: list[TurnRecord]) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for turn in turns:
        result.extend([HumanMessage(content=turn.question), AIMessage(content=turn.answer)])
    return result


async def send(ws: web.WebSocketResponse, state: ConnectionState,
               kind: str, request_id: str | None, payload: dict[str, Any]) -> None:
    async with state.send_lock:
        if not ws.closed:
            await ws.send_json({"version": "1.0", "type": kind,
                                "request_id": request_id, "payload": payload})


async def snapshots(state: ConnectionState) -> dict[str, Any]:
    sessions = await asyncio.to_thread(repository.list_sessions, state.client_id)
    turns: list[TurnRecord] = []
    if state.current_session_id:
        turns = await asyncio.to_thread(repository.load_turns, state.current_session_id, 30)
    return {"current_session_id": state.current_session_id,
            "sessions": [asdict(x) for x in sessions],
            "turns": [asdict(x) for x in turns]}


async def run_model(ws: web.WebSocketResponse, state: ConnectionState,
                    request_id: str, question: str, session_id: str) -> None:
    acquired = False
    parts: list[str] = []
    try:
        duplicate = await asyncio.to_thread(repository.get_turn_by_request, request_id)
        if duplicate:
            await send(ws, state, "duplicate", request_id, {"turn": asdict(duplicate)})
            return

        await send(ws, state, "queued", request_id, {})
        await semaphore.acquire()
        acquired = True
        turns = await asyncio.to_thread(repository.load_turns, session_id, 6)
        messages: list[BaseMessage] = [
            SystemMessage(content="你是一名简洁、准确的编程学习助手。"),
            *model_history(turns), HumanMessage(content=question),
        ]
        await send(ws, state, "start", request_id, {"history_turns": len(turns)})
        async for chunk in llm.astream(messages):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                parts.append(text)
                await send(ws, state, "token", request_id, {"text": text})
        created, turn = await asyncio.to_thread(
            repository.save_turn, request_id, session_id, question, "".join(parts)
        )
        await send(ws, state, "done", request_id,
                   {"created": created, "turn": asdict(turn)})
    except asyncio.CancelledError:
        await send(ws, state, "cancelled", request_id,
                   {"partial_answer": "".join(parts), "saved": False})
    except Exception as error:
        await send(ws, state, "error", request_id,
                   {"error_type": type(error).__name__, "message": str(error)})
    finally:
        if acquired:
            semaphore.release()
        if state.active_request_id == request_id:
            state.active_request_id = None
            state.active_task = None


async def handle(ws: web.WebSocketResponse, state: ConnectionState, raw: str) -> None:
    try:
        command = Command.model_validate_json(raw)
    except ValidationError as error:
        await send(ws, state, "error", None,
                   {"details": error.errors(include_url=False, include_input=False)})
        return

    if command.type == "create_session":
        title = (command.payload.title or "新对话").strip() or "新对话"
        session = await asyncio.to_thread(
            repository.create_session, "session-" + uuid.uuid4().hex[:12],
            state.client_id, title
        )
        state.current_session_id = session.session_id
        await send(ws, state, "state", command.request_id, await snapshots(state))

    elif command.type == "switch_session":
        target = command.payload.session_id or ""
        session = await asyncio.to_thread(repository.get_session, target)
        if not session or session.client_id != state.client_id:
            await send(ws, state, "error", command.request_id,
                       {"message": "会话不存在或不属于当前客户端。"})
        else:
            state.current_session_id = target
            await send(ws, state, "state", command.request_id, await snapshots(state))

    elif command.type == "list_sessions":
        await send(ws, state, "state", command.request_id, await snapshots(state))

    elif command.type == "ask":
        question = (command.payload.question or "").strip()
        if not command.request_id or not question or not state.current_session_id:
            await send(ws, state, "error", command.request_id,
                       {"message": "请先选择会话，并提交非空问题。"})
        elif state.active_task and not state.active_task.done():
            await send(ws, state, "busy", command.request_id,
                       {"active_request_id": state.active_request_id})
        else:
            state.active_request_id = command.request_id
            state.active_task = asyncio.create_task(
                run_model(ws, state, command.request_id, question,
                          state.current_session_id),
                name=f"day20-{command.request_id}",
            )

    elif command.type == "cancel":
        if state.active_task and not state.active_task.done() \
                and command.request_id == state.active_request_id:
            state.active_task.cancel()
        else:
            await send(ws, state, "cancel_not_found", command.request_id, {})

    elif command.type == "status":
        await send(ws, state, "status", command.request_id,
                   {"client_id": state.client_id,
                    "current_session_id": state.current_session_id,
                    "active_request_id": state.active_request_id})
    else:
        await send(ws, state, "pong", command.request_id, {})


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    client_id = request.query.get("client_id", "").strip()
    if not valid_id(client_id):
        await ws.close(code=1008, message=b"invalid client id")
        return ws
    sessions = await asyncio.to_thread(repository.list_sessions, client_id)
    state = ConnectionState(client_id, sessions[0].session_id if sessions else None)
    await send(ws, state, "connected", None,
               {"client_id": client_id, **await snapshots(state)})
    try:
        async for message in ws:
            if message.type == WSMsgType.TEXT:
                await handle(ws, state, message.data)
            elif message.type == WSMsgType.ERROR:
                break
    finally:
        if state.active_task and not state.active_task.done():
            state.active_task.cancel()
            await asyncio.gather(state.active_task, return_exceptions=True)
    return ws


async def file_handler(request: web.Request) -> web.FileResponse:
    files = {"/": HTML, "/app.js": JS, "/app.css": CSS}
    return web.FileResponse(files[request.path])


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", file_handler)
    app.router.add_get("/app.js", file_handler)
    app.router.add_get("/app.css", file_handler)
    app.router.add_get("/ws", websocket_handler)
    return app


if __name__ == "__main__":
    print(f"Day20 最终服务：http://{HOST}:{PORT}")
    print("数据库：", repository.path)
    web.run_app(create_app(), host=HOST, port=PORT, print=None)
