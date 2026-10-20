"""Day21 最终版：文档上传、增量索引、检索守门与流式Web RAG。"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from day21_browser_retrieval_server import search_knowledge
from day21_vector_index import DATABASE_PATH, sync_command


load_dotenv()
HOST, PORT = "127.0.0.1", 8028
MAX_UPLOAD_SIZE = 5 * 1024 * 1024
DAY21_DIR = Path(__file__).resolve().parents[1]
HTML = DAY21_DIR / "html" / "day21_final.html"
JS = DAY21_DIR / "javascript" / "day21_final.js"
CSS = DAY21_DIR / "css" / "day21_final.css"
UPLOAD_DIR = DAY21_DIR / "uploads_validated"

ALLOWED_MIME = {
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {"text/plain", "text/markdown", "application/octet-stream"},
    ".pdf": {"application/pdf", "application/octet-stream"},
}

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"), api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"), temperature=0,
)
index_lock = asyncio.Lock()
model_semaphore = asyncio.Semaphore(2)


class AskPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str | None = Field(None, max_length=2000)


class ClientCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["1.0"]
    type: Literal["ask", "cancel"]
    request_id: str
    payload: AskPayload = Field(default_factory=AskPayload)


@dataclass
class ConnectionState:
    active_request_id: str | None = None
    active_task: asyncio.Task[None] | None = None
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def send(ws: web.WebSocketResponse, state: ConnectionState,
               kind: str, request_id: str | None, payload: dict[str, Any]) -> None:
    async with state.send_lock:
        if not ws.closed:
            await ws.send_json({"type": kind, "request_id": request_id, "payload": payload})


def validate_content(data: bytes, suffix: str) -> tuple[str, int | None]:
    if suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise web.HTTPBadRequest(text="扩展名是PDF，但内容没有PDF文件头。")
        return "pdf", None
    if b"\x00" in data:
        raise web.HTTPBadRequest(text="文本文件不能包含NUL字节。")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise web.HTTPBadRequest(text="文本必须使用UTF-8编码。") from error
    if not text.strip():
        raise web.HTTPBadRequest(text="文档不能是空内容。")
    return "utf-8-text", len(text)


async def upload_handler(request: web.Request) -> web.Response:
    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "document" or not field.filename:
        raise web.HTTPBadRequest(text="缺少document文件字段。")

    suffix = Path(field.filename).suffix.lower()
    mime = (field.headers.get("Content-Type") or "").lower()
    if suffix not in ALLOWED_MIME or mime not in ALLOWED_MIME[suffix]:
        raise web.HTTPUnsupportedMediaType(text="只允许TXT、Markdown和PDF。")

    data_parts: list[bytes] = []
    size = 0
    while True:
        chunk = await field.read_chunk(64 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_UPLOAD_SIZE:
            raise web.HTTPRequestEntityTooLarge(MAX_UPLOAD_SIZE, size)
        data_parts.append(chunk)
    if size == 0:
        raise web.HTTPBadRequest(text="不能上传空文件。")

    data = b"".join(data_parts)
    detected, characters = validate_content(data, suffix)
    digest = hashlib.sha256(data).hexdigest()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    final_path = UPLOAD_DIR / f"sha256-{digest[:24]}{suffix}"
    duplicate = final_path.exists()

    if not duplicate:
        with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, delete=False) as temp:
            temp.write(data)
            temporary_path = Path(temp.name)
        os.replace(temporary_path, final_path)

    # 同步期间检索必须等待，避免读到正在替换的索引。
    async with index_lock:
        await asyncio.to_thread(sync_command)

    return web.json_response({
        "status": "duplicate" if duplicate else "indexed",
        "original_name": Path(field.filename).name,
        "stored_name": final_path.name,
        "sha256": digest,
        "size_bytes": size,
        "detected_content": detected,
        "character_count": characters,
        "duplicate": duplicate,
        "index_database": str(DATABASE_PATH),
    }, status=200 if duplicate else 201)


def context_from_results(results: list[dict[str, Any]]) -> str:
    blocks = []
    for number, item in enumerate(results, start=1):
        blocks.append(f"[资料{number}｜{item['location']}]\n{item['content']}")
    return "\n\n".join(blocks)


async def run_rag(ws: web.WebSocketResponse, state: ConnectionState,
                  request_id: str, question: str) -> None:
    acquired = False
    parts: list[str] = []
    try:
        await send(ws, state, "retrieving", request_id, {})
        async with index_lock:
            retrieval = await asyncio.to_thread(search_knowledge, question)

        await send(ws, state, "retrieval", request_id, retrieval)
        if not retrieval["allowed"]:
            await send(ws, state, "rejected", request_id, {
                "message": "知识库中没有找到足够相关的资料，拒绝让模型凭常识猜测。",
                "best_distance": retrieval["best_distance"],
                "threshold": retrieval["threshold"],
            })
            return

        await send(ws, state, "queued", request_id, {})
        await model_semaphore.acquire()
        acquired = True
        context = context_from_results(retrieval["results"][:4])
        messages = [
            SystemMessage(content=(
                "你是严格的RAG助手。只能根据参考资料回答；"
                "每个关键结论使用[资料1]格式引用。"
                "资料不足时明确说资料不足，不要使用外部常识补充。"
            )),
            HumanMessage(content=f"参考资料：\n{context}\n\n用户问题：{question}"),
        ]
        await send(ws, state, "start", request_id, {"model": llm.model_name})
        async for chunk in llm.astream(messages):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                parts.append(text)
                await send(ws, state, "token", request_id, {"text": text})
        await send(ws, state, "done", request_id, {
            "answer": "".join(parts),
            "sources": retrieval["results"][:4],
        })
    except asyncio.CancelledError:
        await send(ws, state, "cancelled", request_id, {"partial_answer": "".join(parts)})
    except Exception as error:
        await send(ws, state, "error", request_id,
                   {"error_type": type(error).__name__, "message": str(error)})
    finally:
        if acquired:
            model_semaphore.release()
        if state.active_request_id == request_id:
            state.active_request_id = None
            state.active_task = None


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    state = ConnectionState()
    await send(ws, state, "connected", None, {"model": llm.model_name})
    try:
        async for raw in ws:
            if raw.type != WSMsgType.TEXT:
                continue
            try:
                command = ClientCommand.model_validate_json(raw.data)
            except ValidationError as error:
                await send(ws, state, "error", None,
                           {"details": error.errors(include_url=False, include_input=False)})
                continue
            if command.type == "cancel":
                if state.active_task and not state.active_task.done() \
                        and command.request_id == state.active_request_id:
                    state.active_task.cancel()
                else:
                    await send(ws, state, "cancel_not_found", command.request_id, {})
                continue
            question = (command.payload.question or "").strip()
            if not question:
                await send(ws, state, "error", command.request_id,
                           {"message": "问题不能为空。"})
            elif state.active_task and not state.active_task.done():
                await send(ws, state, "busy", command.request_id, {})
            else:
                state.active_request_id = command.request_id
                state.active_task = asyncio.create_task(
                    run_rag(ws, state, command.request_id, question),
                    name=f"rag-{command.request_id}",
                )
    finally:
        if state.active_task and not state.active_task.done():
            state.active_task.cancel()
            await asyncio.gather(state.active_task, return_exceptions=True)
    return ws


async def file_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse({"/": HTML, "/app.js": JS, "/app.css": CSS}[request.path])


@web.middleware
async def errors(request: web.Request, handler: Any) -> web.StreamResponse:
    try:
        return await handler(request)
    except web.HTTPException as error:
        return web.json_response({"status": "error", "message": error.text}, status=error.status)


def create_app() -> web.Application:
    app = web.Application(middlewares=[errors], client_max_size=MAX_UPLOAD_SIZE + 65536)
    app.router.add_get("/", file_handler)
    app.router.add_get("/app.js", file_handler)
    app.router.add_get("/app.css", file_handler)
    app.router.add_post("/upload", upload_handler)
    app.router.add_get("/ws", websocket_handler)
    return app


if __name__ == "__main__":
    print(f"Day21 最终Web RAG：http://{HOST}:{PORT}")
    print("向量数据库：", DATABASE_PATH)
    web.run_app(create_app(), host=HOST, port=PORT, print=None)
