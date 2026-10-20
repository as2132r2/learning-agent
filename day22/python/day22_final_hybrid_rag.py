"""Day22 第六步：混合检索、RRF、Reranker与守门的最终Web RAG。"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from day22_fts_basic import (
    DATABASE_PATH as FTS_DATABASE_PATH,
    open_database as open_fts_database,
    rebuild_index as rebuild_fts_index,
)
from day22_llm_reranker import RerankedResult, rerank_candidates
from day22_retrieval_compare import fts_search, vector_search
from day22_rrf_fusion import RRFResult, reciprocal_rank_fusion


PROJECT_DIR = Path(__file__).resolve().parents[2]
DAY21_DIR = PROJECT_DIR / "day21"
DAY22_DIR = PROJECT_DIR / "day22"
DAY21_PYTHON = DAY21_DIR / "python"
if str(DAY21_PYTHON) not in sys.path:
    sys.path.insert(0, str(DAY21_PYTHON))

from day21_vector_index import (  # noqa: E402
    DATABASE_PATH as VECTOR_DATABASE_PATH,
    sync_command as sync_vector_index,
)


load_dotenv()
HOST, PORT = "127.0.0.1", 8029
MAX_UPLOAD_SIZE = 5 * 1024 * 1024
RRF_CANDIDATE_COUNT = 5
RERANK_THRESHOLD = 95
MAX_CONTEXT_COUNT = 4

HTML = DAY22_DIR / "html" / "day22_final.html"
JS = DAY22_DIR / "javascript" / "day22_final.js"
CSS = DAY22_DIR / "css" / "day22_final.css"
UPLOAD_DIR = DAY21_DIR / "uploads_validated"

ALLOWED_MIME = {
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {"text/plain", "text/markdown", "application/octet-stream"},
    ".pdf": {"application/pdf", "application/octet-stream"},
}

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
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


async def send(
    ws: web.WebSocketResponse,
    state: ConnectionState,
    kind: str,
    request_id: str | None,
    payload: dict[str, Any],
) -> None:
    """串行发送统一格式的WebSocket事件。"""

    async with state.send_lock:
        if not ws.closed:
            await ws.send_json(
                {
                    "type": kind,
                    "request_id": request_id,
                    "payload": payload,
                }
            )


def validate_content(data: bytes, suffix: str) -> tuple[str, int | None]:
    """校验PDF文件头或UTF-8文本内容。"""

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


def sync_both_indexes() -> int:
    """同步向量索引后，重建FTS5关键词索引。"""

    sync_vector_index()
    connection = open_fts_database()
    try:
        return rebuild_fts_index(connection)
    finally:
        connection.close()


async def upload_handler(request: web.Request) -> web.Response:
    """接收文档，内容寻址保存，然后同步两套检索索引。"""

    reader = await request.multipart()
    part = await reader.next()
    if part is None or part.name != "document" or not part.filename:
        raise web.HTTPBadRequest(text="缺少document文件字段。")

    suffix = Path(part.filename).suffix.lower()
    mime = (part.headers.get("Content-Type") or "").lower()
    if suffix not in ALLOWED_MIME or mime not in ALLOWED_MIME[suffix]:
        raise web.HTTPUnsupportedMediaType(text="只允许TXT、Markdown和PDF。")

    data_parts: list[bytes] = []
    size = 0
    while True:
        chunk = await part.read_chunk(64 * 1024)
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

    async with index_lock:
        fts_chunk_count = await asyncio.to_thread(sync_both_indexes)

    return web.json_response(
        {
            "status": "duplicate" if duplicate else "indexed",
            "original_name": Path(part.filename).name,
            "stored_name": final_path.name,
            "sha256": digest,
            "size_bytes": size,
            "detected_content": detected,
            "character_count": characters,
            "duplicate": duplicate,
            "vector_database": str(VECTOR_DATABASE_PATH),
            "fts_database": str(FTS_DATABASE_PATH),
            "fts_chunk_count": fts_chunk_count,
        },
        status=200 if duplicate else 201,
    )


def retrieve_rrf_candidates(question: str) -> tuple[list[RRFResult], dict[str, Any]]:
    """执行两路本地检索并返回RRF前五名及诊断数据。"""

    vector_results = vector_search(question)
    terms, match_query, keyword_results = fts_search(question)
    rrf_results = reciprocal_rank_fusion(
        vector_results,
        keyword_results,
    )[:RRF_CANDIDATE_COUNT]

    diagnostics = {
        "terms": terms,
        "match_query": match_query,
        "vector_count": len(vector_results),
        "keyword_count": len(keyword_results),
    }
    return rrf_results, diagnostics


def rrf_to_dict(item: RRFResult, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": item.chunk_id,
        "location": item.location,
        "content": item.content,
        "vector_rank": item.vector_rank,
        "vector_distance": item.vector_distance,
        "keyword_rank": item.keyword_rank,
        "bm25_score": item.bm25_score,
        "rrf_score": round(item.rrf_score, 6),
    }


def reranked_to_dict(item: RerankedResult, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": item.chunk_id,
        "location": item.location,
        "content": item.content,
        "rrf_rank": item.rrf_rank,
        "rrf_score": round(item.rrf_score, 6),
        "relevance_score": item.relevance_score,
        "reason": item.reason,
    }


def context_from_results(results: list[RerankedResult]) -> str:
    """把通过守门的精排资料组织成生成模型的上下文。"""

    blocks: list[str] = []
    for number, item in enumerate(results, start=1):
        blocks.append(
            f"[资料{number}｜{item.location}｜相关性{item.relevance_score}]\n"
            f"{item.content}"
        )
    return "\n\n".join(blocks)


async def run_rag(
    ws: web.WebSocketResponse,
    state: ConnectionState,
    request_id: str,
    question: str,
) -> None:
    """运行本地混合检索、模型精排、守门和流式答案生成。"""

    acquired = False
    answer_parts: list[str] = []
    try:
        await send(ws, state, "retrieving", request_id, {})
        async with index_lock:
            rrf_results, diagnostics = await asyncio.to_thread(
                retrieve_rrf_candidates,
                question,
            )

        rrf_payload = [
            rrf_to_dict(item, rank)
            for rank, item in enumerate(rrf_results, start=1)
        ]
        await send(
            ws,
            state,
            "rrf_results",
            request_id,
            {"results": rrf_payload, **diagnostics},
        )

        await send(ws, state, "rerank_queued", request_id, {})
        await model_semaphore.acquire()
        acquired = True

        reranked_results, _ = await asyncio.to_thread(
            rerank_candidates,
            question,
            rrf_results,
        )
        reranked_payload = [
            reranked_to_dict(item, rank)
            for rank, item in enumerate(reranked_results, start=1)
        ]
        best_score = (
            reranked_results[0].relevance_score
            if reranked_results
            else 0
        )

        await send(
            ws,
            state,
            "reranked",
            request_id,
            {
                "results": reranked_payload,
                "best_score": best_score,
                "threshold": RERANK_THRESHOLD,
            },
        )

        if best_score < RERANK_THRESHOLD:
            await send(
                ws,
                state,
                "rejected",
                request_id,
                {
                    "message": (
                        "Reranker没有找到达到相关性阈值的资料，"
                        "拒绝让模型凭外部常识猜测。"
                    ),
                    "best_score": best_score,
                    "threshold": RERANK_THRESHOLD,
                },
            )
            return

        accepted_results = [
            item
            for item in reranked_results
            if item.relevance_score >= RERANK_THRESHOLD
        ][:MAX_CONTEXT_COUNT]
        context = context_from_results(accepted_results)

        messages = [
            SystemMessage(
                content=(
                    "你是严格的RAG助手。只能根据参考资料回答。"
                    "每个关键结论使用[资料1]格式引用。"
                    "资料不足时明确说明，不得使用外部常识补充。"
                )
            ),
            HumanMessage(
                content=f"参考资料：\n{context}\n\n用户问题：{question}"
            ),
        ]

        await send(
            ws,
            state,
            "start",
            request_id,
            {
                "model": llm.model_name,
                "accepted_count": len(accepted_results),
            },
        )
        async for chunk in llm.astream(messages):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                answer_parts.append(text)
                await send(
                    ws,
                    state,
                    "token",
                    request_id,
                    {"text": text},
                )

        accepted_payload = [
            reranked_to_dict(item, rank)
            for rank, item in enumerate(accepted_results, start=1)
        ]
        await send(
            ws,
            state,
            "done",
            request_id,
            {
                "answer": "".join(answer_parts),
                "sources": accepted_payload,
            },
        )
    except asyncio.CancelledError:
        await send(
            ws,
            state,
            "cancelled",
            request_id,
            {"partial_answer": "".join(answer_parts)},
        )
    except Exception as error:
        await send(
            ws,
            state,
            "error",
            request_id,
            {
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
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
    await send(
        ws,
        state,
        "connected",
        None,
        {"model": llm.model_name, "threshold": RERANK_THRESHOLD},
    )

    try:
        async for raw in ws:
            if raw.type != WSMsgType.TEXT:
                continue
            try:
                command = ClientCommand.model_validate_json(raw.data)
            except ValidationError as error:
                await send(
                    ws,
                    state,
                    "error",
                    None,
                    {
                        "details": error.errors(
                            include_url=False,
                            include_input=False,
                        )
                    },
                )
                continue

            if command.type == "cancel":
                if (
                    state.active_task
                    and not state.active_task.done()
                    and command.request_id == state.active_request_id
                ):
                    state.active_task.cancel()
                else:
                    await send(
                        ws,
                        state,
                        "cancel_not_found",
                        command.request_id,
                        {},
                    )
                continue

            question = (command.payload.question or "").strip()
            if not question:
                await send(
                    ws,
                    state,
                    "error",
                    command.request_id,
                    {"message": "问题不能为空。"},
                )
            elif state.active_task and not state.active_task.done():
                await send(
                    ws,
                    state,
                    "busy",
                    command.request_id,
                    {"message": "当前连接已有请求正在执行。"},
                )
            else:
                state.active_request_id = command.request_id
                state.active_task = asyncio.create_task(
                    run_rag(ws, state, command.request_id, question),
                    name=f"hybrid-rag-{command.request_id}",
                )
    finally:
        if state.active_task and not state.active_task.done():
            state.active_task.cancel()
            await asyncio.gather(
                state.active_task,
                return_exceptions=True,
            )
    return ws


async def file_handler(request: web.Request) -> web.FileResponse:
    files = {"/": HTML, "/app.js": JS, "/app.css": CSS}
    return web.FileResponse(files[request.path])


@web.middleware
async def errors(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    try:
        return await handler(request)
    except web.HTTPException as error:
        return web.json_response(
            {"status": "error", "message": error.text},
            status=error.status,
        )


def create_app() -> web.Application:
    app = web.Application(
        middlewares=[errors],
        client_max_size=MAX_UPLOAD_SIZE + 65536,
    )
    app.router.add_get("/", file_handler)
    app.router.add_get("/app.js", file_handler)
    app.router.add_get("/app.css", file_handler)
    app.router.add_post("/upload", upload_handler)
    app.router.add_get("/ws", websocket_handler)
    return app


if __name__ == "__main__":
    print(f"Day22 最终混合检索Web RAG：http://{HOST}:{PORT}")
    print("向量数据库：", VECTOR_DATABASE_PATH)
    print("FTS数据库：", FTS_DATABASE_PATH)
    print("Reranker守门阈值：", RERANK_THRESHOLD)
    web.run_app(create_app(), host=HOST, port=PORT, print=None)
