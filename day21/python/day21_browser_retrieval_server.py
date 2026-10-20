"""Day21 第五步：浏览器向量检索、阈值守门和来源展示。

先建立索引：python day21/python/day21_vector_index.py sync
再运行服务：python day21/python/day21_browser_retrieval_server.py
浏览器访问：http://127.0.0.1:8027
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aiohttp import web
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from day21_vector_index import (
    DATABASE_PATH,
    LocalChineseEmbeddings,
    create_vector_store,
    open_connection,
)


HOST = "127.0.0.1"
PORT = 8027
DISTANCE_THRESHOLD = 1.05
TOP_K = 5

DAY21_DIR = Path(__file__).resolve().parents[1]
HTML_FILE = DAY21_DIR / "html" / "day21_browser_retrieval.html"

# 模型只在服务启动时加载一次；每次查询复用。
embedding_model = LocalChineseEmbeddings()


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=1000)


def format_location(metadata: dict[str, Any]) -> str:
    source = str(metadata.get("source", "未知来源"))
    page = metadata.get("page")
    return f"{source}，第 {page} 页" if page is not None else source


def search_knowledge(question: str) -> dict[str, Any]:
    """在线程中建立独立SQLite连接并执行向量查询。"""

    connection = open_connection()
    try:
        vector_store = create_vector_store(connection, embedding_model)
        matches = vector_store.similarity_search_with_score(question, k=TOP_K)

        results: list[dict[str, Any]] = []
        for rank, (document, distance) in enumerate(matches, start=1):
            numeric_distance = float(distance)
            results.append(
                {
                    "rank": rank,
                    "chunk_id": document.metadata.get("chunk_id", document.id),
                    "content": document.page_content,
                    "source": document.metadata.get("source"),
                    "title": document.metadata.get("title"),
                    "file_type": document.metadata.get("file_type"),
                    "page": document.metadata.get("page"),
                    "start_index": document.metadata.get("start_index"),
                    "location": format_location(document.metadata),
                    "distance": round(numeric_distance, 4),
                    "passes_threshold": numeric_distance <= DISTANCE_THRESHOLD,
                }
            )

        best_distance = results[0]["distance"] if results else None
        allowed = bool(results) and float(best_distance) <= DISTANCE_THRESHOLD

        return {
            "question": question,
            "allowed": allowed,
            "decision": "允许进入回答阶段" if allowed else "拒绝进入回答阶段",
            "threshold": DISTANCE_THRESHOLD,
            "best_distance": best_distance,
            "result_count": len(results),
            "results": results,
            "warning": (
                "阈值只能判断距离是否足够近，不能保证排名第一的内容一定正确。"
            ),
        }
    finally:
        connection.close()


async def index_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(HTML_FILE)


async def search_handler(request: web.Request) -> web.Response:
    if not DATABASE_PATH.exists():
        raise web.HTTPServiceUnavailable(
            text="向量数据库不存在，请先运行 day21_vector_index.py sync。"
        )

    try:
        raw_data = await request.json()
        search_request = SearchRequest.model_validate(raw_data)
    except (ValidationError, ValueError, TypeError) as error:
        raise web.HTTPBadRequest(text=f"请求格式错误：{error}") from error

    question = search_request.question.strip()
    if not question:
        raise web.HTTPBadRequest(text="问题不能为空白字符。")

    # Embedding和SQLite查询是同步工作，交给线程避免阻塞aiohttp事件循环。
    result = await asyncio.to_thread(search_knowledge, question)
    return web.json_response(result)


@web.middleware
async def json_error_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    try:
        return await handler(request)
    except web.HTTPException as error:
        return web.json_response(
            {
                "status": "error",
                "http_status": error.status,
                "message": error.text,
            },
            status=error.status,
        )


def create_app() -> web.Application:
    app = web.Application(middlewares=[json_error_middleware])
    app.router.add_get("/", index_handler)
    app.router.add_post("/search", search_handler)
    return app


def main() -> None:
    print("Day21 浏览器知识检索服务已启动。")
    print(f"页面：http://{HOST}:{PORT}")
    print("数据库：", DATABASE_PATH)
    print("Top-K：", TOP_K)
    print("距离阈值：", DISTANCE_THRESHOLD)
    print("按 Ctrl+C 停止服务。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
