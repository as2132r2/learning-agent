"""Day28 第 6 步：最终可观测 RAG Web 控制台。

运行：
    python day28/python/day28_final_observability_dashboard.py

浏览器：
    http://127.0.0.1:8042
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from aiohttp import web

from day28_rag_stage_tracing import (
    DATABASE_PATH,
    create_schema,
    load_trace,
    open_database,
    render_database_tree,
    run_rag_pipeline,
    stage_diagnostics,
)


HOST = "127.0.0.1"
PORT = 8042
DAY28_DIR = Path(__file__).resolve().parents[1]
HTML_PATH = DAY28_DIR / "html" / "day28_final.html"
CSS_PATH = DAY28_DIR / "css" / "day28_final.css"
JS_PATH = DAY28_DIR / "javascript" / "day28_final.js"
VALID_TRACE_ID = re.compile(r"^trace-[a-f0-9]{16}$")

# 避免用户连续点击时在本地演示数据库中同时启动过多写事务。
pipeline_semaphore = asyncio.Semaphore(2)


def read_overview() -> dict[str, Any]:
    with open_database() as connection:
        totals = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
                AVG(duration_ms) AS average_duration_ms
            FROM traces
            """
        ).fetchone()
        slowest_stage = connection.execute(
            """
            SELECT name, AVG(duration_ms) AS average_ms, COUNT(*) AS calls
            FROM spans
            WHERE parent_span_id IS NOT NULL
            GROUP BY name
            ORDER BY average_ms DESC
            LIMIT 1
            """
        ).fetchone()
        failure_rows = connection.execute(
            """
            SELECT name, error_type, COUNT(*) AS count
            FROM spans
            WHERE status = 'error' AND parent_span_id IS NOT NULL
            GROUP BY name, error_type
            ORDER BY count DESC, name
            LIMIT 5
            """
        ).fetchall()

    total = int(totals["total"] or 0)
    success = int(totals["success"] or 0)
    failed = int(totals["failed"] or 0)
    return {
        "total_traces": total,
        "successful_traces": success,
        "failed_traces": failed,
        "success_rate": round(success / total * 100, 1) if total else 0.0,
        "average_duration_ms": round(totals["average_duration_ms"] or 0, 3),
        "slowest_stage": None if slowest_stage is None else dict(slowest_stage),
        "failure_groups": [dict(row) for row in failure_rows],
        "database": str(DATABASE_PATH),
    }


def read_recent_traces(limit: int = 30) -> list[dict[str, Any]]:
    with open_database() as connection:
        rows = connection.execute(
            """
            SELECT trace_id, request_id, question, status,
                   started_at, duration_ms, error_type, error_message
            FROM traces
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def read_trace_detail(trace_id: str) -> dict[str, Any]:
    trace, spans = load_trace(trace_id)
    return {
        "trace": trace,
        "spans": spans,
        "tree_lines": render_database_tree(spans),
        "diagnostics": stage_diagnostics(spans),
    }


@web.middleware
async def error_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    try:
        return await handler(request)
    except web.HTTPException as error:
        return web.json_response(
            {"ok": False, "error": error.text or error.reason},
            status=error.status,
        )
    except Exception as error:
        print(f"[控制台错误] {type(error).__name__}: {error}")
        return web.json_response(
            {"ok": False, "error": "服务器内部错误。"},
            status=500,
        )


async def static_handler(request: web.Request) -> web.FileResponse:
    paths = {"/": HTML_PATH, "/app.css": CSS_PATH, "/app.js": JS_PATH}
    return web.FileResponse(paths[request.path])


async def health_handler(request: web.Request) -> web.Response:
    del request
    return web.json_response(
        {
            "status": "ok",
            "service": "day28-observability-dashboard",
            "database": str(DATABASE_PATH),
            "max_concurrent_pipelines": 2,
        }
    )


async def overview_handler(request: web.Request) -> web.Response:
    del request
    return web.json_response(await asyncio.to_thread(read_overview))


async def traces_handler(request: web.Request) -> web.Response:
    del request
    return web.json_response(
        {"traces": await asyncio.to_thread(read_recent_traces)}
    )


async def trace_detail_handler(request: web.Request) -> web.Response:
    trace_id = request.match_info["trace_id"]
    if not VALID_TRACE_ID.fullmatch(trace_id):
        raise web.HTTPBadRequest(text="trace_id格式无效。")
    try:
        detail = await asyncio.to_thread(read_trace_detail, trace_id)
    except LookupError as error:
        raise web.HTTPNotFound(text=str(error)) from error
    return web.json_response(detail)


async def ask_handler(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception as error:
        raise web.HTTPBadRequest(text="请求正文必须是JSON。") from error

    question = body.get("question") if isinstance(body, dict) else None
    simulate_error = body.get("simulate_error", False) if isinstance(body, dict) else False
    if not isinstance(question, str) or not question.strip():
        raise web.HTTPBadRequest(text="question必须是非空字符串。")
    if not isinstance(simulate_error, bool):
        raise web.HTTPBadRequest(text="simulate_error必须是布尔值。")

    async with pipeline_semaphore:
        result = await asyncio.to_thread(
            run_rag_pipeline,
            question.strip(),
            simulate_reranker_error=simulate_error,
        )

    status = 200 if result["status"] == "success" else 502
    return web.json_response(result, status=status)


def create_app() -> web.Application:
    app = web.Application(middlewares=[error_middleware])
    app.router.add_get("/", static_handler)
    app.router.add_get("/app.css", static_handler)
    app.router.add_get("/app.js", static_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/api/overview", overview_handler)
    app.router.add_get("/api/traces", traces_handler)
    app.router.add_get("/api/traces/{trace_id}", trace_detail_handler)
    app.router.add_post("/api/ask", ask_handler)
    return app


def main() -> None:
    create_schema()
    preview = read_overview()
    print("Day28 最终可观测 RAG 控制台已启动。")
    print(f"访问地址：http://{HOST}:{PORT}")
    print("数据库：", DATABASE_PATH)
    print("现有Trace数量：", preview["total_traces"])
    print("说明：本步使用确定性RAG阶段模拟，不调用真实模型。")
    print("按 Ctrl+C 停止服务。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
