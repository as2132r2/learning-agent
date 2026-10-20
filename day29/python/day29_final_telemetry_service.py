"""Day29 第 6 步：最终标准化遥测 RAG 服务与 Web 控制台。

运行：
    python day29/python/day29_final_telemetry_service.py

浏览器：
    http://127.0.0.1:8043
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Awaitable, Callable

from aiohttp import web

from day29_batch_exporter import BatchSpanExporter, ExportableSpan
from day29_sampling_strategies import ParentBasedSampler, TraceIdRatioBasedSampler
from day29_span_semantics import (
    SpanBuilder,
    SpanKind,
    StatusCode,
    apply_http_status,
)
from day29_telemetry_governance import RawSpan, govern_span
from day29_traceparent_basic import (
    InvalidTraceparentError,
    TraceContext,
    create_root_context,
    parse_traceparent,
)


HOST, PORT = "127.0.0.1", 8043
DAY29_DIR = Path(__file__).resolve().parents[1]
HTML_PATH = DAY29_DIR / "html" / "day29_final.html"
CSS_PATH = DAY29_DIR / "css" / "day29_final.css"
JS_PATH = DAY29_DIR / "javascript" / "day29_final.js"
DATABASE_PATH = DAY29_DIR / "database" / "day29_final_telemetry.db"
EXPORT_PATH = DAY29_DIR / "logs" / "day29_final_export.jsonl"

sampler = ParentBasedSampler(TraceIdRatioBasedSampler(0.50))
pipeline_semaphore = asyncio.Semaphore(2)


def open_database() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def create_schema() -> None:
    with open_database() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS exported_spans (
                span_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                parent_span_id TEXT,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                ended_at TEXT NOT NULL,
                attributes_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_final_trace
            ON exported_spans(trace_id, ended_at);
            """
        )


class SQLiteJsonlSink:
    """批量写 SQLite，同时保留便于观察的 JSONL 副本。"""

    def __init__(self, database_path: Path, jsonl_path: Path) -> None:
        self.database_path = database_path
        self.jsonl_path = jsonl_path

    def _write(self, batch: list[ExportableSpan]) -> None:
        with open_database() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO exported_spans (
                    span_id, trace_id, parent_span_id, name, kind,
                    status, duration_ms, ended_at, attributes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        span.span_id, span.trace_id, span.parent_span_id,
                        span.name, span.kind, span.status, span.duration_ms,
                        span.ended_at,
                        json.dumps(span.attributes, ensure_ascii=False),
                    )
                    for span in batch
                ],
            )
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            for span in batch:
                file.write(json.dumps(asdict(span), ensure_ascii=False) + "\n")

    async def export(self, batch: list[ExportableSpan]) -> None:
        await asyncio.to_thread(self._write, batch)


def semantic_to_exportable(span: Any) -> ExportableSpan:
    """先治理，再转换为批量导出器接受的数据。"""

    governed = govern_span(
        RawSpan(
            span_id=span.context.span_id,
            trace_id=span.context.trace_id,
            name=span.name,
            ended_at=span.ended_at,
            attributes=span.attributes,
            events=span.events,
        )
    )
    attributes = dict(governed.attributes)
    attributes["telemetry.events"] = governed.events
    attributes["telemetry.governance"] = asdict(governed.governance)
    return ExportableSpan(
        trace_id=governed.trace_id,
        span_id=governed.span_id,
        parent_span_id=span.parent_span_id,
        name=governed.name,
        kind=span.kind.value,
        status=span.status_code.value,
        duration_ms=span.duration_ms,
        ended_at=governed.ended_at,
        attributes=attributes,
    )


def resolve_parent(header: str | None) -> tuple[TraceContext, str]:
    if header is None:
        return create_root_context(sampled=False), "missing"
    try:
        return parse_traceparent(header), "valid_remote"
    except InvalidTraceparentError:
        return create_root_context(sampled=False), "invalid_replaced"


@web.middleware
async def telemetry_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    parent, propagation = resolve_parent(request.headers.get("traceparent"))
    sampling = sampler.should_sample(
        parent.trace_id,
        parent if parent.is_remote else None,
    )
    parent = replace(parent, trace_flags=0x01 if sampling.sampled else 0x00)

    builder = SpanBuilder(
        f"{request.method} {request.path}",
        SpanKind.SERVER,
        parent,
        {
            "http.request.method": request.method,
            "http.route": request.path,
            "url.scheme": request.scheme,
            "url.path": request.path,
            "server.address": HOST,
            "server.port": PORT,
            "telemetry.propagation": propagation,
            "telemetry.sampler": sampling.sampler,
            "telemetry.sampled": sampling.sampled,
        },
    )
    root = builder.__enter__()
    request["root_context"] = root.context
    request["semantic_spans"] = []
    request["sampled"] = sampling.sampled

    try:
        try:
            response = await handler(request)
        except web.HTTPException as error:
            response = web.json_response(
                {"ok": False, "error": error.text or error.reason},
                status=error.status,
            )
        except Exception as error:
            root.set_status(StatusCode.ERROR, str(error))
            root.set_attribute("error.type", type(error).__name__)
            response = web.json_response(
                {"ok": False, "error": "服务器内部错误。"}, status=500
            )

        apply_http_status(root, response.status)
        response.headers["traceparent"] = root.context.to_traceparent()
        response.headers["X-Telemetry-Sampled"] = str(sampling.sampled).lower()
        return response
    finally:
        builder.__exit__(None, None, None)
        request["semantic_spans"].append(root)
        if sampling.sampled:
            exporter: BatchSpanExporter = request.app["exporter"]
            for span in request["semantic_spans"]:
                exporter.emit(semantic_to_exportable(span))


async def traced_sleep(
    request: web.Request,
    name: str,
    kind: SpanKind,
    delay: float,
    attributes: dict[str, Any],
) -> Any:
    builder = SpanBuilder(name, kind, request["root_context"], attributes)
    with builder as span:
        await asyncio.sleep(delay)
    request["semantic_spans"].append(span)
    return span


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
        retrieval = await traced_sleep(
            request, "rag.retrieve", SpanKind.INTERNAL, 0.05,
            {"rag.operation.name": "retrieve", "rag.top_k": 3,
             "rag.question": question, "user.email": "demo@example.com",
             "rag.document.count": 2},
        )

        builder = SpanBuilder(
            "POST model.example/v1/chat/completions",
            SpanKind.CLIENT,
            request["root_context"],
            {"http.request.method": "POST", "server.address": "model.example",
             "url.full": "https://model.example/v1/chat/completions?api_key=secret",
             "rag.model.name": "deepseek-chat-demo", "authorization": "Bearer secret"},
        )
        with builder as model_span:
            await asyncio.sleep(0.07)
            apply_http_status(model_span, 503 if simulate_error else 200)
            model_span.set_attribute("rag.input_tokens", 220)
            model_span.set_attribute("rag.output_tokens", 54)
        request["semantic_spans"].append(model_span)

        if simulate_error:
            return web.json_response(
                {"ok": False, "error": "模拟模型服务503。",
                 "trace_id": request["root_context"].trace_id,
                 "sampled": request["sampled"]},
                status=502,
            )

        answer_span = await traced_sleep(
            request, "rag.answer.compose", SpanKind.INTERNAL, 0.025,
            {"rag.operation.name": "answer", "rag.evidence.count": 2,
             "rag.answer.length": 31},
        )
        return web.json_response(
            {"ok": True, "answer": "RAG先检索资料，再结合资料生成回答。",
             "trace_id": request["root_context"].trace_id,
             "sampled": request["sampled"]},
        )


def read_overview() -> dict[str, Any]:
    with open_database() as connection:
        total = connection.execute(
            "SELECT COUNT(DISTINCT trace_id) AS value FROM exported_spans"
        ).fetchone()["value"]
        spans = connection.execute(
            "SELECT COUNT(*) AS value FROM exported_spans"
        ).fetchone()["value"]
        errors = connection.execute(
            "SELECT COUNT(*) AS value FROM exported_spans WHERE status='ERROR'"
        ).fetchone()["value"]
        kinds = connection.execute(
            "SELECT kind, COUNT(*) AS count FROM exported_spans GROUP BY kind"
        ).fetchall()
    return {"exported_traces": total, "exported_spans": spans,
            "error_spans": errors, "kinds": [dict(row) for row in kinds]}


def read_traces() -> list[dict[str, Any]]:
    with open_database() as connection:
        rows = connection.execute(
            """SELECT trace_id, COUNT(*) AS span_count,
                      MAX(ended_at) AS latest_at,
                      MAX(CASE WHEN status='ERROR' THEN 1 ELSE 0 END) AS has_error
               FROM exported_spans GROUP BY trace_id
               ORDER BY latest_at DESC LIMIT 30"""
        ).fetchall()
    return [dict(row) for row in rows]


def read_trace(trace_id: str) -> list[dict[str, Any]]:
    with open_database() as connection:
        rows = connection.execute(
            "SELECT * FROM exported_spans WHERE trace_id=? ORDER BY ended_at",
            (trace_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["attributes"] = json.loads(item.pop("attributes_json"))
        result.append(item)
    return result


async def api_overview(request: web.Request) -> web.Response:
    exporter: BatchSpanExporter = request.app["exporter"]
    data = await asyncio.to_thread(read_overview)
    data["exporter"] = asdict(exporter.metrics)
    data["sampler"] = sampler.description
    return web.json_response(data)


async def api_traces(request: web.Request) -> web.Response:
    return web.json_response({"traces": await asyncio.to_thread(read_traces)})


async def api_trace(request: web.Request) -> web.Response:
    spans = await asyncio.to_thread(read_trace, request.match_info["trace_id"])
    if not spans:
        raise web.HTTPNotFound(text="Trace不存在或未被采样导出。")
    return web.json_response({"spans": spans})


async def static_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(
        {"/": HTML_PATH, "/app.css": CSS_PATH, "/app.js": JS_PATH}[request.path]
    )


async def start_exporter(app: web.Application) -> None:
    create_schema()
    sink = SQLiteJsonlSink(DATABASE_PATH, EXPORT_PATH)
    app["exporter"] = BatchSpanExporter(
        sink, max_queue_size=64, max_batch_size=8, schedule_delay_seconds=0.15
    )
    await app["exporter"].start()


async def stop_exporter(app: web.Application) -> None:
    await app["exporter"].shutdown()


def create_app() -> web.Application:
    app = web.Application(middlewares=[telemetry_middleware])
    app.router.add_get("/", static_handler)
    app.router.add_get("/app.css", static_handler)
    app.router.add_get("/app.js", static_handler)
    app.router.add_post("/api/ask", ask_handler)
    app.router.add_get("/api/overview", api_overview)
    app.router.add_get("/api/traces", api_traces)
    app.router.add_get("/api/traces/{trace_id}", api_trace)
    app.on_startup.append(start_exporter)
    app.on_cleanup.append(stop_exporter)
    return app


def main() -> None:
    logging.getLogger("day28.trace").disabled = True
    print(f"Day29 标准化遥测RAG控制台：http://{HOST}:{PORT}")
    print("采样器：", sampler.description)
    print("数据库：", DATABASE_PATH)
    print("按 Ctrl+C 停止服务。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
