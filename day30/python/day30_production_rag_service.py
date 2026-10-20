"""Day30 第 6 步：最终生产化RAG服务与运维入口。

启动：
    python day30/python/day30_production_rag_service.py

主要接口：
    POST /ask
    GET  /requests/{request_id}
    GET  /ops/status
    GET  /livez
    GET  /readyz
    GET  /health/dependencies
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiohttp import web
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from day30_config_management import AppConfig, ConfigurationError, load_config
from day30_database_migrations import migrate_database, open_database
from day30_health_checks import (
    dependencies_handler,
    live_handler,
    ready_handler,
)
from day30_release_preflight import (
    check_database,
    check_latest_backup,
    check_migration_history,
    check_packages,
    check_writable_directories,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAY22_PYTHON = PROJECT_ROOT / "day22" / "python"
if str(DAY22_PYTHON) not in sys.path:
    sys.path.insert(0, str(DAY22_PYTHON))

from day22_final_hybrid_rag import (  # noqa: E402
    RERANK_THRESHOLD,
    retrieve_rrf_candidates,
)
from day22_llm_reranker import rerank_candidates  # noqa: E402


LOGGER = logging.getLogger("day30.production")
MAX_QUESTION_LENGTH = 2000
MAX_EVIDENCE_COUNT = 4


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def empty_config_arguments() -> argparse.Namespace:
    return argparse.Namespace(
        environment=None,
        host=None,
        port=None,
        log_level=None,
        request_timeout_seconds=None,
        max_concurrency=None,
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def log_event(event: str, **fields: Any) -> None:
    """结构化日志不记录API密钥和完整上下文正文。"""

    LOGGER.info(
        json.dumps(
            {"event": event, "timestamp": utc_now(), **fields},
            ensure_ascii=False,
        )
    )


def run_required_preflight(config: AppConfig) -> None:
    """服务启动前只执行必须项；失败时拒绝绑定端口。"""

    checks = (
        ("packages", check_packages),
        ("database", lambda: check_database(config)),
        ("migration_history", lambda: check_migration_history(config)),
        ("latest_backup", check_latest_backup),
        ("writable_directories", lambda: check_writable_directories(config)),
    )
    failures: list[str] = []
    print("启动前必须项检查：")
    for name, function in checks:
        try:
            status, detail = function()
        except Exception as error:
            status, detail = "FAIL", f"{type(error).__name__}: {error}"
        print(f"- {name:<22} {status}: {detail}")
        if status == "FAIL":
            failures.append(f"{name}: {detail}")
    if failures:
        raise RuntimeError("；".join(failures))


def open_business_database(config: AppConfig) -> sqlite3.Connection:
    connection = sqlite3.connect(config.database_path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def create_request_record(
    config: AppConfig,
    request_id: str,
    trace_id: str,
    question: str,
) -> None:
    with open_business_database(config) as connection:
        connection.execute(
            """
            INSERT INTO rag_requests
                (request_id, question, answer, status, created_at, trace_id)
            VALUES (?, ?, NULL, 'pending', ?, ?)
            """,
            (request_id, question, utc_now(), trace_id),
        )


def complete_request_record(
    config: AppConfig,
    request_id: str,
    answer: str,
    sources: list[dict[str, Any]],
) -> None:
    with open_business_database(config) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                UPDATE rag_requests
                SET answer = ?, status = 'success'
                WHERE request_id = ?
                """,
                (answer, request_id),
            )
            connection.executemany(
                """
                INSERT INTO rag_sources
                    (request_id, chunk_id, source_path, relevance_score)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        request_id,
                        source["chunk_id"],
                        source["location"],
                        source["relevance_score"],
                    )
                    for source in sources
                ],
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def fail_request_record(
    config: AppConfig,
    request_id: str,
    public_error: str,
) -> None:
    with open_business_database(config) as connection:
        connection.execute(
            """
            UPDATE rag_requests
            SET answer = ?, status = 'failed'
            WHERE request_id = ?
            """,
            (public_error, request_id),
        )


def read_request_record(
    config: AppConfig,
    request_id: str,
) -> dict[str, Any] | None:
    with open_business_database(config) as connection:
        request_row = connection.execute(
            "SELECT * FROM rag_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if request_row is None:
            return None
        source_rows = connection.execute(
            """
            SELECT chunk_id, source_path, relevance_score
            FROM rag_sources WHERE request_id = ? ORDER BY source_id
            """,
            (request_id,),
        ).fetchall()
    result = dict(request_row)
    result["sources"] = [dict(row) for row in source_rows]
    return result


def read_operational_status(config: AppConfig) -> dict[str, Any]:
    with open_business_database(config) as connection:
        counts = {
            row["status"]: row["count"]
            for row in connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM rag_requests GROUP BY status
                """
            )
        }
        recent = [
            dict(row)
            for row in connection.execute(
                """
                SELECT request_id, status, trace_id, created_at
                FROM rag_requests ORDER BY created_at DESC LIMIT 10
                """
            )
        ]
    return {"request_counts": counts, "recent_requests": recent}


def build_evidence_text(sources: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for number, source in enumerate(sources, start=1):
        blocks.append(
            f"[资料{number}]\n"
            f"来源：{source['location']}\n"
            f"正文：{source['content']}"
        )
    return "\n\n".join(blocks)


async def execute_rag_pipeline(
    app: web.Application,
    question: str,
) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    """混合检索、LLM精排、阈值守门和基于证据回答。"""

    rrf_results, _ = await asyncio.to_thread(retrieve_rrf_candidates, question)
    reranked_results, _ = await asyncio.to_thread(
        rerank_candidates,
        question,
        rrf_results,
    )
    qualified = [
        item
        for item in reranked_results
        if item.relevance_score >= RERANK_THRESHOLD
    ][:MAX_EVIDENCE_COUNT]
    if not qualified:
        return (
            "知识库中没有足够可靠的资料回答这个问题。",
            [],
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

    sources = [
        {
            "chunk_id": item.chunk_id,
            "location": item.location,
            "relevance_score": item.relevance_score,
            "content": item.content,
        }
        for item in qualified
    ]
    evidence_text = build_evidence_text(sources)
    llm: ChatOpenAI = app["llm"]
    response = await llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "你是严格的RAG助手。只能根据参考资料回答；"
                    "不得使用外部知识。每个关键结论必须标注[资料N]。"
                    "资料不足时必须明确拒绝。回答保持简洁。"
                )
            ),
            HumanMessage(
                content=f"用户问题：\n{question}\n\n参考资料：\n{evidence_text}"
            ),
        ]
    )
    usage = response.usage_metadata or {}
    token_usage = {
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }
    return str(response.content), sources, token_usage


@web.middleware
async def request_context_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    started = time.perf_counter()
    # request_id由服务端生成，避免客户端重复ID撞上数据库主键。
    request_id = f"req-{uuid.uuid4().hex[:16]}"
    trace_id = request.headers.get("X-Trace-ID", "").strip()
    if not trace_id or len(trace_id) > 100:
        trace_id = uuid.uuid4().hex
    request["request_id"] = request_id
    request["trace_id"] = trace_id

    status = 500
    try:
        response = await handler(request)
        status = response.status
        return response
    except web.HTTPException as error:
        status = error.status
        raise
    except Exception:
        LOGGER.exception("unhandled_request_error request_id=%s", request_id)
        return web.json_response(
            {
                "ok": False,
                "error": "服务器内部错误。",
                "request_id": request_id,
                "trace_id": trace_id,
            },
            status=500,
        )
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        log_event(
            "http_request_finished",
            request_id=request_id,
            trace_id=trace_id,
            method=request.method,
            path=request.path,
            status=status,
            duration_ms=duration_ms,
        )


async def index_handler(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "service": request.app["config"].app_name,
            "version": "day30-v1",
            "endpoints": [
                "POST /ask",
                "GET /requests/{request_id}",
                "GET /ops/status",
                "GET /livez",
                "GET /readyz",
                "GET /health/dependencies",
            ],
        }
    )


async def ask_handler(request: web.Request) -> web.Response:
    try:
        raw_body = await request.json()
        payload = AskRequest.model_validate(raw_body)
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        raise web.HTTPBadRequest(
            text=json.dumps(
                {"ok": False, "error": f"请求参数无效：{error}"},
                ensure_ascii=False,
            ),
            content_type="application/json",
        ) from error

    question = payload.question.strip()
    if not question:
        raise web.HTTPBadRequest(text="question不能只有空白字符。")

    config: AppConfig = request.app["config"]
    request_id = request["request_id"]
    trace_id = request["trace_id"]
    await asyncio.to_thread(
        create_request_record,
        config,
        request_id,
        trace_id,
        question,
    )

    log_event(
        "rag_request_started",
        request_id=request_id,
        trace_id=trace_id,
        question_length=len(question),
    )
    try:
        async with request.app["pipeline_semaphore"]:
            async with asyncio.timeout(config.request_timeout_seconds):
                answer, sources, usage = await execute_rag_pipeline(
                    request.app,
                    question,
                )
        await asyncio.to_thread(
            complete_request_record,
            config,
            request_id,
            answer,
            sources,
        )
    except TimeoutError:
        public_error = "请求处理超时。"
        await asyncio.to_thread(
            fail_request_record, config, request_id, public_error
        )
        raise web.HTTPGatewayTimeout(text=public_error)
    except Exception:
        public_error = "RAG流水线执行失败。"
        await asyncio.to_thread(
            fail_request_record, config, request_id, public_error
        )
        raise

    public_sources = [
        {
            "chunk_id": source["chunk_id"],
            "location": source["location"],
            "relevance_score": source["relevance_score"],
        }
        for source in sources
    ]
    log_event(
        "rag_request_succeeded",
        request_id=request_id,
        trace_id=trace_id,
        source_count=len(public_sources),
        total_tokens=usage["total_tokens"],
    )
    return web.json_response(
        {
            "ok": True,
            "request_id": request_id,
            "trace_id": trace_id,
            "answer": answer,
            "sources": public_sources,
            "usage": usage,
        }
    )


async def request_detail_handler(request: web.Request) -> web.Response:
    config: AppConfig = request.app["config"]
    result = await asyncio.to_thread(
        read_request_record,
        config,
        request.match_info["request_id"],
    )
    if result is None:
        raise web.HTTPNotFound(text="没有找到该请求。")
    return web.json_response(result)


async def ops_status_handler(request: web.Request) -> web.Response:
    config: AppConfig = request.app["config"]
    status = await asyncio.to_thread(read_operational_status, config)
    status.update(
        {
            "service": config.app_name,
            "environment": config.environment,
            "model": config.model,
            "max_concurrency": config.max_concurrency,
        }
    )
    return web.json_response(status)


def create_app(config: AppConfig) -> web.Application:
    app = web.Application(
        middlewares=[request_context_middleware],
        client_max_size=32 * 1024,
    )
    app["config"] = config
    app["llm"] = ChatOpenAI(
        model=config.model,
        api_key=config.api_key.reveal(),
        base_url=config.base_url,
        temperature=0,
        max_retries=2,
        timeout=config.request_timeout_seconds,
    )
    app["pipeline_semaphore"] = asyncio.Semaphore(config.max_concurrency)

    app.router.add_get("/", index_handler)
    app.router.add_post("/ask", ask_handler)
    app.router.add_get("/requests/{request_id}", request_detail_handler)
    app.router.add_get("/ops/status", ops_status_handler)
    app.router.add_get("/livez", live_handler)
    app.router.add_get("/readyz", ready_handler)
    app.router.add_get("/health/dependencies", dependencies_handler)
    return app


def main() -> None:
    try:
        config = load_config(empty_config_arguments()).config
    except ConfigurationError as error:
        print("配置加载失败：", error)
        raise SystemExit(2) from error

    configure_logging(config.log_level)
    print("检查并应用数据库迁移：")
    connection = open_database(config.database_path)
    try:
        before, after = migrate_database(connection)
    finally:
        connection.close()
    print(f"数据库版本：{before} → {after}\n")

    try:
        run_required_preflight(config)
    except RuntimeError as error:
        print("\n启动被拒绝：", error)
        raise SystemExit(1) from error

    print("\nDay30最终生产化RAG服务已启动。")
    print(f"访问地址：http://{config.host}:{config.port}")
    print("模型问答：POST /ask")
    print("运维状态：GET /ops/status")
    print("健康检查：GET /livez 和 GET /readyz")
    print("按 Ctrl+C 优雅停止服务。")
    web.run_app(
        create_app(config),
        host=config.host,
        port=config.port,
        print=None,
    )


if __name__ == "__main__":
    main()
