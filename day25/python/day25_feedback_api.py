"""Day25 第 2 步：点赞、点踩与文字原因 HTTP API。

启动服务：
    python day25/python/day25_feedback_api.py

服务地址：
    http://127.0.0.1:8032

当前文件只负责 HTTP 层，数据库逻辑复用 day25_feedback_store.py。
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from aiohttp import web

from day25_feedback_store import (
    DATABASE_PATH,
    FeedbackInput,
    create_schema,
    list_feedback,
    open_database,
    rating_statistics,
    save_feedback,
)


HOST = "127.0.0.1"
PORT = 8032
VALID_RATINGS = {"up", "down", "neutral"}


def json_response(data: dict[str, Any], status: int = 200) -> web.Response:
    """统一创建 UTF-8 JSON 响应。"""

    return web.json_response(data, status=status, dumps=json_dumps)


def json_dumps(value: Any) -> str:
    """让 aiohttp 输出中文，而不是 Unicode 转义。"""

    import json

    return json.dumps(value, ensure_ascii=False)


async def read_json_object(request: web.Request) -> dict[str, Any]:
    """读取 JSON 请求体，并确保最外层是对象。"""

    try:
        body = await request.json()
    except Exception as error:
        raise web.HTTPBadRequest(
            text=json_dumps(
                {
                    "ok": False,
                    "error": "请求体必须是合法 JSON。",
                    "detail": str(error),
                }
            ),
            content_type="application/json",
        ) from error

    if not isinstance(body, dict):
        raise web.HTTPBadRequest(
            text=json_dumps({"ok": False, "error": "JSON 最外层必须是对象。"}),
            content_type="application/json",
        )
    return body


def require_string(body: dict[str, Any], field: str) -> str:
    """读取必填字符串字段。"""

    value = body.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空字符串。")
    return value.strip()


def optional_string(body: dict[str, Any], field: str) -> str | None:
    """读取可选字符串字段。"""

    value = body.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} 必须是字符串或 null。")
    return value.strip() or None


def build_feedback_input(body: dict[str, Any]) -> FeedbackInput:
    """把不可信 HTTP JSON 转换成 FeedbackInput。"""

    rating = require_string(body, "rating")
    if rating not in VALID_RATINGS:
        raise ValueError("rating 只能是 up、down 或 neutral。")

    source_ids = body.get("source_ids", [])
    if not isinstance(source_ids, list) or not all(
        isinstance(item, str) for item in source_ids
    ):
        raise ValueError("source_ids 必须是字符串列表。")

    pipeline_metrics = body.get("pipeline_metrics", {})
    if not isinstance(pipeline_metrics, dict):
        raise ValueError("pipeline_metrics 必须是 JSON 对象。")

    return FeedbackInput(
        request_id=require_string(body, "request_id"),
        session_id=require_string(body, "session_id"),
        user_id=require_string(body, "user_id"),
        question=require_string(body, "question"),
        answer=require_string(body, "answer"),
        rating=rating,  # type: ignore[arg-type]
        reason=optional_string(body, "reason"),
        answer_version=optional_string(body, "answer_version") or "day25-v1",
        source_ids=source_ids,
        pipeline_metrics=pipeline_metrics,
    )


async def health_handler(request: web.Request) -> web.Response:
    """GET /health：健康检查。"""

    del request
    return json_response(
        {
            "ok": True,
            "service": "day25-feedback-api",
            "database": str(DATABASE_PATH),
        }
    )


async def create_feedback_handler(request: web.Request) -> web.Response:
    """POST /feedback：新增或更新一条用户反馈。"""

    body = await read_json_object(request)
    try:
        feedback = build_feedback_input(body)

        # sqlite3 是同步接口。数据库操作很短，本练习直接执行；生产环境可放入线程池。
        with open_database() as connection:
            create_schema(connection)
            feedback_id, created = save_feedback(connection, feedback)
    except ValueError as error:
        return json_response({"ok": False, "error": str(error)}, status=400)
    except Exception as error:
        print(f"[反馈保存失败] {type(error).__name__}: {error}")
        return json_response(
            {"ok": False, "error": "服务器保存反馈失败。"},
            status=500,
        )

    print(
        f"[反馈已保存] request_id={feedback.request_id} "
        f"user_id={feedback.user_id} rating={feedback.rating} "
        f"operation={'created' if created else 'updated'}"
    )
    return json_response(
        {
            "ok": True,
            "operation": "created" if created else "updated",
            "feedback_id": feedback_id,
            "request_id": feedback.request_id,
            "rating": feedback.rating,
        },
        status=201 if created else 200,
    )


async def list_feedback_handler(request: web.Request) -> web.Response:
    """GET /feedback：读取反馈，支持 rating 查询参数。"""

    rating = request.query.get("rating")
    if rating is not None and rating not in VALID_RATINGS:
        return json_response(
            {"ok": False, "error": "rating 只能是 up、down 或 neutral。"},
            status=400,
        )

    with open_database() as connection:
        create_schema(connection)
        records = list_feedback(connection)

    if rating is not None:
        records = [record for record in records if record.rating == rating]

    return json_response(
        {
            "ok": True,
            "count": len(records),
            "items": [asdict(record) for record in records],
        }
    )


async def statistics_handler(request: web.Request) -> web.Response:
    """GET /feedback/statistics：返回点赞、点踩数量和满意率。"""

    del request
    with open_database() as connection:
        create_schema(connection)
        statistics = rating_statistics(connection)

    rated_total = statistics["up"] + statistics["down"]
    satisfaction_rate = (
        round(statistics["up"] / rated_total * 100, 1) if rated_total else None
    )

    return json_response(
        {
            "ok": True,
            "ratings": statistics,
            "rated_total": rated_total,
            "satisfaction_rate": satisfaction_rate,
            "satisfaction_rate_unit": "percent",
        }
    )


@web.middleware
async def error_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    """把 aiohttp HTTP 异常也统一转换成 JSON。"""

    try:
        return await handler(request)
    except web.HTTPException as error:
        if error.content_type == "application/json":
            raise
        return json_response(
            {"ok": False, "error": error.reason},
            status=error.status,
        )


def create_app() -> web.Application:
    """创建 aiohttp Application 并注册路由。"""

    app = web.Application(middlewares=[error_middleware])
    app.router.add_get("/health", health_handler)
    app.router.add_post("/feedback", create_feedback_handler)
    app.router.add_get("/feedback", list_feedback_handler)
    app.router.add_get("/feedback/statistics", statistics_handler)
    return app


def main() -> None:
    with open_database() as connection:
        create_schema(connection)

    print("Day25 用户反馈 HTTP API 已启动")
    print(f"服务地址：http://{HOST}:{PORT}")
    print("GET  /health")
    print("POST /feedback")
    print("GET  /feedback")
    print("GET  /feedback?rating=down")
    print("GET  /feedback/statistics")
    print("按 Ctrl+C 停止服务。")

    web.run_app(
        create_app(),
        host=HOST,
        port=PORT,
        print=None,
        handle_signals=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    except asyncio.CancelledError:
        print("\n服务任务已取消。")
