"""Day25 第 6 步：带在线反馈闭环的最终 Web RAG。

运行：
    python day25/python/day25_final_feedback_rag.py

浏览器：
    http://127.0.0.1:8034
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError


PROJECT_DIR = Path(__file__).resolve().parents[2]
DAY25_DIR = PROJECT_DIR / "day25"
for directory in (
    PROJECT_DIR / "day24" / "python",
    DAY25_DIR / "python",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import day24_final_optimized_rag as day24  # noqa: E402
from day25_feedback_linkage import (  # noqa: E402
    AnswerRun,
    LinkedFeedback,
    SourceTrace,
    create_schema,
    open_database,
    save_answer_run,
    save_linked_feedback,
)


HOST, PORT = "127.0.0.1", 8034
HTML = DAY25_DIR / "html" / "day25_final.html"
JS = DAY25_DIR / "javascript" / "day25_final.js"
CSS = DAY25_DIR / "css" / "day25_final.css"
BASE_CSS = PROJECT_DIR / "day24" / "css" / "day24_final.css"


class AskPayload(day24.AskPayload):
    """Day24 请求参数加上会话和用户身份。"""

    session_id: str = Field(min_length=1, max_length=100)
    user_id: str = Field(min_length=1, max_length=100)


class ClientCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["1.0"]
    type: Literal["ask", "cancel"]
    request_id: str = Field(min_length=1, max_length=150)
    payload: AskPayload


class FeedbackPayload(BaseModel):
    """浏览器只提交反馈本身，回答详情由服务端按 request_id 查找。"""

    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1, max_length=150)
    user_id: str = Field(min_length=1, max_length=100)
    rating: Literal["up", "down", "neutral"]
    reason: str | None = Field(default=None, max_length=1000)


def initialize_database() -> None:
    with open_database() as connection:
        create_schema(connection)


def save_completed_answer(
    request_id: str,
    payload: AskPayload,
    plan: dict[str, Any],
    reranked: dict[str, Any],
    done: dict[str, Any],
) -> None:
    """把已成功交付的回答转换成 Day25 规范化追踪记录。"""

    accepted_count = int(reranked.get("accepted_count", 0))
    accepted_results = list(reranked.get("results", []))[:accepted_count]
    sources = [
        SourceTrace(
            chunk_id=str(item["chunk_id"]),
            source_path=str(item["location"]),
            citation_number=index,
            reranker_score=int(item["relevance_score"]),
            excerpt=str(item["content"]),
        )
        for index, item in enumerate(accepted_results, start=1)
    ]

    stages = done.get("stages", [])
    latency_ms = round(float(done.get("wall_seconds", 0)) * 1000)
    answer_run = AnswerRun(
        request_id=request_id,
        session_id=payload.session_id,
        user_id=payload.user_id,
        question=(payload.question or "").strip(),
        answer=str(done["answer"]),
        answer_version="day25-feedback-rag-v1",
        risk_level=str(plan.get("executed_risk") or "low"),
        model_name=os.getenv("DEEPSEEK_MODEL", "unknown"),
        model_calls=int(done.get("model_calls", 0)),
        # Day24阶段只聚合总Token，因此保存到input_tokens，output_tokens暂记0。
        input_tokens=int(done.get("total_tokens", 0)),
        output_tokens=0,
        latency_ms=latency_ms,
        cache_hits=int(done.get("cache_hits", 0)),
        quality_passed=done.get("status") == "success",
        sources=sources,
    )

    with open_database() as connection:
        create_schema(connection)
        save_answer_run(connection, answer_run)

    print(
        f"[回答追踪已保存] request_id={request_id} "
        f"sources={len(sources)} calls={answer_run.model_calls} "
        f"tokens={answer_run.input_tokens} stages={len(stages)}"
    )


@dataclass(slots=True)
class PipelineRecorder:
    """观察 Day24 发出的事件，并在 done 之前持久化完整回答。"""

    request_id: str
    payload: AskPayload
    plan: dict[str, Any] = field(default_factory=dict)
    reranked: dict[str, Any] = field(default_factory=dict)
    saved: bool = False

    async def observe(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        data = message.get("payload") or {}
        if kind == "plan":
            self.plan = data
        elif kind == "reranked":
            self.reranked = data
        elif kind == "done" and not self.saved:
            # 先写数据库，再把 done 发给浏览器，避免用户立即反馈时查不到回答。
            await asyncio.to_thread(
                save_completed_answer,
                self.request_id,
                self.payload,
                self.plan,
                self.reranked,
                data,
            )
            self.saved = True


class RecordingWebSocket:
    """WebSocket代理：记录事件后原样转发，不修改Day24流水线。"""

    def __init__(
        self,
        real_ws: web.WebSocketResponse,
        recorder: PipelineRecorder,
    ) -> None:
        self.real_ws = real_ws
        self.recorder = recorder

    @property
    def closed(self) -> bool:
        return self.real_ws.closed

    async def send_json(self, data: dict[str, Any]) -> None:
        await self.recorder.observe(data)
        await self.real_ws.send_json(data)


async def run_tracked_pipeline(
    real_ws: web.WebSocketResponse,
    state: day24.ConnectionState,
    command: ClientCommand,
) -> None:
    recorder = PipelineRecorder(command.request_id, command.payload)
    proxy_ws = RecordingWebSocket(real_ws, recorder)
    core_payload = day24.AskPayload(
        question=command.payload.question,
        risk=command.payload.risk,
        max_calls=command.payload.max_calls,
        max_tokens=command.payload.max_tokens,
        allow_downgrade=command.payload.allow_downgrade,
    )
    await day24.run_pipeline(
        proxy_ws,  # type: ignore[arg-type]
        state,
        command.request_id,
        core_payload,
    )


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    state = day24.ConnectionState()
    await day24.send(
        ws,
        state,
        "connected",
        None,
        {
            "cache_entries": day24.cache.count(),
            "pipeline_limit": day24.PIPELINE_LIMIT,
            "feedback_enabled": True,
        },
    )

    try:
        async for raw in ws:
            if raw.type != WSMsgType.TEXT:
                continue
            try:
                command = ClientCommand.model_validate_json(raw.data)
            except ValidationError as error:
                await day24.send(
                    ws,
                    state,
                    "error",
                    None,
                    {"details": error.errors(include_url=False, include_input=False)},
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
                    await day24.send(
                        ws, state, "cancel_not_found", command.request_id, {}
                    )
                continue

            if not (command.payload.question or "").strip():
                await day24.send(
                    ws,
                    state,
                    "error",
                    command.request_id,
                    {"message": "问题不能为空。"},
                )
            elif state.active_task and not state.active_task.done():
                await day24.send(
                    ws,
                    state,
                    "busy",
                    command.request_id,
                    {"message": "当前连接已有任务。"},
                )
            else:
                state.active_request_id = command.request_id
                state.active_task = asyncio.create_task(
                    run_tracked_pipeline(ws, state, command),
                    name=f"feedback-rag-{command.request_id}",
                )
    finally:
        if state.active_task and not state.active_task.done():
            state.active_task.cancel()
            await asyncio.gather(state.active_task, return_exceptions=True)
    return ws


async def feedback_handler(request: web.Request) -> web.Response:
    """POST /feedback：只接受评价数据，回答信息由服务端数据库提供。"""

    try:
        body = await request.json()
        payload = FeedbackPayload.model_validate(body)
    except (ValidationError, ValueError, TypeError) as error:
        details = (
            error.errors(include_url=False, include_input=False)
            if isinstance(error, ValidationError)
            else str(error)
        )
        return web.json_response(
            {"ok": False, "error": "反馈参数无效。", "details": details},
            status=400,
        )

    def persist() -> tuple[str, bool]:
        with open_database() as connection:
            create_schema(connection)
            answer = connection.execute(
                "SELECT user_id FROM answer_runs WHERE request_id = ?",
                (payload.request_id,),
            ).fetchone()
            if answer is None:
                raise LookupError("找不到对应的已完成回答。")
            if answer["user_id"] != payload.user_id:
                raise PermissionError("不能评价其他用户的回答。")
            return save_linked_feedback(
                connection,
                LinkedFeedback(
                    request_id=payload.request_id,
                    user_id=payload.user_id,
                    rating=payload.rating,
                    reason=payload.reason.strip() if payload.reason else None,
                ),
            )

    try:
        feedback_id, created = await asyncio.to_thread(persist)
    except LookupError as error:
        return web.json_response({"ok": False, "error": str(error)}, status=404)
    except PermissionError as error:
        return web.json_response({"ok": False, "error": str(error)}, status=403)

    return web.json_response(
        {
            "ok": True,
            "operation": "created" if created else "updated",
            "feedback_id": feedback_id,
            "request_id": payload.request_id,
            "rating": payload.rating,
        },
        status=201 if created else 200,
    )


async def feedback_statistics_handler(request: web.Request) -> web.Response:
    """GET /feedback/statistics：按版本统计反馈和平均性能。"""

    del request

    def query() -> list[dict[str, Any]]:
        with open_database() as connection:
            create_schema(connection)
            rows = connection.execute(
                """
                SELECT
                    ar.answer_version,
                    COUNT(lf.feedback_id) AS feedback_count,
                    SUM(CASE WHEN lf.rating = 'up' THEN 1 ELSE 0 END) AS up_count,
                    SUM(CASE WHEN lf.rating = 'down' THEN 1 ELSE 0 END) AS down_count,
                    SUM(CASE WHEN lf.rating = 'neutral' THEN 1 ELSE 0 END)
                        AS neutral_count,
                    ROUND(AVG(ar.latency_ms), 1) AS average_latency_ms,
                    ROUND(AVG(ar.model_calls), 2) AS average_model_calls
                FROM answer_runs AS ar
                LEFT JOIN linked_feedback AS lf
                    ON lf.request_id = ar.request_id
                GROUP BY ar.answer_version
                ORDER BY ar.answer_version
                """
            ).fetchall()
            return [dict(row) for row in rows]

    return web.json_response({"ok": True, "versions": await asyncio.to_thread(query)})


async def file_handler(request: web.Request) -> web.FileResponse:
    files = {
        "/": HTML,
        "/app.js": JS,
        "/app.css": CSS,
        "/base.css": BASE_CSS,
    }
    return web.FileResponse(files[request.path])


@web.middleware
async def error_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    try:
        return await handler(request)
    except web.HTTPException as error:
        return web.json_response(
            {"ok": False, "error": error.text or error.reason},
            status=error.status,
        )
    except Exception as error:
        print(f"[HTTP错误] {type(error).__name__}: {error}")
        return web.json_response(
            {"ok": False, "error": "服务器内部错误。"}, status=500
        )


async def close_resources(app: web.Application) -> None:
    del app
    day24.cache.close()


def create_app() -> web.Application:
    app = web.Application(
        middlewares=[error_middleware],
        client_max_size=day24.MAX_UPLOAD_SIZE + 65536,
    )
    app.router.add_get("/", file_handler)
    app.router.add_get("/app.js", file_handler)
    app.router.add_get("/app.css", file_handler)
    app.router.add_get("/base.css", file_handler)
    app.router.add_post("/upload", day24.upload_handler)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_post("/feedback", feedback_handler)
    app.router.add_get("/feedback/statistics", feedback_statistics_handler)
    app.on_cleanup.append(close_resources)
    return app


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    initialize_database()
    print(f"Day25 在线反馈闭环 Web RAG：http://{HOST}:{PORT}")
    print("RAG核心：复用 Day24 最终优化流水线")
    print("回答与反馈数据库：", DAY25_DIR / "database" / "day25_feedback.db")
    print("模型缓存：", day24.CACHE_DATABASE_PATH)
    print("按 Ctrl+C 停止服务。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
