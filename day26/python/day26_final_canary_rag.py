"""Day26 第 6 步：A/B 实验与灰度发布最终 Web RAG。

运行：
    python day26/python/day26_final_canary_rag.py

浏览器：
    http://127.0.0.1:8035
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError


PROJECT_DIR = Path(__file__).resolve().parents[2]
DAY26_DIR = PROJECT_DIR / "day26"
for directory in (
    PROJECT_DIR / "day23" / "python",
    PROJECT_DIR / "day24" / "python",
    PROJECT_DIR / "day25" / "python",
    DAY26_DIR / "python",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import day24_final_optimized_rag as day24  # noqa: E402
import day25_final_feedback_rag as day25  # noqa: E402
from day23_structured_citations import (  # noqa: E402
    StructuredRAGAnswer,
    format_evidence,
)
from day25_feedback_linkage import (  # noqa: E402
    LinkedFeedback,
    create_schema as create_feedback_schema,
    open_database as open_feedback_database,
    save_linked_feedback,
)
from day26_exposure_events import (  # noqa: E402
    FeedbackEventInput,
    OutcomeInput,
    create_event_schema,
    log_exposure,
    log_feedback,
    log_outcome,
)
from day26_guardrails import create_guardrail_schema  # noqa: E402
from day26_stable_assignment import (  # noqa: E402
    EXPERIMENT,
    assign_user,
    create_schema as create_experiment_schema,
    open_database as open_experiment_database,
    save_experiment,
)
from day26_variant_config import (  # noqa: E402
    VARIANT_CONFIGS,
    VariantConfig,
    create_config_schema,
    freeze_variant_configs,
    load_frozen_config,
)


HOST, PORT = "127.0.0.1", 8035
HTML = DAY26_DIR / "html" / "day26_final.html"
JS = DAY26_DIR / "javascript" / "day26_final.js"
CSS = DAY26_DIR / "css" / "day26_final.css"
DAY24_CSS = PROJECT_DIR / "day24" / "css" / "day24_final.css"
DAY25_CSS = PROJECT_DIR / "day25" / "css" / "day25_final.css"

# ContextVar 的值属于当前异步上下文，asyncio.to_thread 也会复制该上下文。
current_variant_config: contextvars.ContextVar[VariantConfig] = contextvars.ContextVar(
    "current_variant_config",
    default=VARIANT_CONFIGS[0],
)
current_config_hash: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_config_hash",
    default="control-default",
)

# 保存原函数，再给 Day24 安装一次请求级适配器。
original_cached_model_call = day24.async_cached_model_call


def generate_variant_answer(question: str, evidence: list[Any]) -> StructuredRAGAnswer:
    """使用当前请求所属版本的冻结 Prompt 生成结构化回答。"""

    config = current_variant_config.get()
    llm = ChatOpenAI(
        model=config.model_name,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=config.temperature,
    )
    model = llm.with_structured_output(
        StructuredRAGAnswer,
        method="function_calling",
    )
    structure_rule = (
        "把回答拆成可独立验证的claims，并为每条claim填写真正支持它的"
        "citation_numbers。引用编号只能来自资料；资料不足时can_answer=false且"
        "claims为空列表。"
    )
    return model.invoke(
        [
            SystemMessage(content=f"{config.system_prompt}\n{structure_rule}"),
            HumanMessage(
                content=(
                    f"用户问题：\n{question}\n\n"
                    f"参考资料：\n{format_evidence(evidence)}"
                )
            ),
        ]
    )


async def experiment_cached_model_call(
    trace,
    controller,
    stage_name,
    cache_input,
    function,
    function_args,
    serializer,
    deserializer,
    estimated_tokens,
):
    """让版本相关阶段的缓存键包含冻结配置哈希。"""

    if stage_name in {"StructuredAnswer", "RepairAnswer"}:
        cache_input = {
            "experiment_config_hash": current_config_hash.get(),
            "original_input": cache_input,
        }
    return await original_cached_model_call(
        trace,
        controller,
        stage_name,
        cache_input,
        function,
        function_args,
        serializer,
        deserializer,
        estimated_tokens,
    )


# Day24运行时从模块全局查找这两个函数，因此只安装一次即可。
day24.generate_structured_answer = generate_variant_answer
day24.async_cached_model_call = experiment_cached_model_call


def initialize_databases() -> None:
    with open_experiment_database() as connection:
        create_experiment_schema(connection)
        create_config_schema(connection)
        create_event_schema(connection)
        create_guardrail_schema(connection)
        save_experiment(connection, EXPERIMENT)
        freeze_variant_configs(connection, EXPERIMENT, VARIANT_CONFIGS)
    with open_feedback_database() as connection:
        create_feedback_schema(connection)


def experiment_status() -> str:
    with open_experiment_database() as connection:
        row = connection.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (EXPERIMENT.experiment_id,),
        ).fetchone()
    return "stopped" if row is None else str(row["status"])


def resolve_experiment(user_id: str) -> tuple[str, VariantConfig, str, bool]:
    """运行中按Assignment分流；停止后全部安全回退control且不计曝光。"""

    with open_experiment_database() as connection:
        status = connection.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (EXPERIMENT.experiment_id,),
        ).fetchone()["status"]
        if status != "running":
            control = next(item for item in VARIANT_CONFIGS if item.variant == "control")
            _, digest = load_frozen_config(
                connection,
                EXPERIMENT.experiment_id,
                EXPERIMENT.allocation_version,
                "control",
            )
            return "control", control, digest, False

        assignment = assign_user(connection, EXPERIMENT, user_id)
        config, digest = load_frozen_config(
            connection,
            assignment.experiment_id,
            assignment.allocation_version,
            assignment.variant,
        )
        return assignment.variant, config, digest, True


def record_exposure(request_id: str, user_id: str) -> None:
    with open_experiment_database() as connection:
        log_exposure(connection, request_id, user_id)


@dataclass(slots=True)
class OutcomeRecorder:
    """从WebSocket终止事件生成实验Outcome，并在转发前写入。"""

    request_id: str
    exposed: bool
    started: float
    recorded: bool = False

    async def observe(self, message: dict[str, Any]) -> None:
        if not self.exposed or self.recorded:
            return
        kind = str(message.get("type"))
        payload = message.get("payload") or {}
        mapping = {
            "done": "success",
            "quality_rejected": "rejected",
            "retrieval_rejected": "rejected",
            "budget_rejected": "rejected",
            "budget_exceeded": "rejected",
            "error": "error",
            "cancelled": "cancelled",
        }
        status = mapping.get(kind)
        if status is None:
            return
        outcome = OutcomeInput(
            request_id=self.request_id,
            status=status,  # type: ignore[arg-type]
            quality_passed=kind == "done",
            latency_ms=round((time.perf_counter() - self.started) * 1000),
            model_calls=int(payload.get("model_calls", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            cache_hits=int(payload.get("cache_hits", 0)),
            error_type=(
                str(payload.get("error_type"))
                if payload.get("error_type") is not None
                else None
            ),
        )

        def persist() -> None:
            with open_experiment_database() as connection:
                log_outcome(connection, outcome)

        await asyncio.to_thread(persist)
        self.recorded = True


class OutcomeWebSocket:
    def __init__(self, real_ws: Any, recorder: OutcomeRecorder) -> None:
        self.real_ws = real_ws
        self.recorder = recorder

    @property
    def closed(self) -> bool:
        return self.real_ws.closed

    async def send_json(self, data: dict[str, Any]) -> None:
        await self.recorder.observe(data)
        await self.real_ws.send_json(data)


async def run_experiment_pipeline(
    ws: web.WebSocketResponse,
    state: day24.ConnectionState,
    command: day25.ClientCommand,
) -> None:
    user_id = command.payload.user_id
    variant, config, digest, exposed = await asyncio.to_thread(
        resolve_experiment, user_id
    )
    if exposed:
        await asyncio.to_thread(record_exposure, command.request_id, user_id)

    await day24.send(
        ws,
        state,
        "experiment_assigned",
        command.request_id,
        {
            "experiment_id": EXPERIMENT.experiment_id,
            "allocation_version": EXPERIMENT.allocation_version,
            "variant": variant,
            "prompt_version": config.prompt_version,
            "config_hash": digest[:12],
            "counted_as_exposure": exposed,
            "experiment_status": "running" if exposed else "stopped",
        },
    )

    token_config = current_variant_config.set(config)
    token_hash = current_config_hash.set(digest)
    recorder = OutcomeRecorder(command.request_id, exposed, time.perf_counter())
    proxy = OutcomeWebSocket(ws, recorder)
    try:
        await day25.run_tracked_pipeline(
            proxy,  # type: ignore[arg-type]
            state,
            command,
        )
    finally:
        current_variant_config.reset(token_config)
        current_config_hash.reset(token_hash)


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
            "experiment_id": EXPERIMENT.experiment_id,
            "experiment_status": experiment_status(),
        },
    )
    try:
        async for raw in ws:
            if raw.type != WSMsgType.TEXT:
                continue
            try:
                command = day25.ClientCommand.model_validate_json(raw.data)
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
                    await day24.send(ws, state, "cancel_not_found", command.request_id, {})
            elif not (command.payload.question or "").strip():
                await day24.send(
                    ws, state, "error", command.request_id, {"message": "问题不能为空。"}
                )
            elif state.active_task and not state.active_task.done():
                await day24.send(
                    ws, state, "busy", command.request_id, {"message": "当前连接已有任务。"}
                )
            else:
                state.active_request_id = command.request_id
                state.active_task = asyncio.create_task(
                    run_experiment_pipeline(ws, state, command),
                    name=f"canary-rag-{command.request_id}",
                )
    finally:
        if state.active_task and not state.active_task.done():
            state.active_task.cancel()
            await asyncio.gather(state.active_task, return_exceptions=True)
    return ws


async def feedback_handler(request: web.Request) -> web.Response:
    """同时写入Day25反馈闭环和Day26实验反馈事件。"""

    try:
        payload = day25.FeedbackPayload.model_validate(await request.json())
    except (ValidationError, ValueError, TypeError) as error:
        return web.json_response(
            {"ok": False, "error": "反馈参数无效。", "details": str(error)},
            status=400,
        )

    def persist() -> tuple[str, bool, bool]:
        with open_feedback_database() as connection:
            create_feedback_schema(connection)
            answer = connection.execute(
                "SELECT user_id FROM answer_runs WHERE request_id = ?",
                (payload.request_id,),
            ).fetchone()
            if answer is None:
                raise LookupError("找不到对应回答。")
            if answer["user_id"] != payload.user_id:
                raise PermissionError("不能评价其他用户的回答。")
            feedback_id, created = save_linked_feedback(
                connection,
                LinkedFeedback(
                    request_id=payload.request_id,
                    user_id=payload.user_id,
                    rating=payload.rating,
                    reason=payload.reason,
                ),
            )

        experiment_saved = False
        with open_experiment_database() as connection:
            exposure = connection.execute(
                "SELECT 1 FROM experiment_exposures WHERE request_id = ?",
                (payload.request_id,),
            ).fetchone()
            if exposure is not None:
                log_feedback(
                    connection,
                    FeedbackEventInput(
                        request_id=payload.request_id,
                        user_id=payload.user_id,
                        rating=payload.rating,
                        reason=payload.reason,
                    ),
                )
                experiment_saved = True
        return feedback_id, created, experiment_saved

    try:
        feedback_id, created, experiment_saved = await asyncio.to_thread(persist)
    except LookupError as error:
        return web.json_response({"ok": False, "error": str(error)}, status=404)
    except PermissionError as error:
        return web.json_response({"ok": False, "error": str(error)}, status=403)

    return web.json_response(
        {
            "ok": True,
            "operation": "created" if created else "updated",
            "feedback_id": feedback_id,
            "experiment_feedback_recorded": experiment_saved,
        },
        status=201 if created else 200,
    )


async def experiment_status_handler(request: web.Request) -> web.Response:
    del request
    with open_experiment_database() as connection:
        row = connection.execute(
            """
            SELECT status, allocation_version, updated_at
            FROM experiments WHERE experiment_id = ?
            """,
            (EXPERIMENT.experiment_id,),
        ).fetchone()
        counts = connection.execute(
            """
            SELECT variant, COUNT(*) AS exposures
            FROM experiment_exposures
            WHERE experiment_id = ? AND allocation_version = ?
            GROUP BY variant
            """,
            (EXPERIMENT.experiment_id, EXPERIMENT.allocation_version),
        ).fetchall()
    return web.json_response(
        {
            "experiment_id": EXPERIMENT.experiment_id,
            "status": row["status"],
            "allocation_version": row["allocation_version"],
            "updated_at": row["updated_at"],
            "exposures": {item["variant"]: item["exposures"] for item in counts},
        }
    )


async def file_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(
        {
            "/": HTML,
            "/app.js": JS,
            "/app.css": CSS,
            "/day24.css": DAY24_CSS,
            "/day25.css": DAY25_CSS,
        }[request.path]
    )


@web.middleware
async def errors(request: web.Request, handler: Any) -> web.StreamResponse:
    try:
        return await handler(request)
    except web.HTTPException as error:
        return web.json_response({"error": error.text or error.reason}, status=error.status)
    except Exception as error:
        print(f"[HTTP错误] {type(error).__name__}: {error}")
        return web.json_response({"error": "服务器内部错误。"}, status=500)


async def close_resources(app: web.Application) -> None:
    del app
    day24.cache.close()


def create_app() -> web.Application:
    app = web.Application(
        middlewares=[errors],
        client_max_size=day24.MAX_UPLOAD_SIZE + 65536,
    )
    for path in ("/", "/app.js", "/app.css", "/day24.css", "/day25.css"):
        app.router.add_get(path, file_handler)
    app.router.add_post("/upload", day24.upload_handler)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_post("/feedback", feedback_handler)
    app.router.add_get("/experiment/status", experiment_status_handler)
    app.on_cleanup.append(close_resources)
    return app


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    initialize_databases()
    print(f"Day26 灰度发布 Web RAG：http://{HOST}:{PORT}")
    print("实验：", EXPERIMENT.experiment_id)
    print("状态：", experiment_status())
    print("运行中按用户稳定分流；停止后全部回退control。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
