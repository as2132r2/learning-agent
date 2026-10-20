"""Day24 第六步：缓存、并行、风险分级与预算控制的最终Web RAG。"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aiohttp import WSMsgType, web
from pydantic import BaseModel, ConfigDict, Field, ValidationError


PROJECT_DIR = Path(__file__).resolve().parents[2]
for directory in (
    PROJECT_DIR / "day22" / "python",
    PROJECT_DIR / "day23" / "python",
    PROJECT_DIR / "day24" / "python",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from day22_final_hybrid_rag import (  # noqa: E402
    FTS_DATABASE_PATH,
    MAX_UPLOAD_SIZE,
    RERANK_THRESHOLD,
    VECTOR_DATABASE_PATH,
    index_lock,
    reranked_to_dict,
    retrieve_rrf_candidates,
    rrf_to_dict,
    upload_handler,
)
from day22_llm_reranker import rerank_candidates  # noqa: E402
from day23_answer_quality import (  # noqa: E402
    build_final_quality_report,
    evaluate_answer_quality,
)
from day23_answer_repair import (  # noqa: E402
    AnswerAssessment,
    assessment_feedback,
    repair_answer,
)
from day23_citation_audit import audit_citations, generate_structured_answer  # noqa: E402
from day23_claim_groundedness import (  # noqa: E402
    build_groundedness_report,
    evaluate_claim_support,
)
from day23_structured_citations import build_readable_answer, select_evidence  # noqa: E402
from day24_adaptive_audit import POLICIES, AuditPolicy, RiskDecision, classify_risk  # noqa: E402
from day24_budget_controller import (  # noqa: E402
    BudgetController,
    BudgetExceeded,
    STAGE_TOKEN_ESTIMATES,
    choose_executable_policy,
)
from day24_parallel_evaluators import ModelCallResult, measured_model_call  # noqa: E402
from day24_pipeline_cache import (  # noqa: E402
    DATABASE_PATH as CACHE_DATABASE_PATH,
    ModelCache,
    deserialize_answer,
    deserialize_groundedness,
    deserialize_quality,
    deserialize_reranker,
    evidence_key_data,
    serialize_pydantic,
    serialize_reranker,
)


HOST, PORT = "127.0.0.1", 8031
DAY24_DIR = PROJECT_DIR / "day24"
HTML = DAY24_DIR / "html" / "day24_final.html"
JS = DAY24_DIR / "javascript" / "day24_final.js"
CSS = DAY24_DIR / "css" / "day24_final.css"
PIPELINE_LIMIT = 2
REPAIR_TOKEN_ESTIMATE = 900

model_pipeline_semaphore = asyncio.Semaphore(PIPELINE_LIMIT)
cache = ModelCache(CACHE_DATABASE_PATH)
cache_lock = asyncio.Lock()


class AskPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str | None = Field(None, max_length=2000)
    risk: Literal["auto", "low", "medium", "high"] = "auto"
    max_calls: int = Field(default=7, ge=0, le=20)
    max_tokens: int = Field(default=8500, ge=0, le=100000)
    allow_downgrade: bool = False


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


@dataclass
class ExecutionTrace:
    stages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def cache_hits(self) -> int:
        return sum(stage["cache_hit"] for stage in self.stages)

    @property
    def model_calls(self) -> int:
        return sum(stage["model_calls"] for stage in self.stages)

    @property
    def total_tokens(self) -> int:
        return sum(stage["total_tokens"] for stage in self.stages)


async def send(
    ws: web.WebSocketResponse,
    state: ConnectionState,
    kind: str,
    request_id: str | None,
    payload: dict[str, Any],
) -> None:
    async with state.send_lock:
        if not ws.closed:
            await ws.send_json({"type": kind, "request_id": request_id, "payload": payload})


async def async_cached_model_call(
    trace: ExecutionTrace,
    controller: BudgetController,
    stage_name: str,
    cache_input: Any,
    function,
    function_args: tuple[Any, ...],
    serializer,
    deserializer,
    estimated_tokens: int,
) -> Any:
    """异步服务中的cache-aside模式：命中不占模型预算。"""

    cache_key = cache.make_key(stage_name, cache_input)
    started = time.perf_counter()
    async with cache_lock:
        payload = cache.get(cache_key)

    if payload is not None:
        value = deserializer(payload)
        trace.stages.append({
            "name": stage_name,
            "cache_hit": True,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "model_calls": 0,
            "total_tokens": 0,
            "cache_key": cache_key[:12],
        })
        return value

    controller.reserve(stage_name, estimated_tokens)
    call: ModelCallResult = await asyncio.to_thread(
        measured_model_call,
        stage_name,
        function,
        *function_args,
    )
    controller.complete(call)
    async with cache_lock:
        cache.put(cache_key, stage_name, serializer(call.value))
    trace.stages.append({
        "name": stage_name,
        "cache_hit": False,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "model_calls": call.model_calls,
        "total_tokens": call.total_tokens,
        "cache_key": cache_key[:12],
    })
    return call.value


def assessment_payload(
    assessment: AnswerAssessment,
    version: int,
    label: str,
    policy_result: bool,
) -> dict[str, Any]:
    grounded = assessment.groundedness_report
    quality = assessment.quality_response
    final = assessment.final_report
    return {
        "version": version,
        "label": label,
        "passed": policy_result,
        "answer": assessment.rendered_answer,
        "citation_coverage": assessment.citation_audit.claim_coverage_rate,
        "groundedness": None if grounded is None else grounded.groundedness_rate,
        "relevance": None if quality is None else quality.relevance_score,
        "completeness": None if quality is None else quality.completeness_score,
        "overall": None if final is None else final.overall_score,
        "errors": [] if final is None else final.errors,
    }


async def audit_with_policy(
    question: str,
    answer,
    evidence,
    policy: AuditPolicy,
    trace: ExecutionTrace,
    controller: BudgetController,
) -> AnswerAssessment:
    """按策略审计；高风险的两个评估器并行且都支持缓存。"""

    rendered = build_readable_answer(answer)
    citation = audit_citations(answer, len(evidence), rendered)
    if not citation.passed:
        return AnswerAssessment(answer, rendered, citation, None, None, None)

    evidence_data = evidence_key_data(evidence)
    grounded = None
    quality = None

    if policy.run_groundedness and policy.run_answer_quality:
        grounded_task = asyncio.create_task(
            async_cached_model_call(
                trace, controller, "Groundedness",
                {"answer": answer.model_dump(mode="json"), "evidence": evidence_data},
                evaluate_claim_support, (answer, evidence),
                serialize_pydantic, deserialize_groundedness,
                STAGE_TOKEN_ESTIMATES["Groundedness"],
            ),
            name="optimized-groundedness",
        )
        quality_task = asyncio.create_task(
            async_cached_model_call(
                trace, controller, "AnswerQuality",
                {"question": question, "answer": answer.model_dump(mode="json"),
                 "evidence": evidence_data},
                evaluate_answer_quality, (question, answer, evidence),
                serialize_pydantic, deserialize_quality,
                STAGE_TOKEN_ESTIMATES["AnswerQuality"],
            ),
            name="optimized-answer-quality",
        )
        grounded_response, quality = await asyncio.gather(grounded_task, quality_task)
        grounded = build_groundedness_report(answer, grounded_response)
    elif policy.run_groundedness:
        grounded_response = await async_cached_model_call(
            trace, controller, "Groundedness",
            {"answer": answer.model_dump(mode="json"), "evidence": evidence_data},
            evaluate_claim_support, (answer, evidence),
            serialize_pydantic, deserialize_groundedness,
            STAGE_TOKEN_ESTIMATES["Groundedness"],
        )
        grounded = build_groundedness_report(answer, grounded_response)

    if quality is not None and grounded is not None:
        final = build_final_quality_report(citation, grounded, quality)
    else:
        final = None

    return AnswerAssessment(answer, rendered, citation, grounded, quality, final)


def policy_passed(assessment: AnswerAssessment, policy: AuditPolicy) -> bool:
    if not assessment.citation_audit.passed:
        return False
    if policy.run_answer_quality:
        return assessment.final_report is not None and assessment.final_report.passed
    if policy.run_groundedness:
        return (
            assessment.groundedness_report is not None
            and assessment.groundedness_report.passed
        )
    return True


async def run_pipeline(
    ws: web.WebSocketResponse,
    state: ConnectionState,
    request_id: str,
    payload: AskPayload,
) -> None:
    acquired = False
    started = time.perf_counter()
    trace = ExecutionTrace()
    controller = BudgetController(payload.max_calls, payload.max_tokens)
    question = (payload.question or "").strip()

    try:
        decision: RiskDecision = classify_risk(question, payload.risk)
        policy, downgrade = choose_executable_policy(
            decision, payload.max_calls, payload.max_tokens, payload.allow_downgrade
        )
        await send(ws, state, "plan", request_id, {
            "requested_risk": decision.risk_level,
            "executed_risk": None if policy is None else policy.risk_level,
            "reasons": decision.reasons,
            "downgrade_reason": downgrade,
            "max_calls": payload.max_calls,
            "max_tokens": payload.max_tokens,
        })
        if policy is None:
            await send(ws, state, "budget_rejected", request_id, {"message": downgrade})
            return

        await send(ws, state, "retrieving", request_id, {})
        async with index_lock:
            rrf_results, diagnostics = await asyncio.to_thread(
                retrieve_rrf_candidates, question
            )
        await send(ws, state, "rrf_results", request_id, {
            "results": [rrf_to_dict(item, rank) for rank, item in enumerate(rrf_results, 1)],
            **diagnostics,
        })

        await model_pipeline_semaphore.acquire()
        acquired = True
        reranker_result = await async_cached_model_call(
            trace, controller, "Reranker",
            {"question": question, "candidates": [rrf_to_dict(i, n) for n, i in enumerate(rrf_results, 1)]},
            rerank_candidates, (question, rrf_results),
            serialize_reranker, deserialize_reranker,
            STAGE_TOKEN_ESTIMATES["Reranker"],
        )
        reranked, _ = reranker_result
        evidence = select_evidence(reranked)
        await send(ws, state, "reranked", request_id, {
            "results": [reranked_to_dict(item, rank) for rank, item in enumerate(reranked, 1)],
            "accepted_count": len(evidence), "threshold": RERANK_THRESHOLD,
        })
        if not evidence:
            await send(ws, state, "retrieval_rejected", request_id,
                       {"message": "没有资料达到相关性阈值。"})
            return

        evidence_data = evidence_key_data(evidence)
        answer = await async_cached_model_call(
            trace, controller, "StructuredAnswer",
            {"question": question, "evidence": evidence_data},
            generate_structured_answer, (question, evidence),
            serialize_pydantic, deserialize_answer,
            STAGE_TOKEN_ESTIMATES["StructuredAnswer"],
        )
        assessment = await audit_with_policy(
            question, answer, evidence, policy, trace, controller
        )
        first_passed = policy_passed(assessment, policy)
        await send(ws, state, "version_assessed", request_id,
                   assessment_payload(assessment, 1, "模型初稿", first_passed))

        repair_count = 0
        if not policy_passed(assessment, policy):
            repair_count = 1
            await send(ws, state, "repairing", request_id, {})
            feedback = assessment_feedback(assessment)
            repaired = await async_cached_model_call(
                trace, controller, "RepairAnswer",
                {"question": question, "evidence": evidence_data,
                 "answer": answer.model_dump(mode="json"), "feedback": feedback},
                repair_answer, (question, evidence, assessment),
                serialize_pydantic, deserialize_answer, REPAIR_TOKEN_ESTIMATE,
            )
            assessment = await audit_with_policy(
                question, repaired, evidence, policy, trace, controller
            )
            repaired_passed = policy_passed(assessment, policy)
            await send(ws, state, "version_assessed", request_id,
                       assessment_payload(
                           assessment, 2, "第一次修复稿", repaired_passed
                       ))

        summary = {
            "status": "success" if policy_passed(assessment, policy) else "quality_rejected",
            "answer": assessment.rendered_answer if policy_passed(assessment, policy) else None,
            "repair_count": repair_count,
            "wall_seconds": round(time.perf_counter() - started, 3),
            "cache_hits": trace.cache_hits,
            "model_calls": trace.model_calls,
            "total_tokens": trace.total_tokens,
            "stages": trace.stages,
            "reserved_calls": controller.reserved_calls,
            "reserved_tokens": controller.reserved_tokens,
            "actual_calls": controller.actual_calls,
            "actual_tokens": controller.actual_tokens,
        }
        await send(ws, state, "done" if summary["status"] == "success" else "quality_rejected",
                   request_id, summary)
    except BudgetExceeded as error:
        await send(ws, state, "budget_exceeded", request_id, {"message": str(error)})
    except asyncio.CancelledError:
        await send(ws, state, "cancelled", request_id, {})
    except Exception as error:
        await send(ws, state, "error", request_id,
                   {"error_type": type(error).__name__, "message": str(error)})
    finally:
        if acquired:
            model_pipeline_semaphore.release()
        if state.active_request_id == request_id:
            state.active_request_id = None
            state.active_task = None


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    state = ConnectionState()
    await send(ws, state, "connected", None, {
        "cache_entries": cache.count(), "pipeline_limit": PIPELINE_LIMIT
    })
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
            if not (command.payload.question or "").strip():
                await send(ws, state, "error", command.request_id,
                           {"message": "问题不能为空。"})
            elif state.active_task and not state.active_task.done():
                await send(ws, state, "busy", command.request_id,
                           {"message": "当前连接已有任务。"})
            else:
                state.active_request_id = command.request_id
                state.active_task = asyncio.create_task(
                    run_pipeline(ws, state, command.request_id, command.payload),
                    name=f"optimized-rag-{command.request_id}",
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


async def close_resources(app: web.Application) -> None:
    cache.close()


def create_app() -> web.Application:
    app = web.Application(middlewares=[errors], client_max_size=MAX_UPLOAD_SIZE + 65536)
    app.router.add_get("/", file_handler)
    app.router.add_get("/app.js", file_handler)
    app.router.add_get("/app.css", file_handler)
    app.router.add_post("/upload", upload_handler)
    app.router.add_get("/ws", websocket_handler)
    app.on_cleanup.append(close_resources)
    return app


if __name__ == "__main__":
    print(f"Day24 最终优化Web RAG：http://{HOST}:{PORT}")
    print("向量数据库：", VECTOR_DATABASE_PATH)
    print("FTS数据库：", FTS_DATABASE_PATH)
    print("模型缓存：", CACHE_DATABASE_PATH, "已有", cache.count(), "条")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)
