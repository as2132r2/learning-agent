"""Day23 第六步：带引用审计、质量评估和受控修复的最终Web RAG。"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aiohttp import WSMsgType, web
from pydantic import BaseModel, ConfigDict, Field, ValidationError


PROJECT_DIR = Path(__file__).resolve().parents[2]
DAY22_PYTHON = PROJECT_DIR / "day22" / "python"
DAY23_PYTHON = PROJECT_DIR / "day23" / "python"
for directory in (DAY22_PYTHON, DAY23_PYTHON):
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
from day23_answer_repair import (  # noqa: E402
    AnswerAssessment,
    MAX_REPAIR_ATTEMPTS,
    assess_answer,
    repair_answer,
)
from day23_citation_audit import generate_structured_answer  # noqa: E402
from day23_structured_citations import select_evidence  # noqa: E402


HOST, PORT = "127.0.0.1", 8030
DAY23_DIR = PROJECT_DIR / "day23"
HTML = DAY23_DIR / "html" / "day23_final.html"
JS = DAY23_DIR / "javascript" / "day23_final.js"
CSS = DAY23_DIR / "css" / "day23_final.css"

# 一条许可覆盖该请求的Reranker、生成、评估和可能的修复。
model_pipeline_semaphore = asyncio.Semaphore(2)


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
    async with state.send_lock:
        if not ws.closed:
            await ws.send_json(
                {
                    "type": kind,
                    "request_id": request_id,
                    "payload": payload,
                }
            )


def assessment_to_dict(
    assessment: AnswerAssessment,
    version_number: int,
    label: str,
) -> dict[str, Any]:
    """把多层Python审计对象转换成浏览器可接收的JSON数据。"""

    citation = assessment.citation_audit
    groundedness = assessment.groundedness_report
    quality = assessment.quality_response
    final = assessment.final_report

    return {
        "version_number": version_number,
        "label": label,
        "passed": assessment.passed,
        "rendered_answer": assessment.rendered_answer,
        "claims": [
            {
                "claim": claim.claim,
                "citation_numbers": claim.citation_numbers,
            }
            for claim in assessment.answer.claims
        ],
        "citation": {
            "passed": citation.passed,
            "coverage_rate": round(citation.claim_coverage_rate, 4),
            "used_evidence_numbers": citation.used_evidence_numbers,
            "unused_evidence_numbers": citation.unused_evidence_numbers,
            "errors": citation.errors,
        },
        "groundedness": None
        if groundedness is None
        else {
            "passed": groundedness.passed,
            "rate": round(groundedness.groundedness_rate, 4),
            "judgments": [
                {
                    "claim_number": item.claim_number,
                    "verdict": item.verdict,
                    "support_score": item.support_score,
                    "explanation": item.explanation,
                }
                for item in groundedness.judgments
            ],
            "errors": groundedness.errors,
        },
        "quality": None
        if quality is None
        else {
            "relevance_score": quality.relevance_score,
            "completeness_score": quality.completeness_score,
            "addressed_points": quality.addressed_points,
            "missing_points": quality.missing_points,
            "irrelevant_points": quality.irrelevant_points,
            "explanation": quality.explanation,
        },
        "final": None
        if final is None
        else {
            "passed": final.passed,
            "overall_score": round(final.overall_score, 2),
            "errors": final.errors,
        },
    }


async def run_audited_rag(
    ws: web.WebSocketResponse,
    state: ConnectionState,
    request_id: str,
    question: str,
) -> None:
    """执行召回、精排、回答、审计、一次修复和最终门控。"""

    acquired = False
    try:
        await send(ws, state, "retrieving", request_id, {})
        async with index_lock:
            rrf_results, diagnostics = await asyncio.to_thread(
                retrieve_rrf_candidates,
                question,
            )

        await send(
            ws,
            state,
            "rrf_results",
            request_id,
            {
                "results": [
                    rrf_to_dict(item, rank)
                    for rank, item in enumerate(rrf_results, start=1)
                ],
                **diagnostics,
            },
        )

        await send(ws, state, "pipeline_queued", request_id, {})
        await model_pipeline_semaphore.acquire()
        acquired = True

        reranked_results, _ = await asyncio.to_thread(
            rerank_candidates,
            question,
            rrf_results,
        )
        evidence = select_evidence(reranked_results)
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
                "results": [
                    reranked_to_dict(item, rank)
                    for rank, item in enumerate(reranked_results, start=1)
                ],
                "best_score": best_score,
                "threshold": RERANK_THRESHOLD,
                "accepted_count": len(evidence),
            },
        )

        if not evidence:
            await send(
                ws,
                state,
                "retrieval_rejected",
                request_id,
                {
                    "message": "没有资料达到Reranker守门阈值。",
                    "best_score": best_score,
                    "threshold": RERANK_THRESHOLD,
                },
            )
            return

        await send(ws, state, "drafting", request_id, {})
        initial_answer = await asyncio.to_thread(
            generate_structured_answer,
            question,
            evidence,
        )

        await send(ws, state, "auditing", request_id, {"version": 1})
        current_assessment = await asyncio.to_thread(
            assess_answer,
            question,
            initial_answer,
            evidence,
        )
        await send(
            ws,
            state,
            "version_assessed",
            request_id,
            assessment_to_dict(current_assessment, 1, "模型初稿"),
        )

        repair_attempt = 0
        if not current_assessment.passed:
            repair_attempt = 1
            await send(
                ws,
                state,
                "repairing",
                request_id,
                {"attempt": repair_attempt, "maximum": MAX_REPAIR_ATTEMPTS},
            )
            repaired_answer = await asyncio.to_thread(
                repair_answer,
                question,
                evidence,
                current_assessment,
            )
            await send(ws, state, "auditing", request_id, {"version": 2})
            current_assessment = await asyncio.to_thread(
                assess_answer,
                question,
                repaired_answer,
                evidence,
            )
            await send(
                ws,
                state,
                "version_assessed",
                request_id,
                assessment_to_dict(
                    current_assessment,
                    2,
                    "第一次修复稿",
                ),
            )

        if not current_assessment.passed:
            await send(
                ws,
                state,
                "quality_rejected",
                request_id,
                {
                    "message": "回答在一次修复后仍未通过质量门控。",
                    "repair_attempts": repair_attempt,
                },
            )
            return

        await send(
            ws,
            state,
            "done",
            request_id,
            {
                "answer": current_assessment.rendered_answer,
                "repair_attempts": repair_attempt,
                "sources": [
                    reranked_to_dict(item, rank)
                    for rank, item in enumerate(evidence, start=1)
                ],
            },
        )
    except asyncio.CancelledError:
        await send(ws, state, "cancelled", request_id, {})
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
            model_pipeline_semaphore.release()
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
        {
            "rerank_threshold": RERANK_THRESHOLD,
            "max_repair_attempts": MAX_REPAIR_ATTEMPTS,
        },
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
                    await send(ws, state, "cancel_not_found", command.request_id, {})
                continue

            question = (command.payload.question or "").strip()
            if not question:
                await send(
                    ws, state, "error", command.request_id,
                    {"message": "问题不能为空。"},
                )
            elif state.active_task and not state.active_task.done():
                await send(
                    ws, state, "busy", command.request_id,
                    {"message": "当前连接已有质量评估流程正在执行。"},
                )
            else:
                state.active_request_id = command.request_id
                state.active_task = asyncio.create_task(
                    run_audited_rag(ws, state, command.request_id, question),
                    name=f"audited-rag-{command.request_id}",
                )
    finally:
        if state.active_task and not state.active_task.done():
            state.active_task.cancel()
            await asyncio.gather(state.active_task, return_exceptions=True)
    return ws


async def file_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(
        {"/": HTML, "/app.js": JS, "/app.css": CSS}[request.path]
    )


@web.middleware
async def errors(request: web.Request, handler: Any) -> web.StreamResponse:
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
    print(f"Day23 最终可审计Web RAG：http://{HOST}:{PORT}")
    print("向量数据库：", VECTOR_DATABASE_PATH)
    print("FTS数据库：", FTS_DATABASE_PATH)
    print("Reranker阈值：", RERANK_THRESHOLD)
    print("最大修复次数：", MAX_REPAIR_ATTEMPTS)
    web.run_app(create_app(), host=HOST, port=PORT, print=None)
