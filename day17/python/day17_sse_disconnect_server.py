"""Day17 第九步（服务端）：主动监测客户端断线并取消模型任务。

运行方式：
    python day17/python/day17_sse_disconnect_server.py

服务地址：
    http://127.0.0.1:8007

上一版主要在 response.write() 时发现客户端断线。如果模型迟迟没有返回
首个 chunk，服务端暂时没有 write()，就可能较晚发现断线。

本版同时运行：
    1. model_stream_task：读取模型流并写 SSE；
    2. disconnect_task：定期检查 HTTP transport 是否关闭。

任何一个任务先结束，服务端都会取消和回收另一个任务。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from typing import Any

from aiohttp import web
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()


HOST = "127.0.0.1"
PORT = 8007
MAX_MODEL_CONCURRENCY = 2
DISCONNECT_POLL_SECONDS = 0.1


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


model_semaphore = asyncio.Semaphore(MAX_MODEL_CONCURRENCY)
statistics_lock = asyncio.Lock()
active_model_streams = 0
waiting_model_streams = 0
cancelled_model_streams = 0


def encode_sse(
    *,
    event: str,
    data: dict[str, Any],
    event_id: str | None = None,
) -> bytes:
    """把 Python 字典编码成一个 SSE 事件。"""

    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append("data: " + json.dumps(data, ensure_ascii=False))
    return ("\n".join(lines) + "\n\n").encode("utf-8")


async def send_event(
    response: web.StreamResponse,
    *,
    event: str,
    data: dict[str, Any],
    event_id: str | None = None,
) -> None:
    """向客户端写入一个 SSE 事件。"""

    await response.write(
        encode_sse(event=event, data=data, event_id=event_id)
    )


async def update_statistics(
    *,
    active_delta: int = 0,
    waiting_delta: int = 0,
    cancelled_delta: int = 0,
) -> tuple[int, int, int]:
    """在 Lock 保护下更新并返回三项统计数据。"""

    global active_model_streams
    global waiting_model_streams
    global cancelled_model_streams

    async with statistics_lock:
        active_model_streams += active_delta
        waiting_model_streams += waiting_delta
        cancelled_model_streams += cancelled_delta
        return (
            active_model_streams,
            waiting_model_streams,
            cancelled_model_streams,
        )


async def get_statistics() -> tuple[int, int, int]:
    """取得当前活动、等待和累计取消数量。"""

    async with statistics_lock:
        return (
            active_model_streams,
            waiting_model_streams,
            cancelled_model_streams,
        )


async def wait_for_disconnect(request: web.Request) -> None:
    """定期检查客户端 transport，直到连接关闭。"""

    while True:
        transport = request.transport

        if transport is None or transport.is_closing():
            return

        await asyncio.sleep(DISCONNECT_POLL_SECONDS)


async def produce_model_stream(
    *,
    response: web.StreamResponse,
    request_id: str,
    question: str,
) -> dict[str, Any]:
    """读取模型 chunk、发送 token，并返回完成统计。"""

    messages = [
        SystemMessage(
            content=(
                "你是一名 Python 编程学习助手。"
                "只使用 Python 术语，不要使用 JavaScript 术语。"
            )
        ),
        HumanMessage(content=question),
    ]

    chunk_count = 0
    answer_length = 0
    started_at = time.perf_counter()

    async for chunk in llm.astream(messages):
        if isinstance(chunk.content, str):
            text = chunk.content
        else:
            text = str(chunk.content)

        if not text:
            continue

        chunk_count += 1
        answer_length += len(text)

        await send_event(
            response,
            event="token",
            event_id=f"{request_id}:{chunk_count}",
            data={
                "request_id": request_id,
                "index": chunk_count,
                "text": text,
            },
        )

    return {
        "chunk_count": chunk_count,
        "answer_length": answer_length,
        "model_seconds": round(time.perf_counter() - started_at, 3),
    }


async def health_handler(request: web.Request) -> web.Response:
    """显示并发数量和累计主动取消次数。"""

    active, waiting, cancelled = await get_statistics()
    return web.json_response(
        {
            "status": "ok",
            "service": "day17-disconnect-aware-sse-server",
            "model": llm.model_name,
            "max_model_concurrency": MAX_MODEL_CONCURRENCY,
            "active_model_streams": active,
            "waiting_model_streams": waiting,
            "cancelled_model_streams": cancelled,
        }
    )


async def stream_handler(request: web.Request) -> web.StreamResponse:
    """竞争等待模型完成或客户端断线，并进行统一清理。"""

    question = request.query.get("question", "").strip()
    if not question:
        return web.json_response(
            {"error": "查询参数 question 不能为空。"},
            status=400,
        )

    request_id = "req-" + uuid.uuid4().hex[:12]
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    acquired = False
    waiting_registered = False
    model_task: asyncio.Task[dict[str, Any]] | None = None
    disconnect_task: asyncio.Task[None] | None = None

    try:
        _, waiting, _ = await update_statistics(waiting_delta=1)
        waiting_registered = True

        await send_event(
            response,
            event="queued",
            event_id=f"{request_id}:queued",
            data={
                "request_id": request_id,
                "waiting": waiting,
            },
        )

        await model_semaphore.acquire()
        acquired = True

        active, _, _ = await update_statistics(
            active_delta=1,
            waiting_delta=-1,
        )
        waiting_registered = False

        await send_event(
            response,
            event="start",
            event_id=f"{request_id}:0",
            data={
                "request_id": request_id,
                "active_model_streams": active,
            },
        )

        model_task = asyncio.create_task(
            produce_model_stream(
                response=response,
                request_id=request_id,
                question=question,
            ),
            name=f"model-stream-{request_id}",
        )
        disconnect_task = asyncio.create_task(
            wait_for_disconnect(request),
            name=f"disconnect-watch-{request_id}",
        )

        # FIRST_COMPLETED：模型完成或断线监测完成，谁先发生就立即返回。
        done, pending = await asyncio.wait(
            {model_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if disconnect_task in done and not model_task.done():
            print(f"[主动发现断线] request_id={request_id}")
            await update_statistics(cancelled_delta=1)

            model_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await model_task

        else:
            # result() 会把模型任务中的异常重新抛到当前处理函数。
            result = model_task.result()

            await send_event(
                response,
                event="done",
                event_id=f"{request_id}:{result['chunk_count'] + 1}",
                data={"request_id": request_id, **result},
            )
            print(
                f"[模型正常完成] request_id={request_id} "
                f"chunks={result['chunk_count']}"
            )

        # 回收仍在运行的辅助任务，防止后台 Task 泄漏。
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    except (ConnectionResetError, asyncio.CancelledError):
        print(f"[连接结束] request_id={request_id}")

    except Exception as error:
        print(
            f"[请求错误] request_id={request_id} "
            f"{type(error).__name__}: {error}"
        )
        try:
            await send_event(
                response,
                event="error",
                data={
                    "request_id": request_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
        except (ConnectionResetError, RuntimeError):
            pass

    finally:
        # 无论正常、断线、上游错误还是协程取消，都回收遗留 Task。
        for task in (model_task, disconnect_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        if waiting_registered:
            await update_statistics(waiting_delta=-1)

        if acquired:
            active, _, _ = await update_statistics(active_delta=-1)
            model_semaphore.release()
            print(
                f"[释放许可] request_id={request_id} active={active}"
            )

        try:
            await response.write_eof()
        except (ConnectionResetError, RuntimeError):
            pass

    return response


@web.middleware
async def logging_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    """记录进入服务端的 HTTP 请求。"""

    print(f"[HTTP] {request.method} {request.path_qs}")
    return await handler(request)


def create_app() -> web.Application:
    """创建支持主动断线检测的 SSE 应用。"""

    app = web.Application(middlewares=[logging_middleware])
    app.router.add_get("/health", health_handler)
    app.router.add_get("/stream", stream_handler)
    return app


def main() -> None:
    """启动服务。"""

    print("Day17 主动断线检测 SSE Server 正在启动。")
    print(f"服务地址：http://{HOST}:{PORT}")
    print("最大模型并发数：", MAX_MODEL_CONCURRENCY)
    print("断线检测间隔：", DISCONNECT_POLL_SECONDS, "秒")
    print("按 Ctrl+C 停止服务。")

    web.run_app(
        create_app(),
        host=HOST,
        port=PORT,
        print=None,
    )


if __name__ == "__main__":
    main()
