"""Day17 第六步（服务端）：用 Semaphore 限制 SSE 模型流并发数。

运行方式：
    python day17/python/day17_sse_limited_server.py

服务地址：
    http://127.0.0.1:8006

与上一版的区别：
    1. HTTP/SSE 连接仍然可以同时建立；
    2. 每个请求先进入等待队列；
    3. 只有获得 Semaphore 许可后才能调用 llm.astream()；
    4. 最多同时运行 MAX_MODEL_CONCURRENCY 条模型流；
    5. 模型流结束或异常时自动释放许可。
"""

from __future__ import annotations

import asyncio
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
PORT = 8006
MAX_MODEL_CONCURRENCY = 2


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


# Semaphore 内部维护许可数量。
# 初始值为 2，表示最多两个协程能同时进入模型调用区域。
model_semaphore = asyncio.Semaphore(MAX_MODEL_CONCURRENCY)

# 以下计数只用于观察运行状态，不参与真正的并发控制。
active_model_streams = 0
waiting_model_streams = 0

# 多个协程修改统计变量前先获取 Lock，避免更新过程相互覆盖。
statistics_lock = asyncio.Lock()


def encode_sse(
    *,
    event: str,
    data: dict[str, Any],
    event_id: str | None = None,
) -> bytes:
    """把事件名称、JSON 数据和可选 ID 编码成 SSE 字节。"""

    lines: list[str] = []

    if event_id is not None:
        lines.append(f"id: {event_id}")

    lines.append(f"event: {event}")
    lines.append(
        "data: " + json.dumps(data, ensure_ascii=False)
    )

    return ("\n".join(lines) + "\n\n").encode("utf-8")


async def send_event(
    response: web.StreamResponse,
    *,
    event: str,
    data: dict[str, Any],
    event_id: str | None = None,
) -> None:
    """向当前 SSE 连接立即写入一个事件。"""

    await response.write(
        encode_sse(
            event=event,
            data=data,
            event_id=event_id,
        )
    )


async def change_waiting(delta: int) -> int:
    """原子更新等待数量，并返回更新后的值。"""

    global waiting_model_streams

    async with statistics_lock:
        waiting_model_streams += delta
        return waiting_model_streams


async def change_active(delta: int) -> int:
    """原子更新活动模型流数量，并返回更新后的值。"""

    global active_model_streams

    async with statistics_lock:
        active_model_streams += delta
        return active_model_streams


async def get_statistics() -> tuple[int, int]:
    """在同一把锁中取得活动数量和等待数量。"""

    async with statistics_lock:
        return active_model_streams, waiting_model_streams


async def health_handler(request: web.Request) -> web.Response:
    """返回服务状态和当前并发统计。"""

    active, waiting = await get_statistics()

    return web.json_response(
        {
            "status": "ok",
            "service": "day17-limited-sse-server",
            "model": llm.model_name,
            "max_model_concurrency": MAX_MODEL_CONCURRENCY,
            "active_model_streams": active,
            "waiting_model_streams": waiting,
        }
    )


async def stream_handler(request: web.Request) -> web.StreamResponse:
    """建立 SSE 连接，排队获取许可，再流式调用模型。"""

    question = request.query.get("question", "").strip()

    if not question:
        return web.json_response(
            {"error": "查询参数 question 不能为空。"},
            status=400,
        )

    request_id = "req-" + uuid.uuid4().hex[:12]
    connected_at = time.perf_counter()

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

    try:
        current_waiting = await change_waiting(1)

        print(
            f"[进入队列] request_id={request_id} "
            f"waiting={current_waiting} question={question!r}"
        )

        # queued 事件让客户端知道连接正常，只是在等待模型调用名额。
        await send_event(
            response,
            event="queued",
            event_id=f"{request_id}:queued",
            data={
                "request_id": request_id,
                "waiting": current_waiting,
                "max_model_concurrency": MAX_MODEL_CONCURRENCY,
            },
        )

        # 如果两个许可已经被占用，此 await 会挂起当前请求协程。
        # 挂起期间不会阻塞事件循环，其他 HTTP 请求仍可被处理。
        await model_semaphore.acquire()
        acquired = True

        await change_waiting(-1)
        current_active = await change_active(1)

        model_started_at = time.perf_counter()
        queue_wait_seconds = model_started_at - connected_at

        print(
            f"[获得许可] request_id={request_id} "
            f"active={current_active} "
            f"queue_wait={queue_wait_seconds:.3f}s"
        )

        await send_event(
            response,
            event="start",
            event_id=f"{request_id}:0",
            data={
                "request_id": request_id,
                "model": llm.model_name,
                "question": question,
                "queue_wait_seconds": round(queue_wait_seconds, 3),
                "active_model_streams": current_active,
            },
        )

        messages = [
            SystemMessage(
                content="你是一名简洁、友好的编程学习助手。"
            ),
            HumanMessage(content=question),
        ]

        chunk_count = 0
        answer_length = 0

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

        await send_event(
            response,
            event="done",
            event_id=f"{request_id}:{chunk_count + 1}",
            data={
                "request_id": request_id,
                "chunk_count": chunk_count,
                "answer_length": answer_length,
                "model_seconds": round(
                    time.perf_counter() - model_started_at,
                    3,
                ),
            },
        )

        print(
            f"[模型完成] request_id={request_id} "
            f"chunks={chunk_count}"
        )

    except (ConnectionResetError, asyncio.CancelledError):
        print(f"[客户端断开] request_id={request_id}")

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
        if acquired:
            current_active = await change_active(-1)
            model_semaphore.release()
            print(
                f"[释放许可] request_id={request_id} "
                f"active={current_active}"
            )
        else:
            # 如果在排队期间取消，撤销前面增加的 waiting 计数。
            await change_waiting(-1)

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
    """显示每个到达服务端的 HTTP 请求。"""

    print(f"[HTTP] {request.method} {request.path_qs}")
    return await handler(request)


def create_app() -> web.Application:
    """创建带并发限制的 SSE 应用。"""

    app = web.Application(middlewares=[logging_middleware])
    app.router.add_get("/health", health_handler)
    app.router.add_get("/stream", stream_handler)
    return app


def main() -> None:
    """启动 8006 端口上的限流 SSE Server。"""

    print("Day17 限制并发的 SSE Server 正在启动。")
    print(f"服务地址：http://{HOST}:{PORT}")
    print("最大模型并发数：", MAX_MODEL_CONCURRENCY)
    print("按 Ctrl+C 停止服务。")

    web.run_app(
        create_app(),
        host=HOST,
        port=PORT,
        print=None,
    )


if __name__ == "__main__":
    main()
