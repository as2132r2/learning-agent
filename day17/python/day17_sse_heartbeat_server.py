"""Day17 第十一步（服务端）：SSE 心跳、Queue 与背压。

运行方式：
    python day17/python/day17_sse_heartbeat_server.py

服务地址：
    http://127.0.0.1:8008

本步骤重点：
    1. 模型生产者只负责读取 llm.astream()；
    2. HTTP 消费者只负责从 Queue 取数据并写入 SSE；
    3. 模型暂时无数据时发送 SSE 注释心跳；
    4. 有限 Queue 满时，await queue.put() 会让生产者等待，形成背压；
    5. 客户端断开后取消模型生产者并释放 Semaphore。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from aiohttp import web
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()


HOST = "127.0.0.1"
PORT = 8008
MAX_MODEL_CONCURRENCY = 2
HEARTBEAT_INTERVAL_SECONDS = 0.5
QUEUE_MAX_SIZE = 20


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


model_semaphore = asyncio.Semaphore(MAX_MODEL_CONCURRENCY)


@dataclass
class QueueItem:
    """模型生产者交给 SSE 消费者的一条内部消息。"""

    kind: Literal["token", "done", "error"]
    data: dict[str, Any]


def encode_sse(
    *,
    event: str,
    data: dict[str, Any],
    event_id: str | None = None,
) -> bytes:
    """编码普通 SSE 事件。"""

    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append("data: " + json.dumps(data, ensure_ascii=False))
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def encode_heartbeat() -> bytes:
    """编码 SSE 注释心跳；冒号开头的行不会成为业务事件。"""

    timestamp = round(time.time(), 3)
    return f": heartbeat {timestamp}\n\n".encode("utf-8")


async def produce_chunks(
    *,
    question: str,
    request_id: str,
    queue: asyncio.Queue[QueueItem],
) -> None:
    """读取模型流并把内部消息放入有限 Queue。"""

    messages = [
        SystemMessage(
            content=(
                "你是一名 Python 编程学习助手。"
                "回答准确、简洁，只使用 Python 术语。"
            )
        ),
        HumanMessage(content=question),
    ]

    chunk_count = 0
    answer_length = 0
    started_at = time.perf_counter()

    try:
        async for chunk in llm.astream(messages):
            if isinstance(chunk.content, str):
                text = chunk.content
            else:
                text = str(chunk.content)

            if not text:
                continue

            chunk_count += 1
            answer_length += len(text)

            # Queue 已满时，这里会异步等待消费者腾出空间。
            await queue.put(
                QueueItem(
                    kind="token",
                    data={
                        "request_id": request_id,
                        "index": chunk_count,
                        "text": text,
                    },
                )
            )

        await queue.put(
            QueueItem(
                kind="done",
                data={
                    "request_id": request_id,
                    "chunk_count": chunk_count,
                    "answer_length": answer_length,
                    "model_seconds": round(
                        time.perf_counter() - started_at,
                        3,
                    ),
                },
            )
        )

    except asyncio.CancelledError:
        print(f"[模型生产者取消] request_id={request_id}")
        raise

    except Exception as error:
        await queue.put(
            QueueItem(
                kind="error",
                data={
                    "request_id": request_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
        )


async def health_handler(request: web.Request) -> web.Response:
    """返回心跳、队列和模型并发配置。"""

    return web.json_response(
        {
            "status": "ok",
            "service": "day17-heartbeat-sse-server",
            "model": llm.model_name,
            "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
            "queue_max_size": QUEUE_MAX_SIZE,
            "max_model_concurrency": MAX_MODEL_CONCURRENCY,
        }
    )


async def stream_handler(request: web.Request) -> web.StreamResponse:
    """消费内部 Queue，发送 token 或定时发送心跳。"""

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

    producer_task: asyncio.Task[None] | None = None
    acquired = False

    try:
        await response.write(
            encode_sse(
                event="queued",
                event_id=f"{request_id}:queued",
                data={
                    "request_id": request_id,
                    "max_model_concurrency": MAX_MODEL_CONCURRENCY,
                },
            )
        )

        await model_semaphore.acquire()
        acquired = True

        await response.write(
            encode_sse(
                event="start",
                event_id=f"{request_id}:0",
                data={
                    "request_id": request_id,
                    "heartbeat_interval_seconds": (
                        HEARTBEAT_INTERVAL_SECONDS
                    ),
                    "queue_max_size": QUEUE_MAX_SIZE,
                },
            )
        )

        queue: asyncio.Queue[QueueItem] = asyncio.Queue(
            maxsize=QUEUE_MAX_SIZE
        )

        producer_task = asyncio.create_task(
            produce_chunks(
                question=question,
                request_id=request_id,
                queue=queue,
            ),
            name=f"producer-{request_id}",
        )

        token_event_count = 0
        heartbeat_count = 0

        while True:
            try:
                # 如果期限内没有 QueueItem，就发送一次心跳。
                item = await asyncio.wait_for(
                    queue.get(),
                    timeout=HEARTBEAT_INTERVAL_SECONDS,
                )
            except TimeoutError:
                heartbeat_count += 1
                await response.write(encode_heartbeat())
                print(
                    f"[发送心跳] request_id={request_id} "
                    f"count={heartbeat_count}"
                )
                continue

            if item.kind == "token":
                token_event_count += 1
                await response.write(
                    encode_sse(
                        event="token",
                        event_id=f"{request_id}:{token_event_count}",
                        data=item.data,
                    )
                )

            elif item.kind == "done":
                await response.write(
                    encode_sse(
                        event="done",
                        event_id=f"{request_id}:{token_event_count + 1}",
                        data={
                            **item.data,
                            "heartbeat_count": heartbeat_count,
                        },
                    )
                )
                print(
                    f"[流完成] request_id={request_id} "
                    f"heartbeats={heartbeat_count} "
                    f"tokens={token_event_count}"
                )
                queue.task_done()
                break

            elif item.kind == "error":
                await response.write(
                    encode_sse(event="error", data=item.data)
                )
                print(
                    f"[模型错误] request_id={request_id} "
                    f"{item.data['error_type']}: {item.data['message']}"
                )
                queue.task_done()
                break

            # 告知 Queue 当前项目已经处理完，供 join() 等机制使用。
            queue.task_done()

    except (ConnectionResetError, asyncio.CancelledError):
        print(f"[客户端断开] request_id={request_id}")

    finally:
        if producer_task is not None and not producer_task.done():
            producer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer_task

        if acquired:
            model_semaphore.release()
            print(f"[释放模型许可] request_id={request_id}")

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
    """记录 HTTP 请求。"""

    print(f"[HTTP] {request.method} {request.path_qs}")
    return await handler(request)


def create_app() -> web.Application:
    """创建 SSE 心跳服务。"""

    app = web.Application(middlewares=[logging_middleware])
    app.router.add_get("/health", health_handler)
    app.router.add_get("/stream", stream_handler)
    return app


def main() -> None:
    """启动服务。"""

    print("Day17 SSE 心跳与背压服务正在启动。")
    print(f"服务地址：http://{HOST}:{PORT}")
    print("心跳间隔：", HEARTBEAT_INTERVAL_SECONDS, "秒")
    print("Queue 最大长度：", QUEUE_MAX_SIZE)
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
