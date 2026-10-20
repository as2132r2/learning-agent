"""Day17 第五步：多个客户端并发消费 SSE 模型流。

先启动服务端：
    python day17/python/day17_sse_server.py

再运行本程序：
    python day17/python/day17_sse_concurrent_clients.py

本步骤重点：
    1. 每个 HTTP 请求拥有独立的 SSE 长连接；
    2. asyncio.create_task() 同时启动多个客户端任务；
    3. 一条模型流等待数据时，其他模型流可以继续处理；
    4. asyncio.gather() 按传入 Task 的顺序收集最终结果。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout


STREAM_URL = "http://127.0.0.1:8005/stream"

QUESTIONS = [
    "请用一句话解释什么是 SSE。",
    "请用一句话解释什么是异步迭代器。",
    "请用一句话解释什么是长连接。",
]


@dataclass
class SSEEvent:
    """从 HTTP 字节流中解析出来的一条 SSE 事件。"""

    event: str = "message"
    data: str = ""
    event_id: str | None = None


@dataclass
class StreamResult:
    """一个并发客户端任务的最终结果。"""

    task_number: int
    question: str
    request_id: str | None
    answer: str
    token_event_count: int
    first_token_seconds: float
    total_seconds: float
    status: str
    error: str | None = None


async def iter_sse_events(response: Any) -> AsyncIterator[SSEEvent]:
    """逐行读取 HTTP 响应，并按 SSE 空行边界产出事件。"""

    event_name = "message"
    event_id: str | None = None
    data_lines: list[str] = []

    async for raw_line in response.content:
        line = raw_line.decode("utf-8").rstrip("\r\n")

        if line == "":
            if data_lines:
                yield SSEEvent(
                    event=event_name,
                    data="\n".join(data_lines),
                    event_id=event_id,
                )

            event_name = "message"
            event_id = None
            data_lines = []
            continue

        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]

        if field == "event":
            event_name = value
        elif field == "id":
            event_id = value
        elif field == "data":
            data_lines.append(value)

    if data_lines:
        yield SSEEvent(
            event=event_name,
            data="\n".join(data_lines),
            event_id=event_id,
        )


def decode_data(event: SSEEvent) -> dict[str, Any]:
    """把 SSE data 字段中的 JSON 文本转换为字典。"""

    value = json.loads(event.data)
    if not isinstance(value, dict):
        raise ValueError("SSE data 不是 JSON object。")
    return value


async def consume_one_stream(
    session: ClientSession,
    task_number: int,
    question: str,
) -> StreamResult:
    """消费一条独立 SSE 连接，并返回该任务的统计结果。"""

    started_at = time.perf_counter()
    first_token_at: float | None = None
    request_id: str | None = None
    text_parts: list[str] = []
    token_event_count = 0

    print(f"[任务 {task_number} 开始] {question}")

    try:
        async with session.get(
            STREAM_URL,
            params={"question": question},
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(
                    f"HTTP {response.status}: {error_text}"
                )

            async for event in iter_sse_events(response):
                data = decode_data(event)

                if event.event == "start":
                    request_id = str(data.get("request_id"))
                    print(
                        f"[任务 {task_number} 已连接] "
                        f"request_id={request_id}"
                    )

                elif event.event == "token":
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                        print(
                            f"[任务 {task_number} 收到首个文本事件] "
                            f"{first_token_at - started_at:.3f} 秒"
                        )

                    token_event_count += 1
                    text_parts.append(str(data.get("text", "")))

                elif event.event == "done":
                    print(
                        f"[任务 {task_number} 收到 done] "
                        f"服务端块数={data.get('chunk_count')}"
                    )

                elif event.event == "error":
                    raise RuntimeError(
                        f"{data.get('error_type')}: {data.get('message')}"
                    )

        finished_at = time.perf_counter()
        first_token_seconds = (
            first_token_at - started_at
            if first_token_at is not None
            else finished_at - started_at
        )

        print(
            f"[任务 {task_number} 完成] "
            f"耗时={finished_at - started_at:.3f} 秒"
        )

        return StreamResult(
            task_number=task_number,
            question=question,
            request_id=request_id,
            answer="".join(text_parts),
            token_event_count=token_event_count,
            first_token_seconds=first_token_seconds,
            total_seconds=finished_at - started_at,
            status="success",
        )

    except Exception as error:
        finished_at = time.perf_counter()
        print(
            f"[任务 {task_number} 失败] "
            f"{type(error).__name__}: {error}"
        )

        return StreamResult(
            task_number=task_number,
            question=question,
            request_id=request_id,
            answer="".join(text_parts),
            token_event_count=token_event_count,
            first_token_seconds=(
                first_token_at - started_at
                if first_token_at is not None
                else 0.0
            ),
            total_seconds=finished_at - started_at,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )


async def main() -> None:
    """创建三个并发 Task，等待并展示全部流式结果。"""

    print("Day17 SSE 多客户端并发练习")
    print("服务地址：", STREAM_URL)
    print("任务数量：", len(QUESTIONS))
    print("=" * 72)

    timeout = ClientTimeout(
        total=None,
        connect=10,
        sock_read=60,
    )

    overall_started_at = time.perf_counter()

    async with ClientSession(timeout=timeout) as session:
        tasks: list[asyncio.Task[StreamResult]] = []

        for index, question in enumerate(QUESTIONS, start=1):
            task = asyncio.create_task(
                consume_one_stream(session, index, question),
                name=f"sse-client-{index}",
            )
            tasks.append(task)

        print("创建的 Task：")
        for task in tasks:
            print(f"- {task.get_name()}，类型={type(task)}")
        print()

        # gather 并发等待全部任务，结果顺序与 tasks 顺序一致。
        results: list[StreamResult] = await asyncio.gather(*tasks)

    overall_finished_at = time.perf_counter()

    print("\n" + "=" * 72)
    print("全部结果：")

    for result in results:
        print("\n" + "-" * 72)
        print("任务：", result.task_number)
        print("状态：", result.status)
        print("request_id：", result.request_id)
        print("问题：", result.question)
        print("回答：", result.answer)
        print("token 事件数：", result.token_event_count)
        print(
            "首个文本事件等待：",
            round(result.first_token_seconds, 3),
            "秒",
        )
        print("单任务总耗时：", round(result.total_seconds, 3), "秒")
        if result.error is not None:
            print("错误：", result.error)

    sum_of_task_times = sum(result.total_seconds for result in results)
    concurrent_total = overall_finished_at - overall_started_at

    print("\n" + "=" * 72)
    print("三个单任务耗时之和：", round(sum_of_task_times, 3), "秒")
    print("并发执行实际总耗时：", round(concurrent_total, 3), "秒")
    print("gather 返回顺序：", [result.task_number for result in results])

    print("\n本步数据流：")
    print("3 个 question")
    print("→ 3 个 asyncio.Task")
    print("→ 共享一个 ClientSession")
    print("→ 3 条独立 GET /stream 长连接")
    print("→ 服务端并发运行 3 个 llm.astream()")
    print("→ 每条连接独立接收 start / token / done")
    print("→ asyncio.gather() 收集 3 个 StreamResult")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ClientError as error:
        print("无法连接 SSE 服务：", error)
        print("请先在另一个终端启动 day17_sse_server.py。")
