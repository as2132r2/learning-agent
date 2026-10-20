"""Day17 第七步：用 4 个客户端验证 SSE Server 的 Semaphore 限流。

先启动限流服务端：
    python day17/python/day17_sse_limited_server.py

再运行本测试客户端：
    python day17/python/day17_sse_limit_test.py

预期现象：
    - 4 个 SSE 连接都能建立；
    - 前 2 个请求很快获得模型许可；
    - 后 2 个请求保持连接并等待；
    - 前面的请求释放许可后，等待请求才开始调用模型。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientSession, ClientTimeout


STREAM_URL = "http://127.0.0.1:8006/stream"

QUESTIONS = [
    "请用两句话解释什么是协程。",
    "请用两句话解释什么是事件循环。",
    "请用两句话解释什么是异步迭代器。",
    "请用两句话解释什么是 Semaphore。",
]


@dataclass
class SSEEvent:
    """解析后的一条 SSE 事件。"""

    event: str
    data: dict[str, Any]
    event_id: str | None


@dataclass
class LimitedStreamResult:
    """一条受限流模型请求的最终结果和时间统计。"""

    task_number: int
    request_id: str | None
    question: str
    answer: str
    queue_position: int | None
    server_queue_wait_seconds: float
    client_start_wait_seconds: float
    total_seconds: float
    reported_active_streams: int | None
    status: str
    error: str | None = None


async def iter_sse_events(response: Any) -> AsyncIterator[SSEEvent]:
    """逐行解析 id、event 和 JSON data，按空行产出事件。"""

    event_name = "message"
    event_id: str | None = None
    data_lines: list[str] = []

    async for raw_line in response.content:
        line = raw_line.decode("utf-8").rstrip("\r\n")

        if line == "":
            if data_lines:
                raw_data = "\n".join(data_lines)
                decoded = json.loads(raw_data)

                if not isinstance(decoded, dict):
                    raise ValueError("SSE data 不是 JSON object。")

                yield SSEEvent(
                    event=event_name,
                    data=decoded,
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


async def run_one_request(
    session: ClientSession,
    task_number: int,
    question: str,
) -> LimitedStreamResult:
    """建立一条 SSE 连接，观察 queued、start、token 和 done。"""

    requested_at = time.perf_counter()
    request_id: str | None = None
    queue_position: int | None = None
    server_queue_wait_seconds = 0.0
    client_start_wait_seconds = 0.0
    reported_active_streams: int | None = None
    text_parts: list[str] = []

    print(f"[任务 {task_number} 提交] {question}")

    try:
        async with session.get(
            STREAM_URL,
            params={"question": question},
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"HTTP {response.status}: {await response.text()}"
                )

            async for event in iter_sse_events(response):
                if event.event == "queued":
                    request_id = str(event.data.get("request_id"))
                    queue_position = int(event.data.get("waiting", 0))
                    print(
                        f"[任务 {task_number} 已排队] "
                        f"request_id={request_id} "
                        f"当时等待数={queue_position}"
                    )

                elif event.event == "start":
                    client_start_wait_seconds = (
                        time.perf_counter() - requested_at
                    )
                    server_queue_wait_seconds = float(
                        event.data.get("queue_wait_seconds", 0.0)
                    )
                    reported_active_streams = int(
                        event.data.get("active_model_streams", 0)
                    )
                    print(
                        f"[任务 {task_number} 获得许可] "
                        f"排队={server_queue_wait_seconds:.3f} 秒，"
                        f"当前活动模型流={reported_active_streams}"
                    )

                elif event.event == "token":
                    text_parts.append(str(event.data.get("text", "")))

                elif event.event == "done":
                    print(
                        f"[任务 {task_number} 完成] "
                        f"模型耗时={event.data.get('model_seconds')} 秒"
                    )

                elif event.event == "error":
                    raise RuntimeError(
                        f"{event.data.get('error_type')}: "
                        f"{event.data.get('message')}"
                    )

        finished_at = time.perf_counter()

        return LimitedStreamResult(
            task_number=task_number,
            request_id=request_id,
            question=question,
            answer="".join(text_parts),
            queue_position=queue_position,
            server_queue_wait_seconds=server_queue_wait_seconds,
            client_start_wait_seconds=client_start_wait_seconds,
            total_seconds=finished_at - requested_at,
            reported_active_streams=reported_active_streams,
            status="success",
        )

    except Exception as error:
        return LimitedStreamResult(
            task_number=task_number,
            request_id=request_id,
            question=question,
            answer="".join(text_parts),
            queue_position=queue_position,
            server_queue_wait_seconds=server_queue_wait_seconds,
            client_start_wait_seconds=client_start_wait_seconds,
            total_seconds=time.perf_counter() - requested_at,
            reported_active_streams=reported_active_streams,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )


async def main() -> None:
    """同时提交四条请求并比较各自等待时间。"""

    print("Day17 SSE Semaphore 限流测试")
    print("服务地址：", STREAM_URL)
    print("客户端请求数：", len(QUESTIONS))
    print("服务端最大模型并发数：2")
    print("=" * 72)

    timeout = ClientTimeout(
        total=None,
        connect=10,
        sock_read=60,
    )
    overall_started_at = time.perf_counter()

    async with ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                run_one_request(session, index, question),
                name=f"limited-sse-{index}",
            )
            for index, question in enumerate(QUESTIONS, start=1)
        ]

        results: list[LimitedStreamResult] = await asyncio.gather(
            *tasks
        )

    overall_seconds = time.perf_counter() - overall_started_at

    print("\n" + "=" * 72)
    print("最终结果：")

    for result in results:
        print("\n" + "-" * 72)
        print("任务：", result.task_number)
        print("状态：", result.status)
        print("request_id：", result.request_id)
        print("排队事件中的等待数：", result.queue_position)
        print(
            "服务端记录的排队时间：",
            round(result.server_queue_wait_seconds, 3),
            "秒",
        )
        print(
            "客户端从请求到 start：",
            round(result.client_start_wait_seconds, 3),
            "秒",
        )
        print("start 时活动模型流：", result.reported_active_streams)
        print("请求总耗时：", round(result.total_seconds, 3), "秒")
        print("回答：", result.answer)
        if result.error is not None:
            print("错误：", result.error)

    successful_results = [
        result for result in results if result.status == "success"
    ]
    maximum_reported_active = max(
        (
            result.reported_active_streams or 0
            for result in successful_results
        ),
        default=0,
    )

    print("\n" + "=" * 72)
    print("并发验证：")
    print("服务端报告的最大活动模型流：", maximum_reported_active)
    print("配置允许的最大活动模型流：2")
    print("四条请求的整体耗时：", round(overall_seconds, 3), "秒")

    print("\n本步数据流：")
    print("4 个客户端 Task 同时建立 SSE 连接")
    print("→ 4 个 queued 事件")
    print("→ 前 2 个请求 acquire() 成功")
    print("→ 后 2 个请求挂起等待 Semaphore")
    print("→ 前一批完成并 release()")
    print("→ 后一批获得许可并发送 start")
    print("→ 活动模型流始终不超过 2")


if __name__ == "__main__":
    asyncio.run(main())
