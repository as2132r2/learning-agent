"""Day17 第十步：按时间取消 SSE，而不是按 token 数量取消。

先启动主动断线检测服务端：
    python day17/python/day17_sse_disconnect_server.py

再运行本客户端：
    python day17/python/day17_sse_timeout_cancel.py

程序会在 CANCEL_AFTER_SECONDS 秒后取消 SSE 消费协程。即使模型还没有
返回第一个 token，超时也能触发。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientSession, ClientTimeout


SERVER_URL = "http://127.0.0.1:8007"
STREAM_URL = f"{SERVER_URL}/stream"
HEALTH_URL = f"{SERVER_URL}/health"

CANCEL_AFTER_SECONDS = 1.0

QUESTION = (
    "请用 Python 术语详细分十点讲解异步任务调度，"
    "每一点都给出原理、示例和常见错误。"
)


@dataclass
class SSEEvent:
    """一条已经完成解析的 SSE 事件。"""

    event: str
    data: dict[str, Any]
    event_id: str | None


@dataclass
class PartialStreamState:
    """保存取消发生前已经收到的信息。"""

    request_id: str | None = None
    token_event_count: int = 0
    partial_answer: str = ""
    received_start: bool = False
    received_done: bool = False


async def iter_sse_events(response: Any) -> AsyncIterator[SSEEvent]:
    """将 HTTP 字节流解析成 SSEEvent。"""

    event_name = "message"
    event_id: str | None = None
    data_lines: list[str] = []

    async for raw_line in response.content:
        line = raw_line.decode("utf-8").rstrip("\r\n")

        if line == "":
            if data_lines:
                value = json.loads("\n".join(data_lines))
                if not isinstance(value, dict):
                    raise ValueError("SSE data 不是 JSON object。")

                yield SSEEvent(
                    event=event_name,
                    data=value,
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


async def read_health(session: ClientSession) -> dict[str, Any]:
    """读取服务端当前并发和取消统计。"""

    async with session.get(HEALTH_URL) as response:
        response.raise_for_status()
        value = await response.json()

    if not isinstance(value, dict):
        raise ValueError("健康检查没有返回 JSON object。")

    return value


async def consume_until_cancelled(
    session: ClientSession,
    state: PartialStreamState,
) -> None:
    """持续消费 SSE；外层 timeout 会在时间到达时取消本协程。"""

    text_parts: list[str] = []

    try:
        async with session.get(
            STREAM_URL,
            params={"question": QUESTION},
            headers={"Accept": "text/event-stream"},
        ) as response:
            print("HTTP 状态码：", response.status)
            response.raise_for_status()

            async for event in iter_sse_events(response):
                if event.event == "queued":
                    state.request_id = str(
                        event.data.get("request_id")
                    )
                    print("请求进入队列：", state.request_id)

                elif event.event == "start":
                    state.received_start = True
                    print(
                        "服务端已经开始模型流，active=",
                        event.data.get("active_model_streams"),
                    )

                elif event.event == "token":
                    state.token_event_count += 1
                    text = str(event.data.get("text", ""))
                    text_parts.append(text)
                    state.partial_answer = "".join(text_parts)
                    print(text, end="", flush=True)

                elif event.event == "done":
                    state.received_done = True
                    print("\n模型在超时前已经正常完成。")
                    return

                elif event.event == "error":
                    raise RuntimeError(
                        f"{event.data.get('error_type')}: "
                        f"{event.data.get('message')}"
                    )

    except asyncio.CancelledError:
        # 不吞掉取消异常。先打印，再重新抛出，确保 async with 正常退出。
        print("\n[消费协程收到 CancelledError，准备关闭 HTTP 响应]")
        raise


async def wait_until_server_releases(
    session: ClientSession,
    cancelled_before: int,
) -> dict[str, Any]:
    """轮询健康接口，等待活动数量归零且取消计数增加。"""

    latest: dict[str, Any] = {}

    for attempt in range(1, 21):
        await asyncio.sleep(0.1)
        latest = await read_health(session)

        active = int(latest.get("active_model_streams", -1))
        waiting = int(latest.get("waiting_model_streams", -1))
        cancelled = int(latest.get("cancelled_model_streams", -1))

        print(
            f"[健康检查 {attempt}] "
            f"active={active} waiting={waiting} "
            f"cancelled={cancelled}"
        )

        if (
            active == 0
            and waiting == 0
            and cancelled > cancelled_before
        ):
            break

    return latest


async def main() -> None:
    """限时消费 SSE，并验证服务端主动取消模型任务。"""

    print("Day17 SSE 定时取消练习")
    print("服务地址：", STREAM_URL)
    print("自动取消时间：", CANCEL_AFTER_SECONDS, "秒")
    print("问题：", QUESTION)
    print("=" * 72)

    timeout = ClientTimeout(
        total=None,
        connect=10,
        sock_read=60,
    )

    state = PartialStreamState()
    started_at = time.perf_counter()
    timed_out = False

    async with ClientSession(timeout=timeout) as session:
        health_before = await read_health(session)
        cancelled_before = int(
            health_before.get("cancelled_model_streams", 0)
        )

        print("取消前健康状态：", health_before)
        print("=" * 72)

        try:
            # asyncio.timeout 会在期限到达时取消当前代码块中的等待。
            async with asyncio.timeout(CANCEL_AFTER_SECONDS):
                await consume_until_cancelled(session, state)
        except TimeoutError:
            timed_out = True
            print(
                f"\n[客户端在 {CANCEL_AFTER_SECONDS} 秒后主动超时]"
            )
        except RuntimeError as error:
            print("\n服务端或上游模型发生错误：", error)

        elapsed = time.perf_counter() - started_at

        print("\n" + "=" * 72)
        print("客户端取消结果")
        print("request_id：", state.request_id)
        print("是否触发定时取消：", timed_out)
        print("是否已经收到 start：", state.received_start)
        print("是否已经收到 done：", state.received_done)
        print("取消前收到的 token 事件数：", state.token_event_count)
        print("取消前得到的部分回答：")
        print(state.partial_answer)
        print("客户端阶段耗时：", round(elapsed, 3), "秒")

        if timed_out:
            print("\n等待服务端主动发现断线并释放许可：")
            health_after = await wait_until_server_releases(
                session,
                cancelled_before,
            )
        else:
            health_after = await read_health(session)

        print("\n最终健康状态：", health_after)

        active_after = int(
            health_after.get("active_model_streams", -1)
        )
        cancelled_after = int(
            health_after.get("cancelled_model_streams", -1)
        )

        if (
            timed_out
            and active_after == 0
            and cancelled_after > cancelled_before
        ):
            print("验证成功：断线监测取消了模型 Task，并释放了许可。")
        elif state.received_done:
            print("模型在期限内完成，本次没有触发取消。")
        else:
            print("没有完整验证主动取消，请结合服务端日志判断。")

    print("\n" + "=" * 72)
    print("本步数据流：")
    print("asyncio.timeout(1.0)")
    print("→ consume_until_cancelled() 开始读取 SSE")
    print("→ 期限到达，消费协程收到 CancelledError")
    print("→ ClientResponse 上下文退出，HTTP 连接关闭")
    print("→ 服务端 disconnect_task 发现 transport.is_closing()")
    print("→ model_task.cancel()")
    print("→ finally 释放 Semaphore")
    print("→ /health 的 cancelled 增加且 active 归零")


if __name__ == "__main__":
    asyncio.run(main())
