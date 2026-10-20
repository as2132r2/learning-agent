"""Day17 第十四步（最后一个新功能）：SSE 自动重连客户端。

先启动服务端：
    python day17/python/day17_sse_resume_server.py

再运行客户端：
    python day17/python/day17_sse_resume_client.py

客户端会保存最新事件 ID。连接关闭后，它会等待服务端通过 retry 字段
建议的时间，再携带 Last-Event-ID 请求头重新连接。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientSession, ClientTimeout


EVENTS_URL = "http://127.0.0.1:8009/events"
DEFAULT_RETRY_MILLISECONDS = 1000
MAX_CONNECTION_ATTEMPTS = 10
EXPECTED_NUMBERS = list(range(1, 9))


@dataclass
class SSEEvent:
    """包含重连信息的一条完整 SSE 事件。"""

    event: str
    data: dict[str, Any]
    event_id: str | None
    retry_milliseconds: int | None


async def iter_sse_events(response: Any) -> AsyncIterator[SSEEvent]:
    """解析 id、event、retry 和 JSON data。"""

    event_name = "message"
    event_id: str | None = None
    retry_milliseconds: int | None = None
    data_lines: list[str] = []

    async for raw_line in response.content:
        line = raw_line.decode("utf-8").rstrip("\r\n")

        if line.startswith(":"):
            continue

        if line == "":
            if data_lines:
                value = json.loads("\n".join(data_lines))
                if not isinstance(value, dict):
                    raise ValueError("SSE data 不是 JSON object。")

                yield SSEEvent(
                    event=event_name,
                    data=value,
                    event_id=event_id,
                    retry_milliseconds=retry_milliseconds,
                )

            event_name = "message"
            event_id = None
            retry_milliseconds = None
            data_lines = []
            continue

        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]

        if field == "event":
            event_name = value
        elif field == "id":
            event_id = value
        elif field == "retry":
            try:
                retry_milliseconds = int(value)
            except ValueError:
                # 无效 retry 不应破坏整个连接，继续使用原重试时间。
                retry_milliseconds = None
        elif field == "data":
            data_lines.append(value)


async def main() -> None:
    """反复连接，直到收到 complete 或达到最大重连次数。"""

    print("Day17 SSE 自动重连客户端")
    print("服务地址：", EVENTS_URL)
    print("期望编号：", EXPECTED_NUMBERS)
    print("=" * 72)

    timeout = ClientTimeout(
        total=None,
        connect=10,
        sock_read=30,
    )

    last_event_id: str | None = None
    retry_milliseconds = DEFAULT_RETRY_MILLISECONDS
    received_numbers: list[int] = []
    completed = False
    connection_attempts = 0

    async with ClientSession(timeout=timeout) as session:
        while (
            not completed
            and connection_attempts < MAX_CONNECTION_ATTEMPTS
        ):
            connection_attempts += 1

            headers = {"Accept": "text/event-stream"}
            if last_event_id is not None:
                headers["Last-Event-ID"] = last_event_id

            print("\n" + "-" * 72)
            print("连接次数：", connection_attempts)
            print("发送 Last-Event-ID：", last_event_id)

            try:
                async with session.get(
                    EVENTS_URL,
                    headers=headers,
                ) as response:
                    print("HTTP 状态码：", response.status)
                    response.raise_for_status()

                    async for event in iter_sse_events(response):
                        if event.retry_milliseconds is not None:
                            retry_milliseconds = (
                                event.retry_milliseconds
                            )

                        if event.event_id is not None:
                            last_event_id = event.event_id

                        if event.event == "number":
                            number = int(event.data["number"])

                            # 正常续传不会重复；仍然保留去重保护。
                            if number not in received_numbers:
                                received_numbers.append(number)
                                print(
                                    f"收到事件 id={event.event_id}："
                                    f"{event.data['message']}"
                                )
                            else:
                                print(
                                    f"忽略重复事件 id={event.event_id}"
                                )

                        elif event.event == "complete":
                            completed = True
                            print(
                                f"收到 complete id={event.event_id}："
                                f"{event.data['message']}"
                            )
                            break

            except Exception as error:
                print(
                    "本次连接发生异常：",
                    type(error).__name__,
                    error,
                )

            if not completed:
                wait_seconds = retry_milliseconds / 1000
                print("连接已结束。")
                print("当前 last_event_id：", last_event_id)
                print("等待重连：", wait_seconds, "秒")
                await asyncio.sleep(wait_seconds)

    print("\n" + "=" * 72)
    print("自动重连结果")
    print("是否收到 complete：", completed)
    print("连接总次数：", connection_attempts)
    print("最终 Last-Event-ID：", last_event_id)
    print("收到的编号：", received_numbers)
    print("期望的编号：", EXPECTED_NUMBERS)
    print("是否无遗漏且顺序正确：", received_numbers == EXPECTED_NUMBERS)

    if completed and received_numbers == EXPECTED_NUMBERS:
        print("结论：断线重连和 Last-Event-ID 续传验证成功。")
    else:
        print("结论：续传不完整，请检查客户端和服务端日志。")

    print("\n本步数据流：")
    print("第一次 GET /events，无 Last-Event-ID")
    print("→ 收到 id=1、2、3")
    print("→ 连接关闭，保存 last_event_id=3")
    print("→ 等待 retry 毫秒")
    print("→ 新请求发送 Last-Event-ID: 3")
    print("→ 服务端从 id=4 继续")
    print("→ 重复上述过程直到 complete")


if __name__ == "__main__":
    asyncio.run(main())
