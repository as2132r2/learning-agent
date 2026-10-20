"""Day17 第十二步（客户端）：观察 SSE 心跳并接收模型事件。

先启动服务端：
    python day17/python/day17_sse_heartbeat_server.py

再运行客户端：
    python day17/python/day17_sse_heartbeat_client.py

也可以传入问题：
    python day17/python/day17_sse_heartbeat_client.py "请解释背压"
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from aiohttp import ClientSession, ClientTimeout


STREAM_URL = "http://127.0.0.1:8008/stream"

DEFAULT_QUESTION = (
    "请用 Python 术语分五点解释异步生产者、Queue、消费者和背压的关系。"
)


@dataclass
class SSEFrame:
    """客户端解析的一条 SSE 业务事件或注释心跳。"""

    kind: Literal["event", "comment"]
    event: str | None = None
    data: dict[str, Any] | None = None
    event_id: str | None = None
    comment: str | None = None


async def iter_sse_frames(response: Any) -> AsyncIterator[SSEFrame]:
    """逐行解析 SSE，同时保留通常会被忽略的注释行。"""

    event_name = "message"
    event_id: str | None = None
    data_lines: list[str] = []

    async for raw_line in response.content:
        line = raw_line.decode("utf-8").rstrip("\r\n")

        # 冒号开头是 SSE 注释。服务端用它发送心跳。
        if line.startswith(":"):
            yield SSEFrame(
                kind="comment",
                comment=line[1:].strip(),
            )
            continue

        # 空行表示当前业务事件结束。
        if line == "":
            if data_lines:
                value = json.loads("\n".join(data_lines))
                if not isinstance(value, dict):
                    raise ValueError("SSE data 不是 JSON object。")

                yield SSEFrame(
                    kind="event",
                    event=event_name,
                    data=value,
                    event_id=event_id,
                )

            event_name = "message"
            event_id = None
            data_lines = []
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


async def main() -> None:
    """消费模型事件和心跳，分别统计并展示。"""

    if len(sys.argv) >= 2:
        question = " ".join(sys.argv[1:]).strip()
    else:
        question = DEFAULT_QUESTION

    print("Day17 SSE 心跳观察客户端")
    print("服务地址：", STREAM_URL)
    print("问题：", question)
    print("=" * 72)

    timeout = ClientTimeout(
        total=None,
        connect=10,
        sock_read=60,
    )

    started_at = time.perf_counter()
    first_token_at: float | None = None
    request_id: str | None = None
    heartbeat_count = 0
    token_event_count = 0
    business_event_count = 0
    text_parts: list[str] = []

    async with ClientSession(timeout=timeout) as session:
        async with session.get(
            STREAM_URL,
            params={"question": question},
            headers={"Accept": "text/event-stream"},
        ) as response:
            print("HTTP 状态码：", response.status)
            print("Content-Type：", response.headers.get("Content-Type"))
            response.raise_for_status()
            print()

            async for frame in iter_sse_frames(response):
                if frame.kind == "comment":
                    heartbeat_count += 1
                    print(
                        f"[收到心跳 {heartbeat_count}] "
                        f"{frame.comment}"
                    )
                    continue

                business_event_count += 1
                event_name = frame.event or "message"
                data = frame.data or {}

                if event_name == "queued":
                    request_id = str(data.get("request_id"))
                    print(f"[queued] request_id={request_id}")

                elif event_name == "start":
                    print(
                        "[start] heartbeat_interval=",
                        data.get("heartbeat_interval_seconds"),
                        " queue_max_size=",
                        data.get("queue_max_size"),
                        sep="",
                    )
                    print("\n助手：", end="", flush=True)

                elif event_name == "token":
                    if first_token_at is None:
                        first_token_at = time.perf_counter()

                    token_event_count += 1
                    text = str(data.get("text", ""))
                    text_parts.append(text)
                    print(text, end="", flush=True)

                elif event_name == "done":
                    print("\n\n[done]")
                    print("服务端 token 数：", data.get("chunk_count"))
                    print("服务端心跳数：", data.get("heartbeat_count"))
                    print("模型耗时：", data.get("model_seconds"), "秒")

                elif event_name == "error":
                    print("\n\n[error]")
                    print("错误类型：", data.get("error_type"))
                    print("错误信息：", data.get("message"))

                else:
                    print(f"\n[未知事件 {event_name!r}] {data}")

    finished_at = time.perf_counter()
    complete_answer = "".join(text_parts)

    print("\n" + "=" * 72)
    print("心跳观察结果")
    print("request_id：", request_id)
    print("业务事件数量：", business_event_count)
    print("token 事件数量：", token_event_count)
    print("注释心跳数量：", heartbeat_count)
    print("完整回答长度：", len(complete_answer))
    print("完整回答：")
    print(complete_answer)

    if first_token_at is not None:
        print(
            "首个 token 等待时间：",
            round(first_token_at - started_at, 3),
            "秒",
        )
    print("总耗时：", round(finished_at - started_at, 3), "秒")

    print("\n验证：")
    print("- heartbeat 只增加 heartbeat_count。")
    print("- token.text 才会加入 text_parts。")
    print("- 因此心跳不会出现在 complete_answer 中。")

    print("\n本步数据流：")
    print("HTTP 字节流")
    print("→ 冒号行 → SSEFrame(kind='comment') → 心跳统计")
    print("→ event/data 行 → SSEFrame(kind='event')")
    print("→ token.text → text_parts → 完整回答")


if __name__ == "__main__":
    asyncio.run(main())
