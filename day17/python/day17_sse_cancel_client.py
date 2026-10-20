"""Day17 第八步：客户端中途关闭 SSE，并验证服务端释放许可。

先启动限流服务端：
    python day17/python/day17_sse_limited_server.py

再运行取消客户端：
    python day17/python/day17_sse_cancel_client.py

本程序收到 CANCEL_AFTER_TOKEN_EVENTS 个 token 事件后，不再继续读取，
并主动关闭 HTTP 响应连接。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientSession, ClientTimeout


SERVER_URL = "http://127.0.0.1:8006"
STREAM_URL = f"{SERVER_URL}/stream"
HEALTH_URL = f"{SERVER_URL}/health"

CANCEL_AFTER_TOKEN_EVENTS = 12

QUESTION = (
    "请详细分十点讲解 Python 异步编程，每一点都给出解释和一个小例子。"
)


@dataclass
class SSEEvent:
    """解析完成的一条 SSE 事件。"""

    event: str
    data: dict[str, Any]
    event_id: str | None


async def iter_sse_events(response: Any) -> AsyncIterator[SSEEvent]:
    """从 aiohttp 响应字节流中逐行解析 SSE 事件。"""

    event_name = "message"
    event_id: str | None = None
    data_lines: list[str] = []

    async for raw_line in response.content:
        line = raw_line.decode("utf-8").rstrip("\r\n")

        if line == "":
            if data_lines:
                decoded = json.loads("\n".join(data_lines))
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


async def read_health(session: ClientSession) -> dict[str, Any]:
    """读取服务端并发统计。"""

    async with session.get(HEALTH_URL) as response:
        response.raise_for_status()
        value = await response.json()

    if not isinstance(value, dict):
        raise ValueError("健康检查没有返回 JSON object。")

    return value


async def cancel_stream_after_limit(session: ClientSession) -> None:
    """读取少量 token 后主动关闭 SSE 响应。"""

    requested_at = time.perf_counter()
    request_id: str | None = None
    token_event_count = 0
    partial_parts: list[str] = []
    received_done = False

    print("准备发送一个预期较长的模型请求。")
    print("问题：", QUESTION)
    print("取消阈值：", CANCEL_AFTER_TOKEN_EVENTS, "个 token 事件")

    # 不能只 break 后保留 response；退出 async with 会关闭响应连接。
    async with session.get(
        STREAM_URL,
        params={"question": QUESTION},
        headers={"Accept": "text/event-stream"},
    ) as response:
        print("HTTP 状态码：", response.status)
        response.raise_for_status()

        async for event in iter_sse_events(response):
            if event.event == "queued":
                request_id = str(event.data.get("request_id"))
                print("请求已排队：", request_id)

            elif event.event == "start":
                print(
                    "已经获得模型许可，服务端活动模型流：",
                    event.data.get("active_model_streams"),
                )

            elif event.event == "token":
                token_event_count += 1
                text = str(event.data.get("text", ""))
                partial_parts.append(text)
                print(text, end="", flush=True)

                if token_event_count >= CANCEL_AFTER_TOKEN_EVENTS:
                    print("\n\n[达到阈值，客户端停止读取 SSE]")
                    break

            elif event.event == "done":
                received_done = True
                print("\n服务端已经正常完成，来不及取消。")
                break

            elif event.event == "error":
                print("\n服务端错误：", event.data)
                break

    # 运行到这里时，ClientResponse 上下文已经退出，底层连接被关闭或回收。
    cancelled_at = time.perf_counter()

    print("\n" + "=" * 72)
    print("客户端读取阶段结束")
    print("request_id：", request_id)
    print("收到 token 事件数：", token_event_count)
    print("是否收到 done：", received_done)
    print("目前得到的部分回答：")
    print("".join(partial_parts))
    print(
        "取消前耗时：",
        round(cancelled_at - requested_at, 3),
        "秒",
    )


async def main() -> None:
    """执行取消测试，然后轮询健康接口确认许可被释放。"""

    print("Day17 SSE 客户端取消练习")
    print("服务地址：", STREAM_URL)
    print("=" * 72)

    timeout = ClientTimeout(
        total=None,
        connect=10,
        sock_read=60,
    )

    async with ClientSession(timeout=timeout) as session:
        before = await read_health(session)

        print("取消前服务端状态：")
        print(before)
        print("=" * 72)

        await cancel_stream_after_limit(session)

        # 服务端通常要等到下一次 response.write() 才发现客户端断开。
        # 因此短暂轮询，而不是假设关闭连接后计数会瞬间变为 0。
        final_health: dict[str, Any] | None = None

        for attempt in range(1, 11):
            await asyncio.sleep(0.2)
            current = await read_health(session)
            final_health = current

            active = int(current.get("active_model_streams", -1))
            waiting = int(current.get("waiting_model_streams", -1))

            print(
                f"[状态检查 {attempt}] "
                f"active={active} waiting={waiting}"
            )

            if active == 0 and waiting == 0:
                break

        print("\n" + "=" * 72)
        print("取消后的最终服务端状态：")
        print(final_health)

        if final_health is not None:
            active = int(final_health.get("active_model_streams", -1))
            waiting = int(final_health.get("waiting_model_streams", -1))

            if active == 0 and waiting == 0:
                print("结论：服务端已结束对应模型流并释放 Semaphore。")
            else:
                print("结论：服务端尚未确认释放，请查看服务端日志。")

    print("\n本步数据流：")
    print("客户端建立 SSE 连接")
    print("→ 服务端获得 Semaphore 并调用 llm.astream()")
    print("→ 客户端接收少量 token")
    print("→ 客户端退出 ClientResponse 上下文")
    print("→ HTTP 连接关闭")
    print("→ 服务端下一次 write() 发现 ConnectionResetError")
    print("→ finally: active -= 1 + semaphore.release()")
    print("→ /health 显示 active=0、waiting=0")


if __name__ == "__main__":
    asyncio.run(main())
