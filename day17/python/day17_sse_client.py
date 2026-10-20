"""Day17 第四步（客户端）：连接 SSE 服务并逐事件接收模型流。

先在终端 1 启动服务端：
    python day17/python/day17_sse_server.py

再在终端 2 启动客户端：
    python day17/python/day17_sse_client.py

也可以直接把问题作为命令行参数：
    python day17/python/day17_sse_client.py "什么是 SSE？"
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout


SERVER_URL = "http://127.0.0.1:8005"
STREAM_URL = f"{SERVER_URL}/stream"


@dataclass
class SSEEvent:
    """客户端解析完成的一条 Server-Sent Event。"""

    event: str = "message"
    data: str = ""
    event_id: str | None = None


async def iter_sse_events(
    response: Any,
) -> AsyncIterator[SSEEvent]:
    """从 HTTP 响应体逐行解析 SSE 事件。

    参数：
        response: aiohttp.ClientResponse。

    产出：
        SSEEvent。每遇到一个空行，就产出前面收集的一条完整事件。
    """

    current_event = "message"
    current_id: str | None = None
    data_lines: list[str] = []

    # response.content 是异步字节流，每次迭代通常得到一行 bytes。
    async for raw_line in response.content:
        line = raw_line.decode("utf-8").rstrip("\r\n")

        # 空行表示当前 SSE 事件结束。
        if line == "":
            if data_lines:
                yield SSEEvent(
                    event=current_event,
                    data="\n".join(data_lines),
                    event_id=current_id,
                )

            current_event = "message"
            current_id = None
            data_lines = []
            continue

        # 以冒号开头的是 SSE 注释或心跳，客户端可以忽略。
        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")

        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]

        if field == "event":
            current_event = value
        elif field == "data":
            # SSE 允许一个事件包含多行 data。
            data_lines.append(value)
        elif field == "id":
            current_id = value

    # 如果服务端关闭连接前没有发送最后一个空行，仍尝试产出剩余事件。
    if data_lines:
        yield SSEEvent(
            event=current_event,
            data="\n".join(data_lines),
            event_id=current_id,
        )


def parse_json_data(event: SSEEvent) -> dict[str, Any]:
    """把 SSE 的 data 字符串转换为 JSON 对象。"""

    value = json.loads(event.data)

    if not isinstance(value, dict):
        raise ValueError("SSE data 必须是 JSON object。")

    return value


async def consume_stream(question: str) -> None:
    """发送问题，逐个处理 start、token、done 和 error 事件。"""

    # total=None 表示整个长连接没有固定总超时。
    # connect 和 sock_read 仍分别限制连接、单次读取等待时间。
    timeout = ClientTimeout(
        total=None,
        connect=10,
        sock_read=60,
    )

    start_time = time.perf_counter()
    first_token_time: float | None = None
    text_parts: list[str] = []
    received_event_count = 0

    async with ClientSession(timeout=timeout) as session:
        # params 会自动对中文问题进行 URL 编码。
        async with session.get(
            STREAM_URL,
            params={"question": question},
            headers={"Accept": "text/event-stream"},
        ) as response:
            print("HTTP 状态码：", response.status)
            print("Content-Type：", response.headers.get("Content-Type"))

            if response.status != 200:
                error_text = await response.text()
                print("服务端拒绝请求：", error_text)
                return

            print("\n助手：", end="", flush=True)

            async for event in iter_sse_events(response):
                received_event_count += 1

                try:
                    data = parse_json_data(event)
                except (json.JSONDecodeError, ValueError) as error:
                    print(f"\n[SSE data 解析失败：{error}]")
                    continue

                if event.event == "start":
                    print(
                        f"\n[流开始 request_id={data.get('request_id')} "
                        f"model={data.get('model')}]"
                    )
                    print("助手继续：", end="", flush=True)

                elif event.event == "token":
                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    text = str(data.get("text", ""))
                    text_parts.append(text)
                    print(text, end="", flush=True)

                elif event.event == "done":
                    print("\n\n[服务端发送 done 事件]")
                    print("服务端 chunk_count：", data.get("chunk_count"))
                    print("服务端 answer_length：", data.get("answer_length"))

                elif event.event == "error":
                    print("\n\n[服务端发送 error 事件]")
                    print("错误类型：", data.get("error_type"))
                    print("错误信息：", data.get("message"))

                else:
                    print(
                        f"\n[收到未知事件 {event.event!r}：{data}]"
                    )

    finished_time = time.perf_counter()
    complete_answer = "".join(text_parts)

    print("\n" + "=" * 72)
    print("SSE 连接已经结束")
    print("收到的事件总数：", received_event_count)
    print("客户端拼接后的回答长度：", len(complete_answer))
    print("客户端拼接后的完整回答：")
    print(complete_answer)

    if first_token_time is not None:
        print(
            "首个 token 事件等待时间：",
            round(first_token_time - start_time, 3),
            "秒",
        )

    print("总耗时：", round(finished_time - start_time, 3), "秒")


async def main() -> None:
    """取得问题并启动 SSE 消费流程。"""

    if len(sys.argv) >= 2:
        question = " ".join(sys.argv[1:]).strip()
    else:
        question = input("请输入问题：").strip()

    if not question:
        print("问题不能为空。")
        return

    print("Day17 SSE Client 已启动。")
    print("服务地址：", STREAM_URL)
    print("问题：", question)
    print("=" * 72)

    try:
        await consume_stream(question)
    except ClientError as error:
        print("\n无法连接 SSE 服务。")
        print("错误类型：", type(error).__name__)
        print("错误信息：", error)
        print("请先在另一个终端启动 day17_sse_server.py。")
        return
    except asyncio.TimeoutError:
        print("\n等待 SSE 数据超时。")
        return

    print("\n" + "=" * 72)
    print("客户端数据流：")
    print("question: str")
    print("→ GET /stream?question=...")
    print("→ HTTP 200 + text/event-stream")
    print("→ response.content 异步字节流")
    print("→ 按空行切分 SSE 事件")
    print("→ event + id + JSON data")
    print("→ token 事件立即打印")
    print("→ text_parts 拼接完整回答")


if __name__ == "__main__":
    asyncio.run(main())
