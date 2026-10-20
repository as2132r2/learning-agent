"""Day17 第二步：使用 llm.astream() 异步接收模型流。

运行方式：
    python day17/python/day17_async_stream.py

本步骤重点：
    1. stream() 配合普通 for；
    2. astream() 配合 async for；
    3. 等待模型数据时，事件循环可以运行其他协程；
    4. asyncio.create_task() 把协程安排为并发 Task。
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


messages = [
    SystemMessage(
        content="你是一名简洁、友好的编程学习助手。"
    ),
    HumanMessage(
        content="请分四点解释 Python 中 async 和 await 的作用。"
    ),
]


async def heartbeat(stop_event: asyncio.Event) -> int:
    """等待模型期间定时输出心跳，收到停止信号后结束。

    参数：
        stop_event: asyncio.Event，供另一个协程发送停止信号。

    返回：
        int，实际打印的心跳次数。
    """

    count = 0

    while not stop_event.is_set():
        # asyncio.sleep() 不阻塞线程，而是把执行权交回事件循环。
        await asyncio.sleep(0.3)

        # sleep 结束时模型可能恰好已经完成，因此再次检查。
        if stop_event.is_set():
            break

        count += 1
        print(f"\n[心跳 {count}：事件循环仍然可以处理其他任务]")
        print("助手继续：", end="", flush=True)

    return count


async def receive_model_stream() -> tuple[str, int, float, float]:
    """异步读取模型流并返回完整文本和时间信息。

    返回 tuple 中依次是：
        complete_answer: str，拼接后的完整回答；
        chunk_count: int，收到的消息块数量；
        first_chunk_seconds: float，首块等待时间；
        total_seconds: float，完整流总耗时。
    """

    start_time = time.perf_counter()
    first_chunk_time: float | None = None
    chunk_count = 0
    text_parts: list[str] = []

    # astream() 返回异步迭代器，不能使用普通 for。
    stream: AsyncIterator[AIMessageChunk] = llm.astream(messages)

    async for chunk in stream:
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter()

        chunk_count += 1

        if isinstance(chunk.content, str):
            text = chunk.content
        else:
            text = str(chunk.content)

        text_parts.append(text)
        print(text, end="", flush=True)

    finished_time = time.perf_counter()
    complete_answer = "".join(text_parts)

    # 正常情况下至少会收到一个 chunk；保留后备值让返回类型始终是 float。
    first_chunk_seconds = (
        first_chunk_time - start_time
        if first_chunk_time is not None
        else finished_time - start_time
    )
    total_seconds = finished_time - start_time

    return (
        complete_answer,
        chunk_count,
        first_chunk_seconds,
        total_seconds,
    )


async def main() -> None:
    """并发运行模型流协程和心跳协程。"""

    print("Day17 异步流式输出练习")
    print("模型：", llm.model_name)
    print("问题：", messages[-1].content)
    print("=" * 72)
    print("\n助手：", end="", flush=True)

    # Event 是协程之间传递简单状态信号的异步同步原语。
    stop_event = asyncio.Event()

    # create_task() 会立即把 heartbeat 协程安排给事件循环。
    heartbeat_task: asyncio.Task[int] = asyncio.create_task(
        heartbeat(stop_event),
        name="heartbeat-task",
    )

    try:
        # 当前主协程等待模型流，但等待网络数据时不会堵塞整个事件循环。
        (
            complete_answer,
            chunk_count,
            first_chunk_seconds,
            total_seconds,
        ) = await receive_model_stream()
    finally:
        # 无论模型成功还是抛出异常，都通知心跳协程结束。
        stop_event.set()
        heartbeat_count = await heartbeat_task

    print("\n\n" + "=" * 72)
    print("异步流式调用结束")
    print("完整回答：")
    print(complete_answer)
    print("chunk 数量：", chunk_count)
    print("心跳次数：", heartbeat_count)
    print("首个 chunk 等待时间：", round(first_chunk_seconds, 3), "秒")
    print("总耗时：", round(total_seconds, 3), "秒")
    print("心跳 Task 名称：", heartbeat_task.get_name())
    print("心跳 Task 是否完成：", heartbeat_task.done())

    print("\n" + "=" * 72)
    print("本步数据流：")
    print("asyncio.run(main())")
    print("→ 创建 asyncio.Event")
    print("→ create_task(heartbeat) 创建并发 Task")
    print("→ await receive_model_stream()")
    print("→ llm.astream(messages)")
    print("→ async for 逐块接收 AIMessageChunk")
    print("→ 等待网络时，事件循环切换到 heartbeat")
    print("→ 模型结束后 stop_event.set()")
    print("→ await heartbeat_task 回收后台任务")


if __name__ == "__main__":
    asyncio.run(main())
