"""Day17 第一步：使用 llm.stream() 接收模型的流式输出。

运行方式：
    python day17/python/day17_stream_basic.py

本步骤重点：
    1. invoke() 等完整回答生成后一次性返回；
    2. stream() 返回一个可迭代对象；
    3. 每次迭代得到一个 AIMessageChunk；
    4. 把所有 chunk.content 拼接起来，得到完整回答。
"""

from __future__ import annotations

import os
import time
from typing import Iterator

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# 从项目根目录的 .env 加载模型配置。
load_dotenv()


# ChatOpenAI 是模型客户端对象。
# 虽然类名是 ChatOpenAI，但通过 base_url 可以连接兼容 OpenAI 接口的 DeepSeek。
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


# list 中每个元素都是一条结构化消息，而不是普通字符串。
messages = [
    SystemMessage(
        content="你是一名简洁、友好的编程学习助手。"
    ),
    HumanMessage(
        content="请用三句话解释什么是流式输出。"
    ),
]


def main() -> None:
    """调用流式模型，逐块打印并重新拼接完整回答。"""

    print("Day17 流式输出基础练习")
    print("模型：", llm.model_name)
    print("问题：", messages[-1].content)
    print("=" * 72)
    print("\n助手：", end="", flush=True)

    start_time = time.perf_counter()
    first_chunk_time: float | None = None

    # llm.stream(messages) 不直接返回最终 AIMessage。
    # 它返回 Iterator[AIMessageChunk]，需要用 for 逐块读取。
    stream: Iterator[AIMessageChunk] = llm.stream(messages)

    chunks: list[AIMessageChunk] = []
    text_parts: list[str] = []

    for chunk_index, chunk in enumerate(stream, start=1):
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter()

        chunks.append(chunk)

        # 大多数文本响应的 content 是 str。
        # 某些多模态模型也可能返回内容块列表，所以这里先判断类型。
        if isinstance(chunk.content, str):
            text = chunk.content
        else:
            text = str(chunk.content)

        text_parts.append(text)

        # end="" 防止每个小块都换行；flush=True 立即刷新到终端。
        print(text, end="", flush=True)

        # 前几个块额外显示调试信息，避免终端输出过长。
        if chunk_index <= 3:
            print(
                f"\n[调试：第 {chunk_index} 块，"
                f"类型={type(chunk).__name__}，"
                f"content={chunk.content!r}]",
                end="\n助手继续：",
                flush=True,
            )

    finished_time = time.perf_counter()

    # join() 把每一个文本片段按照原顺序连接成完整回答。
    complete_answer = "".join(text_parts)

    print("\n\n" + "=" * 72)
    print("流式调用结束")
    print("chunk 数量：", len(chunks))
    print("完整回答的数据类型：", type(complete_answer))
    print("完整回答：")
    print(complete_answer)

    if first_chunk_time is not None:
        print(
            "首个 chunk 等待时间：",
            round(first_chunk_time - start_time, 3),
            "秒",
        )

    print(
        "总耗时：",
        round(finished_time - start_time, 3),
        "秒",
    )

    # 最后一个 chunk 通常携带 finish_reason 和 token 用量等元数据。
    if chunks:
        last_chunk = chunks[-1]
        print("\n最后一个 chunk：")
        print("content：", repr(last_chunk.content))
        print("response_metadata：", last_chunk.response_metadata)
        print("usage_metadata：", last_chunk.usage_metadata)

    print("\n" + "=" * 72)
    print("本步数据流：")
    print("messages: list[BaseMessage]")
    print("→ llm.stream(messages)")
    print("→ Iterator[AIMessageChunk]")
    print("→ for 循环逐块读取并立即打印")
    print("→ text_parts: list[str]")
    print("→ ''.join(text_parts)")
    print("→ complete_answer: str")


if __name__ == "__main__":
    main()
