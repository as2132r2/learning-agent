"""Day14 第二步：使用ChatOpenAI.ainvoke()异步调用一次模型。"""

import asyncio
import os
import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()


def require_environment(name: str) -> str:
    """读取必需环境变量。"""

    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"缺少环境变量：{name}")
    return value.strip()


llm = ChatOpenAI(
    model=require_environment("DEEPSEEK_MODEL"),
    api_key=require_environment("DEEPSEEK_API_KEY"),
    base_url=require_environment("DEEPSEEK_BASE_URL"),
    temperature=0,
    timeout=30,
    max_retries=1
)


async def ask_model(question: str):
    """异步调用模型并返回AIMessage。"""

    messages = [
        SystemMessage(
            content="你是一名简洁、友好的编程学习助手。"
        ),
        HumanMessage(content=question)
    ]

    print("传给模型的消息：")
    for index, message in enumerate(messages, start=1):
        print(
            f"第{index}条：{type(message).__name__}，"
            f"content={message.content!r}"
        )

    # 调用ainvoke()会先得到协程；await才真正等待调用完成。
    response_coroutine = llm.ainvoke(messages)

    print("\nainvoke()返回值类型：")
    print(type(response_coroutine))
    print("是否为协程：", asyncio.iscoroutine(response_coroutine))

    start_time = time.perf_counter()
    response = await response_coroutine
    elapsed = time.perf_counter() - start_time

    print("\n模型调用完成。")
    print(f"等待耗时：{elapsed:.2f}秒")

    return response


async def main() -> None:
    """异步程序入口。"""

    question = input("请输入问题：").strip()

    if not question:
        print("问题不能为空。")
        return

    response = await ask_model(question)

    print("\n" + "=" * 60)
    print("模型响应类型：")
    print(type(response))

    print("\n回答正文：")
    print(response.content)

    print("\nToken用量：")
    print(response.usage_metadata)


if __name__ == "__main__":
    asyncio.run(main())


# ============================================================
# 数据流
# ============================================================
# 普通程序入口
#      ↓ asyncio.run(main())
# 创建事件循环并运行main协程
#      ↓ await ask_model(question)
# 创建messages列表
#      ↓ llm.ainvoke(messages)
# coroutine对象
#      ↓ await
# 异步HTTP模型请求
#      ↓
# AIMessage → content / usage_metadata
