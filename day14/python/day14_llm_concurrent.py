"""Day14 第三步：并发执行多个ChatOpenAI.ainvoke()请求。"""

import asyncio
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()


def require_environment(name: str) -> str:
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


QUESTIONS = [
    "请用一句话解释什么是协程。",
    "请用一句话解释什么是事件循环。",
    "请用一句话解释什么是并发。"
]


@dataclass
class ModelResult:
    """保存一次模型调用的输入、输出和耗时。"""

    index: int
    question: str
    answer: str
    elapsed_seconds: float
    total_tokens: int | None


async def ask_one(index: int, question: str) -> ModelResult:
    """执行一个异步模型请求。"""

    print(f"[任务{index}开始] {question}")
    start_time = time.perf_counter()

    response = await llm.ainvoke([
        SystemMessage(
            content="你是一名简洁的Python异步编程老师，只回答一句话。"
        ),
        HumanMessage(content=question)
    ])

    elapsed = time.perf_counter() - start_time
    usage = response.usage_metadata or {}

    print(f"[任务{index}完成] 耗时{elapsed:.2f}秒")

    return ModelResult(
        index=index,
        question=question,
        answer=str(response.content),
        elapsed_seconds=elapsed,
        total_tokens=usage.get("total_tokens")
    )


async def main() -> None:
    """创建多个Task，并等待全部任务完成。"""

    print(f"准备并发执行{len(QUESTIONS)}个模型请求。\n")
    total_start = time.perf_counter()

    tasks = [
        asyncio.create_task(
            ask_one(index, question),
            name=f"model-task-{index}"
        )
        for index, question in enumerate(QUESTIONS, start=1)
    ]

    print("创建的Task对象：")
    for task in tasks:
        print(f"- name={task.get_name()}，type={type(task)}")

    results = await asyncio.gather(*tasks)
    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    print("全部结果：")

    for result in results:
        print(f"\n任务{result.index}")
        print("问题：", result.question)
        print("回答：", result.answer)
        print(f"单任务耗时：{result.elapsed_seconds:.2f}秒")
        print("Token总数：", result.total_tokens)

    individual_time_sum = sum(
        result.elapsed_seconds for result in results
    )

    print("\n" + "=" * 70)
    print(f"三个单任务耗时之和：{individual_time_sum:.2f}秒")
    print(f"并发执行实际总耗时：{total_elapsed:.2f}秒")
    print("结果顺序与QUESTIONS顺序一致，不一定等于完成顺序。")


if __name__ == "__main__":
    asyncio.run(main())


# ============================================================
# 并发流程
# ============================================================
# main()
#   ↓ create_task() × 3
# Task1 ─→ ainvoke() ─→ 等待模型服务 ─→ ModelResult1
# Task2 ─→ ainvoke() ─→ 等待模型服务 ─→ ModelResult2
# Task3 ─→ ainvoke() ─→ 等待模型服务 ─→ ModelResult3
#   ↓ asyncio.gather(*tasks)
# list[ModelResult]（保持传入顺序）
