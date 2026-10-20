"""Day14 第四步：使用Semaphore限制模型请求的最大并发数。"""

import asyncio
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()

MAX_CONCURRENCY = 2

QUESTIONS = [
    "一句话解释async。",
    "一句话解释await。",
    "一句话解释Task。",
    "一句话解释gather。",
    "一句话解释Semaphore。"
]


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
    max_tokens=80,
    timeout=30,
    max_retries=1
)


@dataclass
class LimitedResult:
    index: int
    question: str
    answer: str
    elapsed_seconds: float


async def ask_with_limit(
    index: int,
    question: str,
    semaphore: asyncio.Semaphore,
    active_counter: list[int]
) -> LimitedResult:
    """先等待并发许可，再调用模型。"""

    print(f"[任务{index}排队] 等待Semaphore许可")

    async with semaphore:
        active_counter[0] += 1
        current_active = active_counter[0]

        print(
            f"[任务{index}获得许可] "
            f"当前正在调用模型的任务数={current_active}"
        )

        start_time = time.perf_counter()

        try:
            response = await llm.ainvoke([
                SystemMessage(
                    content="你是Python老师，只用一句简短中文回答。"
                ),
                HumanMessage(content=question)
            ])
        finally:
            active_counter[0] -= 1

        elapsed = time.perf_counter() - start_time
        print(
            f"[任务{index}释放许可] 耗时={elapsed:.2f}秒，"
            f"剩余活动任务数={active_counter[0]}"
        )

        return LimitedResult(
            index=index,
            question=question,
            answer=str(response.content),
            elapsed_seconds=elapsed
        )


async def main() -> None:
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # 列表是可变对象，所有协程共享其中的计数值，仅用于观察演示。
    active_counter = [0]

    print(f"任务总数：{len(QUESTIONS)}")
    print(f"最大并发数：{MAX_CONCURRENCY}\n")

    total_start = time.perf_counter()

    tasks = [
        asyncio.create_task(
            ask_with_limit(
                index=index,
                question=question,
                semaphore=semaphore,
                active_counter=active_counter
            ),
            name=f"limited-task-{index}"
        )
        for index, question in enumerate(QUESTIONS, start=1)
    ]

    results = await asyncio.gather(*tasks)
    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    print("所有结果：")

    for result in results:
        print(f"\n任务{result.index}：{result.question}")
        print("回答：", result.answer)
        print(f"模型调用耗时：{result.elapsed_seconds:.2f}秒")

    print("\n" + "=" * 70)
    print(f"总耗时：{total_elapsed:.2f}秒")
    print(f"整个过程同时调用模型的任务不会超过{MAX_CONCURRENCY}个。")


if __name__ == "__main__":
    asyncio.run(main())


# ============================================================
# 调度关系
# ============================================================
# 创建Task1～Task5（五个任务都存在）
#             ↓
# Semaphore(2)只有两个许可
#             ↓
# 第一批：Task1、Task2调用模型；Task3～Task5排队
#             ↓ 某任务完成并释放许可
# 第二批：排队任务获得许可并开始调用
#             ↓
# 任意时刻实际模型请求数 <= 2
