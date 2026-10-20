"""Day14 第五步：异步任务的超时控制与异常隔离。"""

import asyncio
import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


load_dotenv()

MAX_CONCURRENCY = 2
JobMode = Literal["model", "timeout", "error"]
JobStatus = Literal["success", "timeout", "error"]


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
    max_tokens=60,
    timeout=30,
    max_retries=0
)


@dataclass
class Job:
    index: int
    mode: JobMode
    question: str
    timeout_seconds: float


@dataclass
class JobResult:
    index: int
    status: JobStatus
    answer: str | None = None
    error_type: str | None = None
    error_message: str | None = None


JOBS = [
    Job(1, "model", "一句话说明异步超时有什么用。", 10.0),
    Job(2, "timeout", "模拟一个执行时间过长的任务。", 0.5),
    Job(3, "error", "模拟一个立即失败的任务。", 2.0),
    Job(4, "model", "一句话说明异常隔离有什么用。", 10.0)
]


async def execute_job(job: Job) -> str:
    """执行任务；两种模拟模式用于稳定制造故障。"""

    if job.mode == "timeout":
        print(f"[任务{job.index}] 模拟等待2秒")
        await asyncio.sleep(2)
        return "模拟等待完成"

    if job.mode == "error":
        print(f"[任务{job.index}] 主动抛出模拟异常")
        raise ValueError("这是用于学习异常隔离的模拟错误。")

    print(f"[任务{job.index}] 开始调用模型")
    response = await llm.ainvoke([
        HumanMessage(content=job.question)
    ])
    return str(response.content)


async def run_safely(
    job: Job,
    semaphore: asyncio.Semaphore
) -> JobResult:
    """限制并发，并把成功、超时、错误统一转换成JobResult。"""

    print(f"[任务{job.index}排队] mode={job.mode}")

    async with semaphore:
        print(
            f"[任务{job.index}开始] "
            f"timeout={job.timeout_seconds}秒"
        )

        try:
            answer = await asyncio.wait_for(
                execute_job(job),
                timeout=job.timeout_seconds
            )

            print(f"[任务{job.index}成功]")
            return JobResult(
                index=job.index,
                status="success",
                answer=answer
            )

        except TimeoutError:
            print(f"[任务{job.index}超时]")
            return JobResult(
                index=job.index,
                status="timeout",
                error_type="TimeoutError",
                error_message=(
                    f"任务超过{job.timeout_seconds}秒，已取消等待。"
                )
            )

        except Exception as error:
            print(
                f"[任务{job.index}失败] "
                f"{type(error).__name__}: {error}"
            )
            return JobResult(
                index=job.index,
                status="error",
                error_type=type(error).__name__,
                error_message=str(error)
            )


async def main() -> None:
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = [
        asyncio.create_task(
            run_safely(job, semaphore),
            name=f"safe-job-{job.index}"
        )
        for job in JOBS
    ]

    # 每个任务内部已经把异常转换为JobResult，因此gather可以正常返回全部结果。
    results = await asyncio.gather(*tasks)

    print("\n" + "=" * 70)
    print("汇总结果：")

    for result in results:
        print(f"\n任务{result.index}：{result.status}")
        if result.status == "success":
            print("回答：", result.answer)
        else:
            print("错误类型：", result.error_type)
            print("错误信息：", result.error_message)

    success_count = sum(r.status == "success" for r in results)
    timeout_count = sum(r.status == "timeout" for r in results)
    error_count = sum(r.status == "error" for r in results)

    print("\n" + "=" * 70)
    print("成功数量：", success_count)
    print("超时数量：", timeout_count)
    print("错误数量：", error_count)
    print("一个任务失败或超时，没有阻止其他任务完成。")


if __name__ == "__main__":
    asyncio.run(main())


# asyncio.wait_for()：为单个协程设置超时并在超时时取消它
# run_safely()：把异常转换成数据，不让异常逃出任务
# asyncio.gather()：最终获得每个任务各自的JobResult
