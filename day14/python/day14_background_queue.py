"""Day14 第七步：使用asyncio.Queue实现后台模型任务队列。"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


load_dotenv()

WORKER_COUNT = 2
TaskStatus = Literal["pending", "running", "succeeded", "failed"]


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
class BackgroundJob:
    task_id: str
    question: str
    status: TaskStatus = "pending"
    answer: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=now_text)
    started_at: str | None = None
    finished_at: str | None = None


# Queue保存等待处理的task_id；字典保存每个任务的完整状态。
job_queue: asyncio.Queue[str | None] = asyncio.Queue()
job_store: dict[str, BackgroundJob] = {}


async def submit_job(question: str) -> BackgroundJob:
    """创建任务记录并放入队列，不等待模型回答。"""

    task_id = f"task-{uuid4().hex[:12]}"
    job = BackgroundJob(task_id=task_id, question=question)

    job_store[task_id] = job
    await job_queue.put(task_id)

    print(f"[任务已提交] {task_id} | status={job.status}")
    return job


async def worker(worker_name: str) -> None:
    """持续从Queue取任务并调用模型。None表示停止信号。"""

    print(f"[{worker_name}] 已启动")

    while True:
        task_id = await job_queue.get()

        try:
            if task_id is None:
                print(f"[{worker_name}] 收到停止信号")
                return

            job = job_store[task_id]
            job.status = "running"
            job.started_at = now_text()
            print(f"[{worker_name}] 开始处理 {task_id}")

            try:
                response = await llm.ainvoke([
                    HumanMessage(
                        content=(
                            "请只用一句简短中文回答：" + job.question
                        )
                    )
                ])
                job.answer = str(response.content)
                job.status = "succeeded"

            except Exception as error:
                job.error = f"{type(error).__name__}: {error}"
                job.status = "failed"

            finally:
                job.finished_at = now_text()

            print(
                f"[{worker_name}] 完成 {task_id} | "
                f"status={job.status}"
            )

        finally:
            # 每次get()都必须对应一次task_done()，包括停止信号。
            job_queue.task_done()


def print_status_snapshot() -> None:
    """显示所有任务的当前状态。"""

    status_text = " | ".join(
        f"{job.task_id}={job.status}"
        for job in job_store.values()
    )
    print("[状态查询]", status_text)


async def main() -> None:
    print(f"启动{WORKER_COUNT}个后台Worker。\n")

    workers = [
        asyncio.create_task(
            worker(f"worker-{index}"),
            name=f"worker-{index}"
        )
        for index in range(1, WORKER_COUNT + 1)
    ]

    questions = [
        "什么是后台任务？",
        "什么是任务队列？",
        "Worker在任务队列中负责什么？",
        "为什么提交任务后要返回task_id？"
    ]

    print("提交任务，但不在submit_job()中等待模型回答：")
    for question in questions:
        await submit_job(question)

    print("\n所有任务已经快速提交，后台Worker正在处理。\n")

    # 定期查询，直到Queue中的四个业务任务全部被task_done()确认。
    queue_finished = asyncio.create_task(job_queue.join())

    while not queue_finished.done():
        print_status_snapshot()
        await asyncio.sleep(0.3)

    await queue_finished
    print_status_snapshot()

    print("\n" + "=" * 70)
    print("最终结果：")

    for job in job_store.values():
        print(f"\n任务ID：{job.task_id}")
        print("问题：", job.question)
        print("状态：", job.status)
        print("回答：", job.answer)
        print("错误：", job.error)
        print("创建时间：", job.created_at)
        print("开始时间：", job.started_at)
        print("结束时间：", job.finished_at)

    # 每个Worker发送一个停止信号，并等待它们正常退出。
    for _ in workers:
        await job_queue.put(None)

    await job_queue.join()
    await asyncio.gather(*workers)
    print("\n所有后台Worker已安全停止。")


if __name__ == "__main__":
    asyncio.run(main())


# submit_job() → task_id进入Queue → 立即返回pending任务
#                                  ↓
# 两个worker并发get() → running → ainvoke() → succeeded/failed
#                                  ↓
# job_store[task_id]保存最新状态和结果
