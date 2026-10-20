"""Day14 第八步：aiohttp异步后台任务API。"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import uuid4

from aiohttp import web
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


load_dotenv()

WORKER_COUNT = 2
MAX_QUESTION_LENGTH = 500
TaskStatus = Literal[
    "pending", "running", "succeeded", "failed", "cancelled"
]


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
    max_tokens=120,
    timeout=30,
    max_retries=1
)


@dataclass
class Job:
    task_id: str
    question: str
    status: TaskStatus = "pending"
    answer: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=now_text)
    started_at: str | None = None
    finished_at: str | None = None
    running_operation: asyncio.Task | None = field(
        default=None,
        repr=False
    )


job_queue: asyncio.Queue[str] = asyncio.Queue()
job_store: dict[str, Job] = {}
worker_tasks: list[asyncio.Task] = []


def job_to_dict(job: Job) -> dict:
    """排除不能JSON序列化的running_operation。"""

    return {
        "task_id": job.task_id,
        "question": job.question,
        "status": job.status,
        "answer": job.answer,
        "error": job.error,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at
    }


async def worker(name: str) -> None:
    print(f"[{name}] 已启动")

    while True:
        task_id = await job_queue.get()

        try:
            job = job_store[task_id]

            # pending阶段收到DELETE后，任务仍可能留在Queue中，但Worker会跳过。
            if job.status == "cancelled":
                print(f"[{name}] 跳过已取消任务 {task_id}")
                continue

            job.status = "running"
            job.started_at = now_text()
            print(f"[{name}] 开始 {task_id}")

            job.running_operation = asyncio.create_task(
                llm.ainvoke([
                    HumanMessage(
                        content="请简洁回答：" + job.question
                    )
                ]),
                name=f"model-{task_id}"
            )

            try:
                response = await job.running_operation
                job.answer = str(response.content)
                job.status = "succeeded"
            except asyncio.CancelledError:
                job.status = "cancelled"
                job.error = "任务被用户取消。"
            except Exception as error:
                job.status = "failed"
                job.error = f"{type(error).__name__}: {error}"
            finally:
                job.running_operation = None
                job.finished_at = now_text()

            print(f"[{name}] 结束 {task_id} | {job.status}")

        finally:
            job_queue.task_done()


async def create_task_handler(request: web.Request) -> web.Response:
    """POST /tasks：创建任务并立即返回202。"""

    try:
        data = await request.json()
    except Exception:
        return web.json_response(
            {"error": "invalid_json"},
            status=400
        )

    question = data.get("question") if isinstance(data, dict) else None
    if not isinstance(question, str) or not question.strip():
        return web.json_response(
            {"error": "question必须是非空字符串。"},
            status=422
        )

    question = question.strip()
    if len(question) > MAX_QUESTION_LENGTH:
        return web.json_response(
            {"error": "question不能超过500个字符。"},
            status=422
        )

    task_id = f"task-{uuid4().hex[:12]}"
    job = Job(task_id=task_id, question=question)
    job_store[task_id] = job
    await job_queue.put(task_id)

    print(f"[HTTP提交] {task_id}")
    return web.json_response(job_to_dict(job), status=202)


async def get_task_handler(request: web.Request) -> web.Response:
    """GET /tasks/{task_id}：读取任务最新状态。"""

    task_id = request.match_info["task_id"]
    job = job_store.get(task_id)

    if job is None:
        return web.json_response(
            {"error": "task_not_found", "task_id": task_id},
            status=404
        )

    return web.json_response(job_to_dict(job))


async def cancel_task_handler(request: web.Request) -> web.Response:
    """DELETE /tasks/{task_id}：取消pending或running任务。"""

    task_id = request.match_info["task_id"]
    job = job_store.get(task_id)

    if job is None:
        return web.json_response(
            {"error": "task_not_found", "task_id": task_id},
            status=404
        )

    if job.status in {"succeeded", "failed", "cancelled"}:
        return web.json_response(
            {
                "error": "task_already_finished",
                "task": job_to_dict(job)
            },
            status=409
        )

    if job.status == "pending":
        job.status = "cancelled"
        job.error = "任务在开始前被用户取消。"
        job.finished_at = now_text()

    elif job.running_operation is not None:
        job.running_operation.cancel()

    return web.json_response(job_to_dict(job))


async def health_handler(_request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "workers": WORKER_COUNT,
        "tasks": len(job_store),
        "queue_size": job_queue.qsize()
    })


async def on_startup(_app: web.Application) -> None:
    for index in range(1, WORKER_COUNT + 1):
        worker_tasks.append(
            asyncio.create_task(
                worker(f"worker-{index}"),
                name=f"worker-{index}"
            )
        )


async def on_cleanup(_app: web.Application) -> None:
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    print("所有Worker已停止。")


app = web.Application(client_max_size=10_000)
app.router.add_get("/health", health_handler)
app.router.add_post("/tasks", create_task_handler)
app.router.add_get("/tasks/{task_id}", get_task_handler)
app.router.add_delete("/tasks/{task_id}", cancel_task_handler)
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)


if __name__ == "__main__":
    web.run_app(app, host="127.0.0.1", port=8000)


# POST /tasks → Job(pending) → Queue → 202 + task_id
# Worker → running → ainvoke() → succeeded/failed/cancelled
# GET /tasks/{id} → 读取job_store
# DELETE /tasks/{id} → 标记取消或cancel()当前模型Task
