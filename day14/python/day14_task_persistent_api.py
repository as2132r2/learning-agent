"""Day14 第十步：使用aiosqlite持久化异步后台任务。"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import aiosqlite
from aiohttp import web
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


load_dotenv()

WORKER_COUNT = 2
DAY14_DIRECTORY = Path(__file__).resolve().parents[1]
DATABASE_PATH = DAY14_DIRECTORY / "database" / "day14_tasks.db"
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

job_queue: asyncio.Queue[str] = asyncio.Queue()
worker_tasks: list[asyncio.Task] = []
running_operations: dict[str, asyncio.Task] = {}
database: aiosqlite.Connection | None = None
shutting_down = False


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


def get_database() -> aiosqlite.Connection:
    if database is None:
        raise RuntimeError("数据库尚未初始化。")
    return database


async def fetch_job(task_id: str) -> dict | None:
    """按task_id读取一条任务记录。"""

    db = get_database()
    cursor = await db.execute(
        """
        SELECT task_id, question, status, answer, error,
               created_at, started_at, finished_at
        FROM background_jobs
        WHERE task_id = ?
        """,
        (task_id,)
    )
    row = await cursor.fetchone()
    await cursor.close()
    return dict(row) if row is not None else None


async def update_job(task_id: str, **values: object) -> None:
    """使用参数化SQL更新指定字段。字段名来自服务端代码。"""

    if not values:
        return

    db = get_database()
    assignments = ", ".join(f"{name} = ?" for name in values)
    parameters = [*values.values(), task_id]
    await db.execute(
        f"UPDATE background_jobs SET {assignments} WHERE task_id = ?",
        parameters
    )
    await db.commit()


async def worker(name: str) -> None:
    print(f"[{name}] 已启动")

    while True:
        task_id = await job_queue.get()

        try:
            job = await fetch_job(task_id)
            if job is None or job["status"] == "cancelled":
                continue

            await update_job(
                task_id,
                status="running",
                started_at=now_text(),
                finished_at=None,
                error=None
            )
            print(f"[{name}] 开始 {task_id}")

            operation = asyncio.create_task(
                llm.ainvoke([
                    HumanMessage(content="请简洁回答：" + job["question"])
                ]),
                name=f"model-{task_id}"
            )
            running_operations[task_id] = operation

            try:
                response = await operation
                await update_job(
                    task_id,
                    status="succeeded",
                    answer=str(response.content),
                    error=None,
                    finished_at=now_text()
                )
            except asyncio.CancelledError:
                if shutting_down:
                    await update_job(
                        task_id,
                        status="pending",
                        started_at=None,
                        finished_at=None,
                        error=None
                    )
                else:
                    await update_job(
                        task_id,
                        status="cancelled",
                        error="任务被用户取消。",
                        finished_at=now_text()
                    )
            except Exception as error:
                await update_job(
                    task_id,
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                    finished_at=now_text()
                )
            finally:
                running_operations.pop(task_id, None)

            final_job = await fetch_job(task_id)
            print(f"[{name}] 结束 {task_id} | {final_job['status']}")

        finally:
            job_queue.task_done()


async def create_task_handler(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    question = data.get("question") if isinstance(data, dict) else None
    if not isinstance(question, str) or not question.strip():
        return web.json_response(
            {"error": "question必须是非空字符串。"},
            status=422
        )

    question = question.strip()
    task_id = f"task-{uuid4().hex[:12]}"
    created_at = now_text()
    db = get_database()

    await db.execute(
        """
        INSERT INTO background_jobs(
            task_id, question, status, created_at
        ) VALUES (?, ?, 'pending', ?)
        """,
        (task_id, question, created_at)
    )
    await db.commit()
    await job_queue.put(task_id)

    return web.json_response(await fetch_job(task_id), status=202)


async def get_task_handler(request: web.Request) -> web.Response:
    job = await fetch_job(request.match_info["task_id"])
    if job is None:
        return web.json_response({"error": "task_not_found"}, status=404)
    return web.json_response(job)


async def cancel_task_handler(request: web.Request) -> web.Response:
    task_id = request.match_info["task_id"]
    job = await fetch_job(task_id)

    if job is None:
        return web.json_response({"error": "task_not_found"}, status=404)

    if job["status"] not in {"pending", "running"}:
        return web.json_response(
            {"error": "task_already_finished", "task": job},
            status=409
        )

    if job["status"] == "pending":
        await update_job(
            task_id,
            status="cancelled",
            error="任务在开始前被用户取消。",
            finished_at=now_text()
        )
    else:
        operation = running_operations.get(task_id)
        if operation is not None:
            operation.cancel()

    return web.json_response(await fetch_job(task_id))


async def health_handler(_request: web.Request) -> web.Response:
    db = get_database()
    cursor = await db.execute("SELECT COUNT(*) AS count FROM background_jobs")
    row = await cursor.fetchone()
    await cursor.close()
    return web.json_response({
        "status": "ok",
        "database": str(DATABASE_PATH),
        "saved_tasks": row["count"],
        "queue_size": job_queue.qsize(),
        "running_tasks": len(running_operations)
    })


async def on_startup(_app: web.Application) -> None:
    """初始化数据库，恢复未完成任务，再启动Worker。"""

    global database
    database = await aiosqlite.connect(DATABASE_PATH)
    database.row_factory = aiosqlite.Row

    await database.execute("PRAGMA journal_mode=WAL")
    await database.execute(
        """
        CREATE TABLE IF NOT EXISTS background_jobs(
            task_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            status TEXT NOT NULL,
            answer TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        )
        """
    )

    # 进程退出时的running任务已经不再运行，重置后重新处理。
    await database.execute(
        """
        UPDATE background_jobs
        SET status = 'pending', started_at = NULL,
            finished_at = NULL, error = NULL
        WHERE status IN ('pending', 'running')
        """
    )
    await database.commit()

    cursor = await database.execute(
        """
        SELECT task_id FROM background_jobs
        WHERE status = 'pending'
        ORDER BY created_at
        """
    )
    rows = await cursor.fetchall()
    await cursor.close()

    for row in rows:
        await job_queue.put(row["task_id"])

    print(f"从SQLite恢复了{len(rows)}个未完成任务。")

    for index in range(1, WORKER_COUNT + 1):
        worker_tasks.append(asyncio.create_task(worker(f"worker-{index}")))


async def on_cleanup(_app: web.Application) -> None:
    global shutting_down
    shutting_down = True

    for operation in running_operations.values():
        operation.cancel()
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)

    if database is not None:
        await database.close()
    print("Worker和SQLite连接已关闭。")


app = web.Application(client_max_size=10_000)
app.router.add_get("/health", health_handler)
app.router.add_post("/tasks", create_task_handler)
app.router.add_get("/tasks/{task_id}", get_task_handler)
app.router.add_delete("/tasks/{task_id}", cancel_task_handler)
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)


if __name__ == "__main__":
    print("任务数据库：", DATABASE_PATH)
    web.run_app(app, host="127.0.0.1", port=8000)


# SQLite是任务状态的事实来源；Queue只是当前进程中的调度结构。
# 重启时：SQLite pending/running → 重置pending → 重新放入Queue。
