"""Day15 第四步：带用户所有权隔离的异步任务API。"""

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

from day15_auth_api import authentication_middleware


load_dotenv()

WORKER_COUNT = 2
DATABASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "day15_user_tasks.db"
)
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

job_queue: asyncio.Queue[str] = asyncio.Queue()
workers: list[asyncio.Task] = []
database: aiosqlite.Connection | None = None


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


def db() -> aiosqlite.Connection:
    if database is None:
        raise RuntimeError("数据库尚未初始化。")
    return database


async def fetch_owned_task(task_id: str, user_id: str) -> dict | None:
    """task_id和user_id必须同时匹配。"""

    cursor = await db().execute(
        """
        SELECT task_id, user_id, question, status, answer, error,
               created_at, started_at, finished_at
        FROM user_tasks
        WHERE task_id = ? AND user_id = ?
        """,
        (task_id, user_id)
    )
    row = await cursor.fetchone()
    await cursor.close()
    return dict(row) if row is not None else None


async def worker(name: str) -> None:
    print(f"[{name}] 已启动")

    while True:
        task_id = await job_queue.get()

        try:
            cursor = await db().execute(
                "SELECT question FROM user_tasks WHERE task_id = ?",
                (task_id,)
            )
            row = await cursor.fetchone()
            await cursor.close()

            if row is None:
                continue

            await db().execute(
                """
                UPDATE user_tasks
                SET status = 'running', started_at = ?, error = NULL
                WHERE task_id = ?
                """,
                (now_text(), task_id)
            )
            await db().commit()

            try:
                response = await llm.ainvoke([
                    HumanMessage(content="请简洁回答：" + row["question"])
                ])
                await db().execute(
                    """
                    UPDATE user_tasks
                    SET status = 'succeeded', answer = ?, finished_at = ?
                    WHERE task_id = ?
                    """,
                    (str(response.content), now_text(), task_id)
                )
            except Exception as error:
                await db().execute(
                    """
                    UPDATE user_tasks
                    SET status = 'failed', error = ?, finished_at = ?
                    WHERE task_id = ?
                    """,
                    (f"{type(error).__name__}: {error}", now_text(), task_id)
                )

            await db().commit()
            print(f"[{name}] 完成 {task_id}")

        finally:
            job_queue.task_done()


async def create_task_handler(request: web.Request) -> web.Response:
    identity = request["identity"]
    user_id = identity["sub"]

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

    task_id = f"task-{uuid4().hex[:12]}"
    await db().execute(
        """
        INSERT INTO user_tasks(
            task_id, user_id, question, status, created_at
        ) VALUES (?, ?, ?, 'pending', ?)
        """,
        (task_id, user_id, question.strip(), now_text())
    )
    await db().commit()
    await job_queue.put(task_id)

    task = await fetch_owned_task(task_id, user_id)
    return web.json_response(task, status=202)


async def get_task_handler(request: web.Request) -> web.Response:
    user_id = request["identity"]["sub"]
    task = await fetch_owned_task(request.match_info["task_id"], user_id)

    if task is None:
        return web.json_response({"error": "task_not_found"}, status=404)

    return web.json_response(task)


async def list_tasks_handler(request: web.Request) -> web.Response:
    """只列出当前令牌所属用户的任务。"""

    user_id = request["identity"]["sub"]
    cursor = await db().execute(
        """
        SELECT task_id, user_id, question, status, answer, error,
               created_at, started_at, finished_at
        FROM user_tasks
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 100
        """,
        (user_id,)
    )
    rows = await cursor.fetchall()
    await cursor.close()

    return web.json_response({
        "user_id": user_id,
        "count": len(rows),
        "tasks": [dict(row) for row in rows]
    })


async def health_handler(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def on_startup(_app: web.Application) -> None:
    global database
    database = await aiosqlite.connect(DATABASE_PATH)
    database.row_factory = aiosqlite.Row

    await database.execute("PRAGMA journal_mode=WAL")
    await database.execute(
        """
        CREATE TABLE IF NOT EXISTS user_tasks(
            task_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
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
    await database.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_tasks_user_id "
        "ON user_tasks(user_id, created_at)"
    )
    await database.commit()

    for index in range(1, WORKER_COUNT + 1):
        workers.append(asyncio.create_task(worker(f"worker-{index}")))


async def on_cleanup(_app: web.Application) -> None:
    for task in workers:
        task.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    if database is not None:
        await database.close()


app = web.Application(middlewares=[authentication_middleware])
app.router.add_get("/health", health_handler)
app.router.add_post("/tasks", create_task_handler)
app.router.add_get("/tasks", list_tasks_handler)
app.router.add_get("/tasks/{task_id}", get_task_handler)
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)


if __name__ == "__main__":
    print("Day15用户隔离任务API已启动。")
    print("数据库：", DATABASE_PATH)
    web.run_app(app, host="127.0.0.1", port=8000)


# Token.sub → request identity → user_id写入数据库
# 查询始终使用 WHERE task_id=? AND user_id=?
