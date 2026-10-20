"""Day14 第十二步：通过SSE主动推送后台任务状态。"""

import asyncio
import json

from aiohttp import web

import day14_task_persistent_api as base


# 每个正在被订阅的task_id对应一个事件通知器。
task_events: dict[str, asyncio.Event] = {}

# 保存原更新函数，再用带通知功能的版本替换。
original_update_job = base.update_job


async def notifying_update_job(task_id: str, **values: object) -> None:
    """先持久化状态，再主动通知订阅该任务的SSE协程。"""

    await original_update_job(task_id, **values)

    event = task_events.get(task_id)
    if event is not None:
        event.set()


# base.worker运行时会从其模块全局查找update_job，因此替换后自动获得通知能力。
base.update_job = notifying_update_job


async def write_event(
    response: web.StreamResponse,
    event_name: str,
    payload: dict
) -> None:
    text = (
        f"event: {event_name}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )
    await response.write(text.encode("utf-8"))


async def task_events_handler(request: web.Request) -> web.StreamResponse:
    """GET /tasks/{task_id}/events：持续推送状态，结束后关闭。"""

    task_id = request.match_info["task_id"]
    first_job = await base.fetch_job(task_id)

    if first_job is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "task_not_found"}),
            content_type="application/json"
        )

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
    await response.prepare(request)

    event = task_events.setdefault(task_id, asyncio.Event())
    last_status: str | None = None

    print(f"[SSE订阅开始] {task_id}")

    try:
        while True:
            # 先清除旧通知，再读取数据库，避免读取和等待之间丢失更新。
            event.clear()
            job = await base.fetch_job(task_id)

            if job is None:
                await write_event(
                    response,
                    "error",
                    {"task_id": task_id, "error": "task_not_found"}
                )
                break

            if job["status"] != last_status:
                await write_event(response, "status", job)
                last_status = job["status"]

            if job["status"] in {
                "succeeded", "failed", "cancelled"
            }:
                await write_event(
                    response,
                    "done",
                    {"task_id": task_id, "status": job["status"]}
                )
                break

            try:
                # 平时由Worker的event.set()唤醒；15秒无变化时发送心跳。
                await asyncio.wait_for(event.wait(), timeout=15)
            except TimeoutError:
                await write_event(
                    response,
                    "heartbeat",
                    {"task_id": task_id}
                )

    except (ConnectionResetError, asyncio.CancelledError):
        print(f"[SSE客户端断开] {task_id}")
    finally:
        task_events.pop(task_id, None)
        print(f"[SSE订阅结束] {task_id}")

    return response


# 更具体的/events路由必须在服务启动前注册。
base.app.router.add_get(
    "/tasks/{task_id}/events",
    task_events_handler
)


if __name__ == "__main__":
    print("Day14任务状态SSE服务已启动。")
    print("任务数据库：", base.DATABASE_PATH)
    print("状态推送：GET /tasks/{task_id}/events")
    web.run_app(base.app, host="127.0.0.1", port=8000)


# Worker → update_job() → SQLite commit → Event.set()
# SSE协程 → Event.wait()被唤醒 → 读取SQLite → 推送status事件
