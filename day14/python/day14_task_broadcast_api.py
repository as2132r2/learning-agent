"""Day14 第十三步：用每订阅者独立Queue广播任务状态。"""

import asyncio
import json

from aiohttp import web

import day14_task_persistent_api as base


# task_id → 订阅该任务的多个独立Queue。
subscribers: dict[str, set[asyncio.Queue[dict]]] = {}
original_update_job = base.update_job


async def broadcasting_update_job(task_id: str, **values: object) -> None:
    """状态落库后，把最新任务快照广播给所有订阅者。"""

    await original_update_job(task_id, **values)
    job = await base.fetch_job(task_id)

    if job is None:
        return

    for subscriber_queue in subscribers.get(task_id, set()).copy():
        subscriber_queue.put_nowait(job)


base.update_job = broadcasting_update_job


async def write_event(
    response: web.StreamResponse,
    event_name: str,
    payload: dict
) -> None:
    event_text = (
        f"event: {event_name}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )
    await response.write(event_text.encode("utf-8"))


async def broadcast_events_handler(
    request: web.Request
) -> web.StreamResponse:
    """每个连接创建自己的Queue，互不抢夺通知。"""

    task_id = request.match_info["task_id"]
    initial_job = await base.fetch_job(task_id)

    if initial_job is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "task_not_found"}),
            content_type="application/json"
        )

    subscriber_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=20)
    task_subscribers = subscribers.setdefault(task_id, set())
    task_subscribers.add(subscriber_queue)

    response = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
    await response.prepare(request)

    subscriber_number = len(task_subscribers)
    print(
        f"[广播订阅加入] {task_id} | "
        f"当前订阅数={subscriber_number}"
    )

    try:
        # 新订阅者首先获得数据库中的当前快照。
        await write_event(response, "status", initial_job)

        if initial_job["status"] in {
            "succeeded", "failed", "cancelled"
        }:
            await write_event(
                response,
                "done",
                {
                    "task_id": task_id,
                    "status": initial_job["status"]
                }
            )
            return response

        while True:
            try:
                job = await asyncio.wait_for(
                    subscriber_queue.get(),
                    timeout=15
                )
            except TimeoutError:
                await write_event(
                    response,
                    "heartbeat",
                    {"task_id": task_id}
                )
                continue

            await write_event(response, "status", job)

            if job["status"] in {
                "succeeded", "failed", "cancelled"
            }:
                await write_event(
                    response,
                    "done",
                    {"task_id": task_id, "status": job["status"]}
                )
                break

    except (ConnectionResetError, asyncio.CancelledError):
        print(f"[广播订阅断开] {task_id}")
    finally:
        task_subscribers.discard(subscriber_queue)

        if not task_subscribers:
            subscribers.pop(task_id, None)

        print(
            f"[广播订阅移除] {task_id} | "
            f"剩余订阅数={len(task_subscribers)}"
        )

    return response


async def subscriber_health_handler(_request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "subscribed_task_count": len(subscribers),
        "connection_count": sum(
            len(task_subscribers)
            for task_subscribers in subscribers.values()
        )
    })


base.app.router.add_get(
    "/tasks/{task_id}/events",
    broadcast_events_handler
)
base.app.router.add_get(
    "/subscriber-health",
    subscriber_health_handler
)


if __name__ == "__main__":
    print("Day14多订阅者任务广播服务已启动。")
    print("任务数据库：", base.DATABASE_PATH)
    print("广播接口：GET /tasks/{task_id}/events")
    web.run_app(base.app, host="127.0.0.1", port=8000)


# Worker更新一次SQLite
#        ↓
# 遍历subscribers[task_id]
#        ├→ Queue-A → SSE客户端A
#        ├→ Queue-B → SSE客户端B
#        └→ Queue-C → SSE客户端C
