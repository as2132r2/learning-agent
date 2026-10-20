"""Day17 第十三步（服务端）：SSE Last-Event-ID 断线续传基础。

运行方式：
    python day17/python/day17_sse_resume_server.py

服务地址：
    http://127.0.0.1:8009/events

为了把重连机制演示清楚，本步骤暂时不调用大模型，而是发送编号 1～8 的
确定性事件。每条 HTTP 连接最多发送 3 个事件后主动关闭，客户端必须重连。

客户端重连时发送：
    Last-Event-ID: 3

服务端就从事件 4 继续发送。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp import web


HOST = "127.0.0.1"
PORT = 8009
TOTAL_EVENTS = 8
EVENTS_PER_CONNECTION = 3
RETRY_MILLISECONDS = 500


def encode_sse_event(
    *,
    event_id: int,
    event: str,
    data: dict[str, Any],
    retry: int | None = None,
) -> bytes:
    """编码包含 id、event、data 和可选 retry 的 SSE 事件。"""

    lines = [
        f"id: {event_id}",
        f"event: {event}",
    ]

    if retry is not None:
        lines.append(f"retry: {retry}")

    lines.append("data: " + json.dumps(data, ensure_ascii=False))
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def parse_last_event_id(request: web.Request) -> int:
    """读取 Last-Event-ID 请求头，并转换为非负整数。"""

    raw_value = request.headers.get("Last-Event-ID", "0").strip()

    if raw_value == "":
        return 0

    try:
        value = int(raw_value)
    except ValueError as error:
        raise web.HTTPBadRequest(
            text=json.dumps(
                {"error": "Last-Event-ID 必须是整数。"},
                ensure_ascii=False,
            ),
            content_type="application/json",
        ) from error

    if value < 0 or value > TOTAL_EVENTS:
        raise web.HTTPBadRequest(
            text=json.dumps(
                {
                    "error": (
                        f"Last-Event-ID 必须在 0～{TOTAL_EVENTS} 之间。"
                    )
                },
                ensure_ascii=False,
            ),
            content_type="application/json",
        )

    return value


async def health_handler(request: web.Request) -> web.Response:
    """返回本次重连演示的配置。"""

    return web.json_response(
        {
            "status": "ok",
            "service": "day17-sse-resume-server",
            "total_events": TOTAL_EVENTS,
            "events_per_connection": EVENTS_PER_CONNECTION,
            "retry_milliseconds": RETRY_MILLISECONDS,
        }
    )


async def events_handler(request: web.Request) -> web.StreamResponse:
    """从 Last-Event-ID 的下一条开始，发送一小批编号事件。"""

    last_event_id = parse_last_event_id(request)
    next_event_id = last_event_id + 1

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    print(
        f"[新连接] Last-Event-ID={last_event_id} "
        f"将从 id={next_event_id} 开始"
    )

    sent_count = 0

    try:
        while (
            next_event_id <= TOTAL_EVENTS
            and sent_count < EVENTS_PER_CONNECTION
        ):
            await response.write(
                encode_sse_event(
                    event_id=next_event_id,
                    event="number",
                    retry=RETRY_MILLISECONDS,
                    data={
                        "number": next_event_id,
                        "message": f"这是第 {next_event_id} 个事件",
                    },
                )
            )

            print(f"[发送事件] id={next_event_id}")
            sent_count += 1
            next_event_id += 1
            await asyncio.sleep(0.2)

        if next_event_id > TOTAL_EVENTS:
            # complete 使用下一个 ID，表示这也是可追踪的最后一条事件。
            await response.write(
                encode_sse_event(
                    event_id=TOTAL_EVENTS + 1,
                    event="complete",
                    data={
                        "total_events": TOTAL_EVENTS,
                        "message": "所有编号事件已经发送完成。",
                    },
                )
            )
            print("[全部完成] 发送 complete 事件")
        else:
            print(
                f"[模拟断线] 本连接发送 {sent_count} 条后主动关闭，"
                f"下一条应为 id={next_event_id}"
            )

    except ConnectionResetError:
        print("[真实断线] 客户端提前关闭连接")

    finally:
        try:
            await response.write_eof()
        except (ConnectionResetError, RuntimeError):
            pass

    return response


@web.middleware
async def logging_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    """显示请求方法、路径和续传请求头。"""

    print(
        f"[HTTP] {request.method} {request.path_qs} "
        f"Last-Event-ID={request.headers.get('Last-Event-ID')!r}"
    )
    return await handler(request)


def create_app() -> web.Application:
    """创建断线续传演示服务。"""

    app = web.Application(middlewares=[logging_middleware])
    app.router.add_get("/health", health_handler)
    app.router.add_get("/events", events_handler)
    return app


def main() -> None:
    """启动 8009 端口的 SSE 续传服务。"""

    print("Day17 SSE Last-Event-ID 服务正在启动。")
    print(f"事件地址：http://{HOST}:{PORT}/events")
    print("事件总数：", TOTAL_EVENTS)
    print("每条连接最多发送：", EVENTS_PER_CONNECTION)
    print("建议重连等待：", RETRY_MILLISECONDS, "毫秒")
    print("按 Ctrl+C 停止服务。")

    web.run_app(
        create_app(),
        host=HOST,
        port=PORT,
        print=None,
    )


if __name__ == "__main__":
    main()
