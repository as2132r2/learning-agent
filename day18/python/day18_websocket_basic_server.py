"""Day18 第一步（服务端）：WebSocket 基础双向通信。

运行方式：
    python day18/python/day18_websocket_basic_server.py

服务地址：
    http://127.0.0.1:8010/ws

本步骤不调用大模型，只演示：
    1. HTTP 请求升级为 WebSocket；
    2. 客户端可以连续向服务端发消息；
    3. 服务端也可以在同一条连接上连续回复；
    4. 收到 close 后正确结束连接。
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from aiohttp import WSMsgType, web


HOST = "127.0.0.1"
PORT = 8010


async def health_handler(request: web.Request) -> web.Response:
    """普通 HTTP 健康检查。"""

    return web.json_response(
        {
            "status": "ok",
            "service": "day18-websocket-basic-server",
            "websocket_path": "/ws",
        }
    )


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """建立 WebSocket，并在同一连接上循环收发消息。"""

    connection_id = "conn-" + uuid.uuid4().hex[:10]

    # WebSocketResponse 表示即将把普通 HTTP 请求升级为 WebSocket。
    websocket = web.WebSocketResponse(
        heartbeat=20,
        receive_timeout=60,
    )

    # prepare() 完成 HTTP Upgrade 握手。
    await websocket.prepare(request)

    print(f"[连接建立] connection_id={connection_id}")

    # 服务端可以主动向刚连接的客户端发送欢迎消息。
    await websocket.send_json(
        {
            "type": "connected",
            "connection_id": connection_id,
            "message": "WebSocket 连接已经建立。",
        }
    )

    message_count = 0

    try:
        # WebSocketResponse 是异步迭代器，会持续等待客户端消息。
        async for message in websocket:
            if message.type == WSMsgType.TEXT:
                message_count += 1
                raw_text: str = message.data

                print(
                    f"[收到文本] connection_id={connection_id} "
                    f"number={message_count} data={raw_text!r}"
                )

                try:
                    client_data: Any = json.loads(raw_text)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": "invalid_json",
                            "message": "客户端消息必须是 JSON。",
                        }
                    )
                    continue

                if not isinstance(client_data, dict):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": "invalid_message",
                            "message": "JSON 顶层必须是 object。",
                        }
                    )
                    continue

                client_type = client_data.get("type")

                if client_type == "echo":
                    text = str(client_data.get("text", ""))

                    await websocket.send_json(
                        {
                            "type": "echo_result",
                            "connection_id": connection_id,
                            "message_number": message_count,
                            "original_text": text,
                            "upper_text": text.upper(),
                        }
                    )

                elif client_type == "status":
                    await websocket.send_json(
                        {
                            "type": "status_result",
                            "connection_id": connection_id,
                            "received_messages": message_count,
                            "closed": websocket.closed,
                        }
                    )

                elif client_type == "close":
                    await websocket.send_json(
                        {
                            "type": "closing",
                            "connection_id": connection_id,
                            "message": "服务端准备关闭 WebSocket。",
                        }
                    )
                    await websocket.close(
                        code=1000,
                        message=b"normal closure",
                    )
                    break

                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": "unknown_type",
                            "message": (
                                "type 只能是 echo、status 或 close。"
                            ),
                        }
                    )

            elif message.type == WSMsgType.ERROR:
                print(
                    f"[WebSocket 错误] connection_id={connection_id} "
                    f"error={websocket.exception()}"
                )
                break

            elif message.type in (
                WSMsgType.CLOSE,
                WSMsgType.CLOSED,
                WSMsgType.CLOSING,
            ):
                break

    finally:
        print(
            f"[连接结束] connection_id={connection_id} "
            f"messages={message_count}"
        )

        if not websocket.closed:
            await websocket.close()

    return websocket


@web.middleware
async def logging_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    """记录普通 HTTP 请求和 WebSocket 握手请求。"""

    print(f"[HTTP] {request.method} {request.path_qs}")
    return await handler(request)


def create_app() -> web.Application:
    """创建 aiohttp 应用并注册路由。"""

    app = web.Application(middlewares=[logging_middleware])
    app.router.add_get("/health", health_handler)
    app.router.add_get("/ws", websocket_handler)
    return app


def main() -> None:
    """启动基础 WebSocket Server。"""

    print("Day18 WebSocket 基础服务正在启动。")
    print(f"HTTP 地址：http://{HOST}:{PORT}")
    print(f"WebSocket 地址：ws://{HOST}:{PORT}/ws")
    print("按 Ctrl+C 停止服务。")

    web.run_app(
        create_app(),
        host=HOST,
        port=PORT,
        print=None,
    )


if __name__ == "__main__":
    main()
